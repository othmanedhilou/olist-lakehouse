# Olist Lakehouse

Lakehouse e-commerce construit sur Azure Databricks à partir du jeu de données public
[Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) : environ 100 000 commandes
passées entre 2016 et 2018 sur une marketplace brésilienne, réparties dans 9 fichiers CSV.

Le pipeline suit une architecture médaillon (bronze / silver / gold) en PySpark et Delta Lake,
avec chargement incrémental, contrôles qualité bloquants et un modèle en étoile exploité dans Power BI.
Sur les tables du lakehouse, un modèle XGBoost prédit les commandes qui seront livrées en retard ;
l'entraînement, les métriques et les versions du modèle sont suivis dans MLflow.

## Architecture

```
CSV Olist ──> ADLS Gen2 (landing)
                 │
                 ▼
        bronze  : copie brute, tout en string, + _source / _batch_id / _ingested_at (append)
                 │
                 ▼
        silver  : typage, normalisation, dédoublonnage par clé métier,
                  contrôles qualité, MERGE Delta incrémental
                 │
                 ▼
        gold    : fact_sales + dim_customer / dim_product / dim_seller / dim_date
                  + kpi_monthly
                 │
                 ├──> ml_order_features ──> entraînement XGBoost (MLflow)
                 │                              │ registre Unity Catalog, alias champion
                 │                              ▼
                 │                     late_delivery_scores (commandes en cours)
                 ▼
             Power BI
```

| Couche | Tables | Écriture |
|---|---|---|
| bronze | 9 tables, une par fichier source | append |
| silver | orders, order_items, order_payments, order_reviews, customers, sellers, products, geolocation | MERGE sur la clé métier |
| gold | fact_sales, dim_customer, dim_product, dim_seller, dim_date, kpi_monthly | overwrite |

Quelques choix :

- **Bronze en string** : aucune ligne n'est perdue à l'ingestion à cause d'un type mal inféré, le typage se fait en silver où les erreurs sont visibles.
- **Dédoublonnage** : une même commande peut arriver dans plusieurs lots ; on garde la version la plus récente (`_ingested_at`) et le MERGE ne met à jour une ligne que si la source est plus récente.
- **Géolocalisation** : le fichier brut contient environ 1 million de points pour 19 000 préfixes de code postal ; silver ne garde qu'un centroïde par préfixe et écarte les coordonnées hors du Brésil.
- **Paiements** : ils sont au niveau commande alors que la table de faits est au niveau ligne ; le montant payé est réparti au prorata du prix de chaque ligne pour que les sommes restent justes.

## Contrôles qualité

Définis dans `src/olist_lakehouse/quality.py` et exécutés avant toute écriture en silver :

- non-nullité des clés et des colonnes obligatoires ;
- unicité des clés métier ;
- plages de valeurs (prix ≥ 0, note entre 1 et 5, délai de livraison entre 0 et 365 jours) ;
- valeurs autorisées (statuts de commande).

Un contrôle bloquant en échec arrête le job. Certains contrôles sont déclarés en avertissement seulement (par exemple les délais de livraison aberrants, qui existent dans les données d'origine).

## Prédiction des retards de livraison

Environ 8 % des commandes Olist sont livrées après la date promise au client, et ce sont elles qui
font chuter la note des avis. Le modèle estime, au moment de l'achat, la probabilité qu'une commande
arrive en retard, pour que le service client puisse prévenir ou prioriser les expéditions.

**Variables** (`src/olist_lakehouse/ml/features.py`), une ligne par commande :

- panier : nombre d'articles et de vendeurs, prix, fret, part du fret, poids et volume total ;
- trajet : distance client - vendeur (haversine entre les centroïdes de code postal), même État ou non ;
- promesse : nombre de jours entre l'achat et la date de livraison estimée ;
- contexte : mois, jour, heure d'achat, catégorie principale, mode de paiement, nombre d'échéances.

Seules les informations connues au moment de l'achat sont utilisées. Les dates de remise au
transporteur et de livraison ou la note d'avis sont exclues, un test le vérifie.

**Entraînement** (`src/olist_lakehouse/ml/train.py`) :

- découpage temporel : apprentissage sur les commandes les plus anciennes, évaluation sur les 20 % les plus récentes ;
- régression logistique comme référence, puis XGBoost avec pondération de la classe minoritaire ;
- métriques : ROC AUC, PR AUC (plus parlante quand la classe positive est rare), précision et rappel
  sur les 10 % de commandes jugées les plus risquées ;
- tout est enregistré dans MLflow : paramètres, métriques des deux modèles, importance des variables,
  prédictions sur le jeu de test, modèle avec sa signature ;
- le modèle est enregistré dans le registre (Unity Catalog sur Databricks) et reçoit l'alias
  `champion` seulement si sa PR AUC dépasse celle de la version en place.

**Scoring** : le job `score_orders.py` charge `models:/olist.ml.late_delivery@champion`, note les
commandes pas encore livrées et écrit `gold.late_delivery_scores` (probabilité et niveau de risque),
affichée dans une page Power BI.

En local, avec MLflow sur un fichier SQLite :

```bash
python scripts/make_sample_data.py --out data/sample --orders 6000
python jobs/train_local.py --input data/sample
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Sur les données d'exemple (générées, où le délai dépend de la distance, du poids, du nombre de
vendeurs et du pic de fin d'année), XGBoost obtient une PR AUC de 0,89 contre 0,88 pour la régression
logistique. Ces chiffres ne valent que pour ces données simulées ; les résultats sur le vrai jeu Olist
s'obtiennent avec la même commande après `download_data.sh`.

Sur Databricks, les tâches `train_late_delivery` puis `score_open_orders` s'ajoutent au job après silver,
sur un cluster ML Runtime mono-nœud.

## Lancer le projet en local

Prérequis : Python 3.10+ et Java 17 ou 21.

```bash
pip install -e ".[dev]"
pytest -q
```

Avec un petit jeu de données généré au même format qu'Olist :

```bash
python scripts/make_sample_data.py --out data/sample
python jobs/run_local.py --input data/sample --output data/gold
```

Avec les vraies données (compte Kaggle nécessaire) :

```bash
bash scripts/download_data.sh data/raw
python jobs/run_local.py --input data/raw --output data/gold
```

`run_local.py` applique exactement les mêmes transformations et contrôles que le job Databricks, mais écrit les tables gold en Parquet dans `data/gold/` au lieu de tables Delta. Ces fichiers s'ouvrent directement dans Power BI Desktop.

## Déploiement sur Azure

1. **Infrastructure** (resource group, ADLS Gen2, workspace Databricks, access connector Unity Catalog) :

   ```bash
   az login
   cd infra/terraform
   terraform init
   terraform apply
   ```

2. **Données** : envoyer les CSV dans le conteneur `landing` :

   ```bash
   bash scripts/upload_landing.sh data/raw
   ```

3. **Unity Catalog** : créer une storage credential à partir de l'access connector (sortie `access_connector_id`), une external location sur le conteneur `lakehouse`, puis le catalogue `olist`.

4. **Job** : le fichier `databricks.yml` décrit un job planifié chaque nuit : bronze → silver → gold, et en parallèle de gold l'entraînement puis le scoring.

   ```bash
   databricks bundle deploy -t dev
   databricks bundle run olist_medallion -t dev
   ```

   Le workflow GitHub Actions `deploy.yml` fait la même chose à la demande (secrets `DATABRICKS_HOST` et `DATABRICKS_TOKEN`).

## Power BI

Le modèle reprend les tables gold : `fact_sales` reliée aux trois dimensions et à `dim_date` via `purchase_date_key`.
Les mesures DAX sont dans `powerbi/measures.dax` : chiffre d'affaires, panier moyen, part du fret,
taux de retard, note moyenne en retard / à l'heure, évolution sur un an, taux de clients récurrents.

Pages du rapport :

- **Ventes** : chiffre d'affaires mensuel, panier moyen, top catégories, répartition par État ;
- **Livraison** : délai moyen, taux de retard par État vendeur et par catégorie ;
- **Satisfaction** : note moyenne, impact du retard sur la note ;
- **Risque de retard** : commandes en cours classées par probabilité de retard.

## Structure

```
src/olist_lakehouse/   code du pipeline (bronze, silver, quality, gold, merge)
src/olist_lakehouse/ml variables, entraînement MLflow, scoring
jobs/                  points d'entrée Databricks et local (pipeline, entraînement, scoring)
tests/                 tests unitaires PySpark
scripts/               téléchargement, envoi vers Azure, données d'exemple
infra/terraform/       infrastructure Azure
powerbi/               mesures DAX
databricks.yml         définition du job (Databricks Asset Bundle)
```
