# Olist Lakehouse

Lakehouse e-commerce construit sur Azure Databricks à partir du jeu de données public
[Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) : environ 100 000 commandes
passées entre 2016 et 2018 sur une marketplace brésilienne, réparties dans 9 fichiers CSV.

Le pipeline suit une architecture médaillon (bronze / silver / gold) en PySpark et Delta Lake,
avec chargement incrémental, contrôles qualité bloquants et un modèle en étoile exploité dans Power BI.

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

4. **Job** : le fichier `databricks.yml` décrit un job en trois tâches (bronze → silver → gold) planifié chaque nuit.

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
- **Satisfaction** : note moyenne, impact du retard sur la note.

## Structure

```
src/olist_lakehouse/   code du pipeline (bronze, silver, quality, gold, merge)
jobs/                  points d'entrée Databricks et local
tests/                 tests unitaires PySpark
scripts/               téléchargement, envoi vers Azure, données d'exemple
infra/terraform/       infrastructure Azure
powerbi/               mesures DAX
databricks.yml         définition du job (Databricks Asset Bundle)
```
