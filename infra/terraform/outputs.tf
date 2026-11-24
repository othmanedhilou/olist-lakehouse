output "databricks_workspace_url" {
  value = "https://${azurerm_databricks_workspace.dbw.workspace_url}"
}

output "landing_path" {
  value = "abfss://${azurerm_storage_container.landing.name}@${azurerm_storage_account.lake.name}.dfs.core.windows.net/olist"
}

output "storage_account" {
  value = azurerm_storage_account.lake.name
}

output "access_connector_id" {
  value = azurerm_databricks_access_connector.uc.id
}
