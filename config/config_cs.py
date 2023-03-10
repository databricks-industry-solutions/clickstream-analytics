# Databricks notebook source
# MAGIC %md
# MAGIC As a first step, below are the configuration for setting up the Eventhubs Namespace and topic. Notice that I am setting the listener key as well which will be used for retrieving the Eventhubs secrets.

# COMMAND ----------

EH_NAMESPACE = "sachinpatil"
EH_KAFKA_TOPIC = "clickstreameh"
EH_LISTEN_KEY_NAME = f"ehListen{EH_KAFKA_TOPIC}AccessKey"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Secure Azure Eventhubs connection
# MAGIC 
# MAGIC Azure Eventhubs can be secuted either through Azure Key Vault backed scope or Databricks scope.
# MAGIC https://docs.microsoft.com/en-us/azure/databricks/security/secrets/secret-scopes
# MAGIC 
# MAGIC In this example, secret is setup using a Databricks-backed secret scope using Databricks CLI

# COMMAND ----------

# Get Databricks secret value 
connSharedAccessKeyName = "RootManageSharedAccessKey"
connSharedAccessKey = dbutils.secrets.get(scope = "db_cs_access", key = EH_LISTEN_KEY_NAME)

# COMMAND ----------

EH_BOOTSTRAP_SERVERS = f"{EH_NAMESPACE}.servicebus.windows.net:9093"
EH_SASL = f"kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required username=\"$ConnectionString\" password=\"Endpoint=sb://{EH_NAMESPACE}.servicebus.windows.net/;SharedAccessKeyName={connSharedAccessKeyName};SharedAccessKey={connSharedAccessKey};EntityPath={EH_KAFKA_TOPIC}\";"
