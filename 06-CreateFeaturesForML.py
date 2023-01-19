# Databricks notebook source
# MAGIC %md # Pushing Features to Feature Store Notebook
# MAGIC 
# MAGIC This notebook illustrates the use of different feature computation modes: Batch, Streaming and On-Demand. It has been shown that Machine learning models degrade in performance as the features become stale. This is true more so for certain type of features than others. If the data being generated updates quickly and factors heavily into the outcome of the model, it should be updated regularly. However, updating static data often would lead to increased costs with no perceived benefits. This notebook illustrates various feature computation modes available in Databricks using Databricks Feature Store based on the feature freshness requirements for a travel recommendation website. 
# MAGIC 
# MAGIC ![Feature Computation Options](files/shared_uploads/aakrati.talati@databricks.com/feature_computation_options_resized.png)
# MAGIC 
# MAGIC This notebook builds a Binary Classification model to predict likelihood of a user purchasing a product.
# MAGIC 
# MAGIC The notebook is structured as follows:
# MAGIC 
# MAGIC 1. Compute the features in three computation modes
# MAGIC    * Batch features
# MAGIC    * Streaming features
# MAGIC    * On-demand features
# MAGIC 1. Publishes the features to the online store, based on the freshness requirements using streaming or batch mode (This notebook uses DynamoDB. For a list of supported online stores, see the Databricks documentation (AWS|Azure)) 
# MAGIC 1. Serve realtime queries with automatic feature lookup
# MAGIC 
# MAGIC ### Requirements
# MAGIC * Databricks Runtime 11.2 LTS for Machine Learning or above. 

# COMMAND ----------

# MAGIC %pip install databricks-feature-store==0.8.0

# COMMAND ----------

from databricks import feature_store
import pyspark.sql.functions as F
import mlflow


fs = feature_store.FeatureStoreClient()

# COMMAND ----------

user_product_engage_metrics = 'sachin_clickstream.user_product_engage_metrics'
user_product_time_metrics  = 'sachin_clickstream.user_product_time_metrics'

# COMMAND ----------

# MAGIC %md # Filter only Expired Records and perform Feature Engineering
# MAGIC <br> </br> 
# MAGIC 1. Groupby User ID ,Product ID 
# MAGIC 2. Generate the Engagement Metrics for each of the product user combinations
# MAGIC 3. Generate time feature using start date and End Date for each of the product user combinations

# COMMAND ----------

df = spark.readStream.table('sachin_clickstream.electronics_cs_gold_user')
print('Columns',df.columns)
df = df.filter(F.col('expired') == True).selectExpr('product_id','user_id','numSessions','num_of_clicks','num_of_views',
                                                    'num_of_cart','max_price','min_price','is_purchase',
                                                    'from_unixtime(start_time/1000) as start_time','from_unixtime(end_time/1000) as end_time')

# COMMAND ----------

# MAGIC %md ## Creating Engagement aggregate statistics

# COMMAND ----------

# DBTITLE 1,Create Feature Table with Engagement statistics
user_product_engage_metrics_tbl = df.groupBy('product_id','user_id')\
    .agg(F.min("min_price").alias("min_price"), \
         F.max("max_price").alias("max_price"), \
         F.sum("num_of_cart").alias("num_of_cart"), \
         F.sum("num_of_clicks").alias("num_of_clicks"), \
         F.sum("numSessions").alias("numSessions"), \
         F.sum("num_of_views").alias("num_of_views")) 


# COMMAND ----------

# DBTITLE 1,Create Feature Table or Get if table exists
try:  # Try truncate table incase it already exists with data
#   spark.sql(f'TRUNCATE TABLE {hpt_feature_tbl}')
  fs.get_table(user_product_engage_metrics)
except Exception as e:
  print("Creating New Feature table as previous does not exist")
  fs.create_table(
          name=user_product_engage_metrics,
          primary_keys=["product_id","user_id"],
          schema=user_product_engage_metrics_tbl.schema,
          description=f"This is the feature table containing engagement metrics for each user and product combination")

fs.add_data_sources(feature_table_name = user_product_engage_metrics,
                    source_names = 'sachin_clickstream.electronics_cs_gold_user' ,
                    source_type = 'table')

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- drop table sachin_clickstream.user_product_engage_metrics

# COMMAND ----------

# DBTITLE 1,Stream Features into the table
fs.write_table(name = user_product_engage_metrics,
               df = user_product_engage_metrics_tbl,
               mode = 'merge',
               trigger = {'once': True})

# COMMAND ----------

# MAGIC %md ## Creating time based aggregate statistics for each of the product user combination

# COMMAND ----------

time_feature_tbl = df.groupBy('product_id','user_id')\
    .agg(F.min("start_time").alias("start_time"), \
         F.max("end_time").alias("end_time")) 

time_feature_tbl = time_feature_tbl.selectExpr("product_id","user_id","day(start_time) as session_start_day","month(start_time) as session_start_month",
                                              "dayofweek(start_time) as session_start_dayofweek","day(end_time) as session_end_day",
                                              "month(end_time) as session_end_month","dayofweek(end_time) as session_end_time_dayofweek",
                                              "(unix_timestamp(end_time) - unix_timestamp(start_time)) as duration")

# COMMAND ----------

# DBTITLE 1,Create Feature Table or Get if table exists
try:  # Try truncate table incase it already exists with data
#   spark.sql(f'TRUNCATE TABLE {hpt_feature_tbl}')
  fs.get_table(user_product_time_metrics)
except Exception as e:
  print("Creating New Feature table as previous does not exist")
  fs.create_table(
          name=user_product_time_metrics,
          primary_keys=["product_id","user_id"],
          schema=time_feature_tbl.schema,
          description=f"This is the feature table containing time statistics for each user and product combination")
# f_d =df_pd.groupby(['product_id','user_id']).count().reset_index()[df_pd.numSessions >1]
# f_d[f_d.numSessions >1]

fs.add_data_sources(feature_table_name = user_product_time_metrics,
                    source_names = 'sachin_clickstream.electronics_cs_gold_user' ,
                    source_type = 'table')

# COMMAND ----------

# DBTITLE 1,Stream Features into the feature table
fs.write_table(name = user_product_time_metrics,
               df = time_feature_tbl,
               mode = 'merge',
               trigger = {'once': True})

# COMMAND ----------

user_product_time_metrics.count()

# COMMAND ----------

# MAGIC %md ## Pushing the features to an online feature store
# MAGIC ## Set up Cosmos DB credentials
# MAGIC 
# MAGIC In this section, you need to take some manual steps to make Cosmos DB accessible to this notebook. Databricks needs permission to create and update Cosmos DB containers so that Cosmos DB can work with Feature Store. The following steps stores Cosmos DB keys in Databricks Secrets.
# MAGIC 
# MAGIC ### Look up the keys for Cosmos DB
# MAGIC 1. Go to Azure portal at https://portal.azure.com/
# MAGIC 2. Search and open "Cosmos DB", then create or select an account.
# MAGIC 3. Navigate to "keys" the view the URI and credentials.
# MAGIC 
# MAGIC ### Provide online store credentials using Databricks secrets
# MAGIC 
# MAGIC **Note:** For simplicity, the commands below use predefined names for the scope and secrets. To choose your own scope and secret names, follow the process in the Databricks [documentation](https://docs.microsoft.com/azure/databricks/applications/machine-learning/feature-store/online-feature-stores).
# MAGIC 
# MAGIC 1. Create two secret scopes in Databricks.
# MAGIC 
# MAGIC     ```
# MAGIC     databricks secrets create-scope --scope clickstream-read
# MAGIC     databricks secrets create-scope --scope clickstream-write
# MAGIC     ```
# MAGIC 
# MAGIC 2. Create secrets in the scopes.  
# MAGIC    **Note:** the keys should follow the format `<prefix>-authorization-key`. For simplicity, these commands use predefined names here. When the commands run, you will be prompted to copy your secrets into an editor.
# MAGIC 
# MAGIC     ```
# MAGIC     databricks secrets put --scope clickstream-read --key cosmos-authorization-key
# MAGIC     databricks secrets put --scope clickstream-write --key cosmos-authorization-key
# MAGIC     ```
# MAGIC     
# MAGIC Now the credentials are stored with Databricks Secrets. You will use them below to create the online store.

# COMMAND ----------

from databricks.feature_store.online_store_spec import AzureCosmosDBSpec

account_uri = "https://clickstream-fs.documents.azure.com:443/"

# Specify the online store.
# Note: These commands use the predefined secret prefix. If you used a different secret scope or prefix, edit these commands before running them.
#       Make sure you have a database created with same name as specified below.
online_store_spec = AzureCosmosDBSpec(
  account_uri=account_uri,
  write_secret_prefix="clickstream/cosmos",
  read_secret_prefix="clickstream/cosmos",
  database_name="clickstream_features",
  container_name=user_product_engage_metrics
)

# COMMAND ----------

# Specify the online store.
# Note: These commands use the predefined secret prefix. If you used a different secret scope or prefix, edit these commands before running them.
#       Make sure you have a database created with same name as specified below.
online_store_engage_spec = AzureCosmosDBSpec(
  account_uri=account_uri,
  write_secret_prefix="clickstream/cosmos",
  read_secret_prefix="clickstream/cosmos",
  database_name="clickstream_features",
  container_name=user_product_engage_metrics.split(".")[1]
)

# Push the feature table to online store.
fs.publish_table(user_product_engage_metrics, 
                 online_store_engage_spec,
                streaming= True,
                 mode = 'merge',
                trigger={'once': True})

# COMMAND ----------

online_store_engage_spec = AzureCosmosDBSpec(
  account_uri=account_uri,
  write_secret_prefix="clickstream/cosmos",
  read_secret_prefix="clickstream/cosmos",
  database_name="clickstream_features",
  container_name=user_product_time_metrics.split(".")[1]
)

# Push the feature table to online store.
fs.publish_table(user_product_time_metrics, 
                 online_store_engage_spec,
                streaming= True,
                trigger={'once': True})

# COMMAND ----------


