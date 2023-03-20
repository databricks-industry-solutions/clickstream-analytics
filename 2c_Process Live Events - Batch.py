# Databricks notebook source
# MAGIC %md The purpose of this notebook is to process streaming, real-time data and write features in batch for use in inference scenarios with the Clickstream Propensity solution accelerator.  You may find this notebook at https://github.com/databricks-industry-solutions/clickstream-analytics

# COMMAND ----------

# MAGIC %md ##Important Note
# MAGIC 
# MAGIC This notebook should be running in parallel with notebook *2a*.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC Previously, we wrote historical event data to our lakehouse and from that data derived a set of features used to train a model. In this notebook, we will use the streaming event data to derive features in batch for those features we previously identified as only needing to be recalculated at midnight each day.  Those feature sets include our user, product and user-product features. (Cart and cart-product features are addressed in notebook *2b*.)
# MAGIC 
# MAGIC As part of this work, we will be recording feature data to the Databricks feature store. Because we are using a Databricks standard (not ML) cluster for this work, we must install the feature store library before getting started with our principal work:

# COMMAND ----------

# DBTITLE 1,Install Feature Store Components
# MAGIC %pip install databricks-feature-store

# COMMAND ----------

# DBTITLE 1,Get Config Info
# MAGIC %run "./0a_Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

import time
from datetime import datetime, timedelta

from databricks import feature_store
from databricks.feature_store.online_store_spec import AzureCosmosDBSpec

from delta.tables import *


# COMMAND ----------

# MAGIC %md ##Step 1: Configure Gold Layer
# MAGIC 
# MAGIC In notebook *2b* we addressed the setup of our Bronze and Silver layer tables.  We will use the event data persisted to the Silver-layer in that notebook as the starting point for writing our Gold-layer user, product and user-product feature tables:

# COMMAND ----------

# DBTITLE 1,Define User Features Table
# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS electronics_user_metrics__inference (
# MAGIC     user_id long,
# MAGIC     events long,
# MAGIC     views long,
# MAGIC     carts long,
# MAGIC     purchases long,
# MAGIC     event_time timestamp,
# MAGIC     view_to_events double,
# MAGIC     carts_to_events double,
# MAGIC     purchases_to_events double,
# MAGIC     carts_to_views double,
# MAGIC     purchases_to_views double,
# MAGIC     purchases_to_carts double
# MAGIC     ) USING delta;

# COMMAND ----------

# DBTITLE 1,Define Product Features Table
# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS electronics_product_metrics__inference (
# MAGIC     product_id long,
# MAGIC     events long,
# MAGIC     views long,
# MAGIC     carts long,
# MAGIC     purchases long,
# MAGIC     event_time timestamp,
# MAGIC     view_to_events double,
# MAGIC     carts_to_events double,
# MAGIC     purchases_to_events double,
# MAGIC     carts_to_views double,
# MAGIC     purchases_to_views double,
# MAGIC     purchases_to_carts double
# MAGIC     ) USING delta;

# COMMAND ----------

# DBTITLE 1,Define User-Product Features Table
# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS electronics_user_product_metrics__inference (
# MAGIC     user_id long,
# MAGIC     product_id long,
# MAGIC     events long,
# MAGIC     views long,
# MAGIC     carts long,
# MAGIC     purchases long,
# MAGIC     event_time timestamp,
# MAGIC     view_to_events double,
# MAGIC     carts_to_events double,
# MAGIC     purchases_to_events double,
# MAGIC     carts_to_views double,
# MAGIC     purchases_to_views double,
# MAGIC     purchases_to_carts double
# MAGIC     ) USING delta;

# COMMAND ----------

# MAGIC %md We will now register these tables with our feature store to serve as our offline feature store tables:

# COMMAND ----------

# DBTITLE 1,Connect to Feature Store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# DBTITLE 1,Register Tables with Offline Feature Store
_ = fs.register_table(
  delta_table='electronics_user_metrics__inference',
  primary_keys=['user_id'],
  description=f'User-level features as of prior day'
  )

_ = fs.register_table(
  delta_table='electronics_product_metrics__inference',
  primary_keys=['product_id'],
  description=f'Product-level features as of prior day'
  )

_ = fs.register_table(
  delta_table='electronics_user_product_metrics__inference',
  primary_keys=['user_id','product_id'],
  description=f'User-product features as of prior day'
  )

# COMMAND ----------

# MAGIC %md We will now publish these tables to our online feature store.  As our tables are currently empty, this isn't actually required at this point but it helps to create a sense of the end-to-end pipeline we will be pushing data through later:

# COMMAND ----------

# DBTITLE 1,Define Online Feature Store Spec
online_store_spec = AzureCosmosDBSpec(
  account_uri=config['cosmosdb_uri'],
  read_secret_prefix=f"{config['scope_readonly']}/{config['secret_prefix']}",
  write_secret_prefix=f"{config['scope_readwrite']}/{config['secret_prefix']}"
  )

# COMMAND ----------

# DBTITLE 1,Publish Initial Online Feature Store Tables
for target_table in ['electronics_user_metrics__inference','electronics_product_metrics__inference','electronics_user_product_metrics__inference']:
  
  _  = fs.publish_table(
    f"{config['database']}.{target_table}",
    online_store_spec,
    mode='merge'
    )

# COMMAND ----------

# MAGIC %md ##Step 2: Define Feature Logic
# MAGIC 
# MAGIC We will be writing data to each of our three batch-oriented feature tables using a merge operation, very similar in concept to what we did in the last notebook.  While our needs here are simpler in that we aren't managing long-lived arbitrary state information, the online feature store does not support a complete overwrite of its data with each batch cycle.  Instead, it supports a merge operation where records in the offline and online feature store are mapped via keys. This process can incur quite a bit of overhead so that it helps to limit our data updates to just those users, products and user-product combinations that have seen activity since our last batch cycle.  
# MAGIC 
# MAGIC To support this, we are going to write a function that recalculates metrics for just the records where updates are required.  These updates will be merged to our offline feature table.  The time of the batch, *i.e.* the cutoff_date*, will be used to then identify those records in the offline table that have received updates and therefore must be synced with our online feature store.  While a bit more complex that a complete update to all records, we have round this takes the synchronization time from minutes down to seconds.  Because the logic for this is pretty much the same across all three feature tables, we're writing one generalized function as follows:

# COMMAND ----------

# DBTITLE 1,Define Function to Write Features
def publish_features(cutoff_date, groupby_keys=[], target_table=None):
  
  # assemble features
  df = (
    spark
      .table('electronics_events_silver')
      .filter(f"event_time<'{cutoff_date}'")
      .groupBy(groupby_keys)
        .agg(
          fn.expr("COUNT(*)").alias('events'),
          fn.expr("COUNT_IF(event_type='view')").alias('views'),
          fn.expr("COUNT_IF(event_type='carts')").alias('carts'),
          fn.expr("COUNT_IF(event_type='purchase')").alias('purchases'),
          fn.max('event_time').alias('event_time')
          )
      .withColumn('view_to_events', fn.expr("views/events"))
      .withColumn('carts_to_events', fn.expr("carts/events"))
      .withColumn('purchases_to_events', fn.expr("purchases/events"))
      .withColumn('carts_to_views', fn.expr("carts/views"))
      .withColumn('purchases_to_views', fn.expr("purchases/views"))
      .withColumn('purchases_to_carts', fn.expr("purchases/carts"))
      .withColumn('cutoff_date', fn.lit(cutoff_date))
    )

  # merge data into feature table if record has changed
  features_target = DeltaTable.forName(spark, f"{config['database']}.{target_table}")
  _ = ( 
      features_target.alias('t')
        .merge(
          df.alias('s'),
          condition=' AND '.join([f"s.{c}=t.{c}" for c in groupby_keys])
          )   
      .whenMatchedUpdate(
        condition="s.event_time>t.event_time", # metric has received update
        set={
              "events": "s.events",
              "views": "s.views",
              "carts": "s.carts",
              "purchases": "s.purchases",
              "event_time": "s.cutoff_date", # set event_time to cutoff time which is max possible value for event_time in df
              "view_to_events": "s.view_to_events",
              "carts_to_events": "s.carts_to_events",
              "purchases_to_views": "s.purchases_to_views",
              "purchases_to_carts": "s.purchases_to_carts"
              }
          ) 
      .whenNotMatchedInsertAll()
      .execute()
      )
  
  # publish features
  _  = fs.publish_table(
    f"{config['database']}.{target_table}",
    online_store_spec,
    filter_condition=f"event_time='{cutoff_date}'", # only publish those changed in this cycle
    mode='merge'
    )

# COMMAND ----------

# MAGIC %md ##Step 3: Write to Gold & Publish to Feature Store
# MAGIC 
# MAGIC Now we can write our logic to publish features as we move through simulated time.  In a real-world implementation of this logic, we would not setup such a loop and instead would call a script on a predefined schedule.  But because we are simulating data coming in in accelerated real-time, we setup an indefinite loop and in that loop try to determine when we've crossed into a new day and therefore required new data to be published. 
# MAGIC 
# MAGIC To prevent you from forgetting to turn off this loop, we cap the execution for 3 cycles

# COMMAND ----------

# DBTITLE 1,Set Initial Event Time
last_event_date = datetime.strptime('1970-01-01','%Y-%m-%d')

# COMMAND ----------

# DBTITLE 1,Poll Event Data to Trigger Feature Generation
import time
timeout_start = time.time()

while time.time() < timeout_start + 1800: # To prevent you from forgetting to turn off this loop, we cap the execution for 3 cycles

# while True: # use this while condition if you want to leave the loop running indefinitely; You must stop the execution of this code when you are done with this demonstration or it will trigger your cluster to continue running over time: 
  
  # poll silver table for last observed time
  event_date = (
    spark
      .table('electronics_events_silver')
      .groupBy()
        .agg(
          fn.max('event_time').alias('event_time')
          )
      .withColumn('event_time',fn.expr("DATE_TRUNC('day',event_time)"))
    ).collect()[0]['event_time']
  
  # if last observed time has crossed midnight threshold
  if event_date > last_event_date:
  
    cutoff_date = event_date - timedelta(days=1)
    print(f"Updating features as of {cutoff_date}")    
    
    # calculate and publish features
    publish_features(cutoff_date, groupby_keys=['user_id'], target_table='electronics_user_metrics__inference')
    publish_features(cutoff_date, groupby_keys=['product_id'], target_table='electronics_product_metrics__inference')
    publish_features(cutoff_date, groupby_keys=['user_id','product_id'], target_table='electronics_user_product_metrics__inference')  
  
  # capture last observed time
  last_event_date = event_date
  
  # sleep until next cycle
  time.sleep(600)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.

# COMMAND ----------


