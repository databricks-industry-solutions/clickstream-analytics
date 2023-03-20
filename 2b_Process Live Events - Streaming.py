# Databricks notebook source
# MAGIC %md The purpose of this notebook is to process streaming data and write real-time features for use in inference scenarios with the Clickstream Propensity solution accelerator. You may find this notebook at https://github.com/databricks-industry-solutions/clickstream-analytics

# COMMAND ----------

# MAGIC %md ##Important Note
# MAGIC 
# MAGIC This notebook should be running in parallel with notebook *2a*.

# COMMAND ----------

# MAGIC %md ##Important Note
# MAGIC 
# MAGIC In the last step of this notebook, we will be publishing data to an Azure CosmosDB document store. If you run this notebook as part of the job created by the **RUNME** notebook, or using the `clickstream_photon_cluster` cluster created by the **RUNME** notebook, the cluster is already configured with the necessary packages.  If you use your own cluster and wish to avoid restarting your cluster later on, you may wish to jump to the last step, *Step 4*, and complete the CosmosDB deployment and integration actions before running this notebook.
# MAGIC 
# MAGIC Also see the **RUNME** notebook for details about setting up the secret scopes for your CosmosDB URI and secrets.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC Previously, we wrote historical event data to our lakehouse and from that data derived a set of features used to train a model. In this notebook, we will use the streaming event data to derive features in real-time for those features we previously identified as requiring updated state information with each click.  Those feature sets include our cart and cart-product features.  (User, product and user-product features are addressed in notebook *2c*.)
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
from pyspark.sql.types import *
from databricks import feature_store
from databricks.feature_store.online_store_spec import AzureCosmosDBSpec
from delta.tables import *
import pandas as pd
from datetime import timedelta, datetime
from typing import Iterator
import time

# COMMAND ----------

# MAGIC %md ##Step 1: Write to Bronze
# MAGIC 
# MAGIC With our data streaming to an Azure Event Hub, we can read it via its Kafka interface.  As we read the data, we will archive the messages to a Bronze layer table.  We will record the timestamp on the ingest layer as our *ingest_ts* value what we will attach to each record as it moves through our medallion architecture:
# MAGIC 
# MAGIC **NOTE** Please note, we are explicitly setting our workflow to perform continuous data processing with the *trigger* configuration found below. This setting is the default but we wanted to be explicit to avoid any confusion about our intent to process data as quickly as possible.

# COMMAND ----------

# DBTITLE 1,Read & Persist Raw Messages
bronze_events = (
  spark
    .readStream
      .format('kafka')
      .option('subscribe', config['eh_kafka_topic'])
      .option('kafka.bootstrap.servers', config['eh_bootstrap_servers'])
      .option('kafka.sasl.mechanism', 'PLAIN')
      .option('kafka.security.protocol', 'SASL_SSL')
      .option('kafka.sasl.jaas.config', config['eh_sasl'])
      .option('startingOffsets','latest')
      .option('failOnDataLoss','false')
      .load() 
    .withColumn('ingest_ts', fn.from_json( fn.col('value').cast('string'), StructType([StructField('ingest_ts', TimestampType())])) ) # pull ingest_ts from payload for perf monitoring
    .withColumn('ingest_ts', fn.expr('ingest_ts.ingest_ts')) # pull value from struct
    .withColumn('current_ts',fn.expr("CURRENT_TIMESTAMP()"))
  )

_ = (
  bronze_events
    .writeStream
      .format('delta')
      .outputMode('append')
      .option('checkpointLocation', f"{config['checkpoint_path']}/electronics_events_bronze")
      .trigger(processingTime='0 seconds')
      .toTable('electronics_events_bronze')
  )

# COMMAND ----------

# MAGIC %md ##Step 2: Write Bronze to Silver
# MAGIC 
# MAGIC Building off our Bronze layer data definition, we can now extract the message payload from the incoming event data and transform it back into the event message structure in our original dataset.  
# MAGIC 
# MAGIC Please note that we have the option to read from the Bronze-layer table serving as the data sink in our last step or from the Bronze-layer dataframe which represents the instructions for how to read data from our ingest layer. We've elected to read from the dataframe to avoid even the slightest overhead associated with persisting data mid-stream:

# COMMAND ----------

# DBTITLE 1,Extract Event Record from Raw Message Payload
# event data schema
silver_events_schema = StructType([
  StructField('event_time', TimestampType()),
  StructField('event_type', StringType()),  
  StructField('product_id', LongType()),  
  StructField('category_id', LongType()),
  StructField('category_code', StringType()),
  StructField('brand', StringType()),  
  StructField('price', FloatType()),  
  StructField('user_id', LongType()),
  StructField('user_session', StringType()),
  StructField('ingest_ts', TimestampType())
  ])

# extract events data from payload (value) field
silver_events = (
  bronze_events
    .select(
      fn.from_json( fn.col('value').cast('string'), silver_events_schema ).alias('event')
      )
    .selectExpr('event.*','CURRENT_TIMESTAMP() as current_ts')
  )

# write extracted events data to silver table
_ = (
  silver_events
    .writeStream
      .format('delta')
      .outputMode('append')
      .option('checkpointLocation', f"{config['checkpoint_path']}/electronics_events_silver")
      .toTable('electronics_events_silver')
  )

# COMMAND ----------

# MAGIC %md ##Step 3: Write Silver to Gold
# MAGIC 
# MAGIC Building on our Silver-layer dataframe definition, we can now extract metrics from the incoming events.

# COMMAND ----------

# MAGIC %md Starting with our cart-level metrics, we must first consider how we will manage state across what can be a very long-lived user session.  To calculate metrics across the entire span of a user session, we need to either:</p>
# MAGIC 
# MAGIC * Define a long [watermark](https://www.databricks.com/blog/2022/08/22/feature-deep-dive-watermarking-apache-spark-structured-streaming.html) period over which we will hold onto cart state data
# MAGIC * Define a custom function for managing state information, aka [arbitrary state](https://www.databricks.com/blog/2022/10/18/python-arbitrary-stateful-processing-structured-streaming.html)
# MAGIC 
# MAGIC In very high volume scenarios, we may wish to implement arbitrary state management to allow us greater control over how long data will reside in state.  This is tackled writing a function such as the one that follows:

# COMMAND ----------

# DBTITLE 1,Define Arbitrary State Management Function for Cart Metrics
def cart_func(key, pdfs, state) -> Iterator[pd.DataFrame]:
  
  # set hours following purchase to keep state alive
  hours_to_live_after_purchase = 1 * 24 
  hours_to_live_with_no_purchase = 30 * 24 
  
  # read grouping keys
  (user_id, user_session) = key
  
  # initialize variables from state
  if state.exists:
    (events, views, carts, purchases, event_time, ingest_ts) = state.get
  else:
    (events, views, carts, purchases, event_time, ingest_ts) = \
        (0,0,0,0, datetime.strptime('1970-01-01','%Y-%m-%d'), datetime.strptime('1970-01-01','%Y-%m-%d'))
  
  
  # if you are entering this function because state expired
  if state.hasTimedOut:
    
    # calculate time since last observed event
    seconds_since_last_event = (datetime.now() - event_time).total_seconds()
    
    # if a purchase completed and time to live with purchase expired, remove state
    if purchases > 0 and seconds_since_last_event > (hours_to_live_after_purchase *60*60):
      state.remove()
    
    # if no purchase and time to live w/o purchase expired, remove state
    elif purchases == 0 and seconds_since_last_event > (hours_to_live_with_no_purchase *60*60):
      state.remove()
      
    else: # else renew the current state for limited time
      state.update([events, views, carts, purchases, event_time, ingest_ts]) # set state values
      state.setTimeoutTimestamp(state.getCurrentWatermarkMs() + (1 *60*60*1000)) # increment watermark by 1 hour

    # just return nothing b/c this unit of code triggered by state timeout
    yield pd.DataFrame()
  
  # else state is not timed out and there must be data to proccess
  else: 
    
    # loop through each unit of data
    for pdf in pdfs:
      
      # count metrics from data
      events += pdf.shape[0]
      views += pdf[pdf['event_type']=='view'].shape[0]
      carts += pdf[pdf['event_type']=='cart'].shape[0]
      purchases += pdf[pdf['event_type']=='purchase'].shape[0]
      event_time = max(pdf['event_time'].max(), event_time)
      ingest_ts = max(pdf['ingest_ts'].max(), ingest_ts)
    
    # update state values
    state.update([events, views, carts, purchases, event_time, ingest_ts]) # set state values
    state.setTimeoutTimestamp(state.getCurrentWatermarkMs() + (min(hours_to_live_after_purchase, hours_to_live_with_no_purchase)*60*60*1000)) # renew state minimum increment

    # return to screen
    yield pd.DataFrame.from_dict(
      {"user_id":[user_id], "user_session":[user_session], "events":[events], "views":[views], "carts":[carts], "purchases":[purchases], "event_time":[event_time], "ingest_ts":[ingest_ts]}
      )
  
  
# define schema for function output  
cart_metrics_output_schema = StructType(
  [
    StructField('user_id', LongType()),
    StructField('user_session', StringType()),
    StructField('events', LongType()),
    StructField('views', LongType()),
    StructField('carts', LongType()),
    StructField('purchases', LongType()),
    StructField('event_time', TimestampType()),
    StructField('ingest_ts', TimestampType())
    ]
  )

# define schema for function state 
cart_metrics_state_schema = StructType(
  [
    StructField('events', LongType()),
    StructField('views', LongType()),
    StructField('carts', LongType()),
    StructField('purchases', LongType()),
    StructField('event_time', TimestampType()),
    StructField('ingest_ts', TimestampType())
    ]
  )

# COMMAND ----------

# MAGIC %md There's a lot going on in this function, so let's take a moment to break it down. 
# MAGIC 
# MAGIC First, we define two variables which define how much time we will allow a user session to remain alive in state. When a purchase occurs, we may see a few additional events associated with it within a relatively short window following the purchase event.  When a purchase has yet to occur, there may be an extended period following the last observed event during which we could see the user return to the session.
# MAGIC 
# MAGIC With these variables set, we can focus on retrieving keys and state.  To understand these items, consider that this function will be called on event data grouped around *user_id* and *user_session* values. We grab the specific user_id and user_session values on which in incoming data (in the *pdfs* variable) is grouped.  The in-memory state data for this specific group (from any prior calls on the group) is accessed through the *state* variable.  It is possible this is the first time this group has been called in which case there will be no state data so we call *state.exists* to determine how to initialize the variables associated with group state.
# MAGIC 
# MAGIC Now that that's taken care of, we have to determine why this function has been called.  It may have been called by the Spark engine because a particular group's state has timed out, or it may have been called because new data for that group has been detected.
# MAGIC 
# MAGIC If the reason for the function call is a timeout, we examine the time since the group last received a new event message.  If there has been a purchase and the time after purchase has elapsed, we expire the state.  If there has not been a purchase and the time allowed since the last event when no purchase has occurred has expired, we also expire the state. But if the timeout has occurred prior to these windows (as may have been triggered by the watermark we will define on our streaming query), we simply renew the state and move on.
# MAGIC 
# MAGIC If the reason for the function call is that new data has been observed, we loop through the pandas dataframes that represent the different partitions associated with this group of event records.  We increment our state metrics and also capture the latest event date observed.  These values are recorded to state and a new timeout is set before we return the updated state metrics as our function output.  (The last timeout setting here is optional.  If we didn't set it, we could allow any preconfigured timeouts or the query's watermark to trigger the function again so that the timeout logic could be engaged.)
# MAGIC 
# MAGIC With this function defined, we can now query the data as follows:

# COMMAND ----------

# DBTITLE 1,Apply Function to Stream to Calculate Cart Metrics
cart_features = (
  silver_events
    .select('user_id','user_session','event_time','event_type','ingest_ts')
    .withWatermark('event_time','1 hour')
    .groupBy('user_id','user_session')
      .applyInPandasWithState(
        cart_func, 
        cart_metrics_output_schema, 
        cart_metrics_state_schema, 
        'update', 
        'EventTimeTimeout'
        )
    .withColumn('view_to_events', fn.expr("views/events"))
    .withColumn('carts_to_events', fn.expr("carts/events"))
    .withColumn('purchases_to_events', fn.expr("purchases/events"))
    .withColumn('carts_to_views', fn.expr("carts/views"))
    .withColumn('purchases_to_views', fn.expr("purchases/views"))
    .withColumn('purchases_to_carts', fn.expr("purchases/carts"))
    .withColumn('current_ts', fn.expr('CURRENT_TIMESTAMP()'))
  )

# COMMAND ----------

# MAGIC %md In addition to calculating updated state, we've added our logic to calculate our various ratio features.  We intend to write these to a table as defined here:
# MAGIC 
# MAGIC **NOTE** We've used the *USING delta* option with this table definition in order to explicitly call out that we are recording the data in the delta format.  The delta format is the default format used in Databricks unless another format is specified.  We are explicitly defining our table as using the delta format as this is critical to the actions we will perform in Step 4 of this notebook.

# COMMAND ----------

# DBTITLE 1,Create Cart-Level Feature Table
# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS electronics_cart_metrics__inference (
# MAGIC     user_id long,
# MAGIC     user_session string,
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
# MAGIC     purchases_to_carts double,
# MAGIC     ingest_ts timestamp,
# MAGIC     current_ts timestamp
# MAGIC     ) USING delta;

# COMMAND ----------

# MAGIC %md As we write our feature out put to this table, it's important to keep in mind that we only want our table to capture the current state of a given shopping session.  As a result, our table should only have one record for each *user_id* and *user_session* combination. As state is updated, we need to update any existing records with the new state metrics.
# MAGIC 
# MAGIC This logic requires us to write data to our table in an *update* mode whereas everything we've done in our Bronze and Silver layers has focused on *append* mode logic.  Because we cannot combine *append* and *update* logic across a stream using default capabilities, we will need to write a custom update function to merge our incoming data with our target table:

# COMMAND ----------

# DBTITLE 1,Define Target Update Function
# identify target
cart_features_target = DeltaTable.forName(spark, f"{config['database']}.electronics_cart_metrics__inference")

# function to merge batch data with delta table
def cart_features_upsertToDelta(microBatchOutputDF, batchId):
  ( 
    cart_features_target.alias('t')
      .merge(
        microBatchOutputDF.alias('s'),
        's.user_id=t.user_id AND s.user_session=t.user_session'
        )   
    .whenMatchedUpdateAll() 
    .whenNotMatchedInsertAll()
    .execute()
    )

# COMMAND ----------

# MAGIC %md And now we can write our features to our target table as follows: 

# COMMAND ----------

# DBTITLE 1,Write Cart Features to Table
# start the query to continuously upsert into aggregates tables in update mode
_ = (
  cart_features
    .writeStream
      .format('delta')
      .option('checkpointLocation', f"{config['checkpoint_path']}/electronics_cart_metrics__inference")
      .foreachBatch(cart_features_upsertToDelta)
      .outputMode('update')
      .start()
  )

# COMMAND ----------

# MAGIC %md With our cart metrics in place, we can perform a very similar step to calculate cart-product metrics.  If we wanted to get fancy, we could actually rewrite the function above to examine the keys and provide logic based on the level of granularity for which we are performing calculations.  However, we felt the function was already a lot to absorb and writing a separate (though mostly identical) function for the cart-product metrics would be more accessible for those reviewing this the first time around:

# COMMAND ----------

# DBTITLE 1,Define Arbitrary State Management Function for Cart-Product Metrics
def cart_product_func(key, pdfs, state) -> Iterator[pd.DataFrame]:
  
  # set hours following purchase to keep state alive
  hours_to_live_after_purchase = 1 * 24 
  hours_to_live_with_no_purchase = 30 * 24 
  
  # read grouping keys
  (user_id, user_session, product_id) = key
  
  # initialize variables from state
  if state.exists:
    (events, views, carts, purchases, event_time, ingest_ts) = state.get
  else:
    (events, views, carts, purchases, event_time, ingest_ts) = (0,0,0,0,datetime.strptime('1970-01-01','%Y-%m-%d'),datetime.strptime('1970-01-01','%Y-%m-%d'))
  
  
  # if you are entering this function because state expired
  if state.hasTimedOut:
    
    # calculate time since last observed event
    seconds_since_last_event = (datetime.now() - event_time).total_seconds()
    
    # if a purchase completed and time to live with purchase expired, remove state
    if purchases > 0 and seconds_since_last_event > (hours_to_live_after_purchase *60*60):
      state.remove()
    
    # if no purchase and time to live w/o purchase expired, remove state
    elif purchases == 0 and seconds_since_last_event > (hours_to_live_with_no_purchase *60*60):
      state.remove()
      
    else: # else renew the current state for limited time
      state.update([events, views, carts, purchases, event_time, ingest_ts]) # set state values
      state.setTimeoutTimestamp(state.getCurrentWatermarkMs() + (1 *60*60*1000)) # increment watermark by 1 hour

    # just return nothing b/c this unit of code triggered by state timeout
    yield pd.DataFrame()
  
  # else state is not timed out and there must be data to proccess
  else: 
    
    # loop through each unit of data
    for pdf in pdfs:
      
      # count metrics from data
      events += pdf.shape[0]
      views += pdf[pdf['event_type']=='view'].shape[0]
      carts += pdf[pdf['event_type']=='cart'].shape[0]
      purchases += pdf[pdf['event_type']=='purchase'].shape[0]
      event_time = max(pdf['event_time'].max(), event_time)
      ingest_ts = max(pdf['ingest_ts'].max(), ingest_ts)
    
    # update state values
    state.update([events, views, carts, purchases, event_time, ingest_ts]) # set state values
    state.setTimeoutTimestamp(state.getCurrentWatermarkMs() + (min(hours_to_live_after_purchase, hours_to_live_with_no_purchase)*60*60*1000)) # renew state minimum increment

    # return to screen
    yield pd.DataFrame.from_dict(
      {"user_id":[user_id], "user_session":[user_session], "product_id":product_id, "events":[events], "views":[views], "carts":[carts], "purchases":[purchases], "event_time":[event_time], "ingest_ts":ingest_ts}
      )
    
  
# define schema for function output  
cart_product_metrics_output_schema = StructType(
  [
    StructField('user_id', LongType()),
    StructField('user_session', StringType()),
    StructField('product_id', LongType()),
    StructField('events', LongType()),
    StructField('views', LongType()),
    StructField('carts', LongType()),
    StructField('purchases', LongType()),
    StructField('event_time', TimestampType()),
    StructField('ingest_ts', TimestampType())
    ]
  )

# define schema for function state 
cart_product_metrics_state_schema = StructType(
  [
    StructField('events', LongType()),
    StructField('views', LongType()),
    StructField('carts', LongType()),
    StructField('purchases', LongType()),
    StructField('event_time', TimestampType()),
    StructField('ingest_ts', TimestampType())
    ]
  )

# COMMAND ----------

# DBTITLE 1,Apply Function to Stream to Calculate Cart-Product Metrics
cart_product_features = (
  silver_events
    .select('user_id','user_session','product_id','event_time','event_type','ingest_ts')
    .withWatermark('event_time','1 hour')
    .groupBy('user_id','user_session','product_id')
      .applyInPandasWithState(
        cart_product_func, 
        cart_product_metrics_output_schema, 
        cart_product_metrics_state_schema, 
        'Update', 
        'EventTimeTimeout'
        )
    .withColumn('view_to_events', fn.expr("views/events"))
    .withColumn('carts_to_events', fn.expr("carts/events"))
    .withColumn('purchases_to_events', fn.expr("purchases/events"))
    .withColumn('carts_to_views', fn.expr("carts/views"))
    .withColumn('purchases_to_views', fn.expr("purchases/views"))
    .withColumn('purchases_to_carts', fn.expr("purchases/carts"))
    .withColumn('current_ts', fn.expr('current_timestamp()'))
  )

# COMMAND ----------

# DBTITLE 1,Create Cart-Product Features Table
# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS electronics_cart_product_metrics__inference (
# MAGIC     user_id long,
# MAGIC     user_session string,
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
# MAGIC     purchases_to_carts double,
# MAGIC     ingest_ts timestamp,
# MAGIC     current_ts timestamp
# MAGIC     ) USING delta;

# COMMAND ----------

# DBTITLE 1,Define Target Update Function
# identify target
cart_product_features_target = DeltaTable.forName(spark, f"{config['database']}.electronics_cart_product_metrics__inference")

# function to merge batch data with delta table
def cart_product_features_upsertToDelta(microBatchOutputDF, batchId):
  ( 
    cart_product_features_target.alias('t')
      .merge(
        microBatchOutputDF.alias('s'),
        's.user_id=t.user_id AND s.user_session=t.user_session AND s.product_id=t.product_id'
        )   
    .whenMatchedUpdateAll() 
    .whenNotMatchedInsertAll()
    .execute()
    )

# COMMAND ----------

# DBTITLE 1,Write Cart-Product Features to Table
# start the query to continuously upsert into aggregates tables in update mode
_ = (
  cart_product_features
    .writeStream
      .format('delta')
      .option('checkpointLocation', f"{config['checkpoint_path']}/electronics_cart_product_metrics__inference")
      .foreachBatch(cart_product_features_upsertToDelta)
      .outputMode('update')
      .start()
  )

# COMMAND ----------

# MAGIC %md Before moving on, it's worth exploring how we might deal with very large state information.  If we were operating a large website, its possible we'd have an overwhelming number of shopping carts and product-cart combinations that we'd need to keep up with.  In such a situation, the memory required for state management might put a severe strain on our cluster resources.  In this scenario, we may consider offloading our state information to a [RocksDB state store](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#rocksdb-state-store-implementation).
# MAGIC 
# MAGIC The RocksDB state store provides access to storage space in native memory and on local disk.  It's pre-integrated with Databricks so that setting up the use of this feature is pretty straightforward.  You can read more about how to leverage RocksDB from within the Databricks environment [here](https://docs.databricks.com/structured-streaming/rocksdb-state-store.html).

# COMMAND ----------

# MAGIC %md ##Step 4: Publish to Feature Store
# MAGIC 
# MAGIC With our structured streaming queries in place and writing data to output tables, we now need to consider how we might publish features to our feature store.  To support high-speed inference scenarios, the Databricks feature store supports an online feature store capability. The first step to publishing data to an online feature store is first to record it in an offline feature store table which is what we've already done by writing our data to our two previously defined delta tables.  All we need to do to configure these as our offline feature store tables is register them with the feature store capability:

# COMMAND ----------

# DBTITLE 1,Connect to Feature Store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# DBTITLE 1,Register Tables as Offline Feature Tables
fs.register_table(
  delta_table='electronics_cart_metrics__inference',
  primary_keys=['user_id','user_session'],
  description=f'Cart-level features for each user session'
  )

fs.register_table(
  delta_table='electronics_cart_product_metrics__inference',
  primary_keys=['user_id','user_session','product_id'],
  description=f'Cart-product level features for each user session'
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC We can now publish our data to an online feature store. The online feature store is either a high-performance relational database or document store that's accessible to our model serving layer.  Because we wish to leverage the Databricks serverless real-time inference capability (recently renamed as Databricks *model serving*) for this, we are locked into the use of a [CosmosDB document store](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/online-feature-stores) in the Azure cloud and a [DynamoDB document store](https://docs.databricks.com/machine-learning/feature-store/online-feature-stores.html) in AWS. (The online feature store is not yet available in the Google cloud as of the time of notebook development.)
# MAGIC 
# MAGIC Because we are demonstrating this solution accelerator in the Azure cloud, we will be setting up an Azure CosmosDB document store.  The steps for deploying an Azure CosmosDB document store are found [here](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/quickstart-portal).  The key items to consider are:
# MAGIC </p>
# MAGIC 
# MAGIC * the document store should be deployed to the same region as your Databricks cluster
# MAGIC * the Core (SQL) API (aka *Azure Cosmos DB for NoSQL*) should be specified during CosmosDB deployment
# MAGIC * network connectivity should be set to *All Networks* on the CosmosDB service so that the Databricks service can communicate directly to it
# MAGIC 
# MAGIC Once your CosmosDB document store has been deployed, be sure to get one [authorization key](https://learn.microsoft.com/en-us/azure/cosmos-db/secure-access-to-data?tabs=using-primary-key#primary-keys) from the CosmosDB service with read-only access to the store and another with read-write access to the store. You'll need these in later steps.  You will also need to capture the CosmosDB URI and set it up in a secret scope (see *RUNME* notebook).
# MAGIC 
# MAGIC **NOTE** It's important to note that the CosmosDB database uses 4,000 RU/s for its default throughput. RU/s (request units per second) are explained [here](https://learn.microsoft.com/en-us/azure/cosmos-db/request-units).  You'll need to configure a value appropriate for your needs.  If you experience 429 errors from the service, this is an indication you are pushing more data to CosmosDB than it is configured to handle and you may need to raise the RU/s to keep up.  For this demo running with a speed factor of 100, we sized our database for 1,000 RU/s and did not appear to have any problems.

# COMMAND ----------

# DBTITLE 1,Set Online Feature Store Information
print(f"cosmosdb_uri:\t{config['cosmosdb_uri']}")

# COMMAND ----------

# MAGIC %md Before proceeding, it's a good idea to make sure you've configured your Databricks cluster to use the latest [Azure Cosmos DB Spark 3 OLTP Connector for SQL API](https://github.com/Azure/azure-sdk-for-java/blob/main/sdk/cosmos/azure-cosmos-spark_3-2_2-12/README.md#download).  As a Java JAR, it must be installed as either a [cluster or workspace library](https://learn.microsoft.com/en-us/azure/databricks/libraries/). 
# MAGIC 
# MAGIC **NOTE** At the time of development, the latest connector was *azure-cosmos-spark_3-3_2-12 version 4.17.0*, which is the version of connector in the *clickstream_photon_cluster* that the **RUNME** notebook creates for you.

# COMMAND ----------

# MAGIC %md 
# MAGIC With the Azure CosmosDB document store deployed and the library installed, we now need to record the read-only and read-write authentication keys for the store as [Databricks secrets](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/secrets#create-a-secret-in-a-databricks-backed-scope). In an Azure environment, you can create either a Databricks-backed scope or an Azure Key Vault scope.  In this demo, we have employed a Databricks-backed scope to keep things simpler.
# MAGIC 
# MAGIC ### Option 1
# MAGIC To use the Databricks API to set up the secret scopes, we provided a sample script for you in the **RUNME** notebook. The sample script also helps you set up secrets for other credentials needed, such as the Kaggle credential and the Event hub credentials for streaming.
# MAGIC 
# MAGIC ### Option 2
# MAGIC Another option of settig up the secret scope is to make use of the Databricks CLI. To use the CLI, you first need to install and configure it to your local system, and to do that, you'll need to follow the instructions provided [here](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/). (While the CLI runs on your local system, it creates the secrets in the environment for which it has been configured.  It is critical that you configure your installation of the Databricks CLI to point to the environment where you are running these notebooks.)
# MAGIC 
# MAGIC After it's been configured, you'll need to setup two secret scopes, one to hold your read-only key and the other to hold your read-write key.  For example, you might create scopes as follows:
# MAGIC 
# MAGIC ```
# MAGIC databricks secrets create-scope --scope clickstream-readonly
# MAGIC databricks secrets create-scope --scope clickstream-readwrite
# MAGIC ```
# MAGIC 
# MAGIC Once the scopes are defined, you now need to place the approrpriate authentication keys in them.  Each key will use a set prefix that you will define.  Here, we are using a prefix of `onlinefs`. Please note, the remainder of the key name should be recorded as *authorization-key*:
# MAGIC 
# MAGIC ```
# MAGIC databricks secrets put --scope clickstream-readonly --key onlinefs-authorization-key
# MAGIC databricks secrets put --scope clickstream-readwrite --key onlinefs-authorization-key
# MAGIC ```
# MAGIC As you enter each command, you will be prompted to select a text editor.  Choose the one you are most familiar with and follow the instructions, pasting the appropriate CosmosDB authentication key in each. Be sure to record your scope names and prefix in notebook *0a*.

# COMMAND ----------

# DBTITLE 1,Set Secrets Information
print(f"scope_readonly:\t{config['scope_readonly']}")
print(f"scope_readwrite:\t{config['scope_readwrite']}")
print(f"secret_prefix:\t{config['secret_prefix']}")

# COMMAND ----------

# MAGIC %md With the service behind our online feature store deployed and configuration settings used to connect us to this service captured in notebook *0a*, we can now define our online feature store specification:

# COMMAND ----------

# DBTITLE 1,Define Online Feature Store Spec
online_store_spec = AzureCosmosDBSpec(
  account_uri=config['cosmosdb_uri'],
  read_secret_prefix=f"{config['scope_readonly']}/{config['secret_prefix']}",
  write_secret_prefix=f"{config['scope_readwrite']}/{config['secret_prefix']}"
  )

# COMMAND ----------

# MAGIC %md Using this spec, we can now push updated data to the online feature store as follows:

# COMMAND ----------

# DBTITLE 1,Publish Cart Metrics
_ = fs.publish_table(
  f"{config['database']}.electronics_cart_metrics__inference", # offline feature store table where features come from  
  online_store_spec, # specs for connecting to online feature store
  mode = 'merge', 
  streaming= True,
  checkpoint_location = f"{config['checkpoint_path']}/electronics_cart_metrics__inference_online",
  trigger={'processingTime': '0 seconds'}
  )

# COMMAND ----------

# DBTITLE 1,Publish Cart-Product Metrics
_ = fs.publish_table(
  f"{config['database']}.electronics_cart_product_metrics__inference", 
  online_store_spec,
  mode = 'merge',
  streaming= True,
  checkpoint_location = f"{config['checkpoint_path']}/electronics_cart_product_metrics__inference_online",
  trigger={'processingTime': '0 seconds'}
  )

# COMMAND ----------

# MAGIC %md ##Step 5: Confirm Event Data & Feature Publication
# MAGIC 
# MAGIC With the above logic running, we now have the ability to verify data moving through our streaming workflow.  Each of the tables in our Bronze, Silver and Gold layers is queriable using simple *SELECT* statements or using the pyspark Spark SQL API.  You'll want to review these data to ensure information is being recorded appropriately.  We've elected to skip that step as it can take a few seconds for data from our streaming queries to first start appearing in the tables as the streams are initialized.
# MAGIC 
# MAGIC Looking at the *ingest_ts* and *current_ts* fields in each table, you can get a sense of how quickly data is flowing from our ingest layer to a particular layer in our medallion architecture.  We have observed in some places that data appears to have arrived in our tables ahead of it arriving in the ingest layer.  We believe this reflects differences in clock synchronizations between different cloud services.

# COMMAND ----------

# MAGIC %md To examine the data in the online feature store, you will need to connect to your CosmosDB instance and review the data housed in containers within the database identified in the online feature store spec. Each container will be named for the offline feature store table to which it was mapped at the time of publication.
# MAGIC 
# MAGIC To facilitate this, the CosmosDB service makes available a [Data Explorer utility](https://learn.microsoft.com/en-us/azure/cosmos-db/data-explorer) through the Azure portal UI.  You can see the *ingest_ts* and *current_ts* timestamps originating from the offline feature store table in the individual documents. There is also a *_ts* field in the document written by the CosmosDB service with each document insert/update.  To convert this value to a readable date-time value like the ones shown in the *ingest_ts* and *current_ts* fields, you can use a query like this:
# MAGIC </p>
# MAGIC 
# MAGIC ```
# MAGIC SELECT 
# MAGIC     c.ingest_ts,
# MAGIC     c.current_ts,
# MAGIC     TimestampToDateTime(c._ts * 1000) as cosmosdb_ts
# MAGIC FROM c
# MAGIC ORDER BY c._ts DESC
# MAGIC ```

# COMMAND ----------

# MAGIC %md ##Step 6: Maintenance
# MAGIC 
# MAGIC Our stream processes are writing new records to the feature store tables with each new user session created.  To avoid our feature tables holding onto data they no longer need, we should setup a process to periodically remove expired records from them.
# MAGIC 
# MAGIC For the offline feature tables, this is pretty straightforward.  Simply write a [DELETE](https://docs.databricks.com/sql/language-manual/delta-delete-from.html) statement with an appropriate filter such as *event_time < 'YYYY-MM-DD'* where *YYYY-MM-DD* is the string representation of a date far enough back in time that the session could not be valid/relevant.
# MAGIC 
# MAGIC For the online feature tables, this is a bit trickier.  For this, we can set a time to live (TTL) on the container which serves as a default policy.  After the number of time to live seconds has expired on any given document, that document will be removed from the container through a background process that the Azure service runs for us.  You could do something more sophisticated by calculating a TTL for each individual feature record which the service would then honor but this might be overkill.  To set a container-wide TTL, follow [these steps](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/how-to-time-to-live?tabs=dotnet-sdk-v3#enable-time-to-live-on-a-container-using-the-azure-portal).

# COMMAND ----------

# MAGIC %md To prevent you from forgetting to turn off this loop accidentally, we shut down the streams after 30 minutes. You may remove this line if you intend to leave the streams running indefinitely.

# COMMAND ----------

time.sleep(1800)
for s in spark.streams.active:
  s.stop()

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
