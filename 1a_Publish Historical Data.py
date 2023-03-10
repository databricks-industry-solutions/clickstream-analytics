# Databricks notebook source
# MAGIC %md The purpose of this notebook is to publish the historical data and features for use in the Clickstream Propensity solution accelerator.  This notebook was developed on a **Databricks ML 12.1** cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC This notebook represents the first step in the model training portion of our solution accelerator. In it, we will persist the first four months of a 5-month dataset and generate from these *historical* data the features required to train our model.  We will employ a [medallion architecture](https://www.databricks.com/glossary/medallion-architecture) as is standard in a lakehouse environment which will simplify our data management processes as we move towards operationalization in later notebooks.

# COMMAND ----------

# DBTITLE 1,Get Config Info
# MAGIC %run "./CS 0a: Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

# COMMAND ----------

# MAGIC %md ## Step 1: To Bronze
# MAGIC 
# MAGIC The first layer of the medallion architecture is the Bronze layer.  In it, we persist raw data inputs in a format as close as is possible to what has been delivered by our data source system.  This data is persisted in an untransformed state, allowing us to replay downstream workflows should we need to.
# MAGIC 
# MAGIC In a streaming architecture (which is what we will implement later in this solution accelerator), the Bronze layer is often the raw event message acquired from the Kafka or other streaming ingest service.  In the processing of our historical data, there's little value in recreating the data in the Kafka message format so for this portion of the solution accelerator, we will be skipping the Bronze layer implementation. 

# COMMAND ----------

# MAGIC %md ##Step 2: Bronze to Silver
# MAGIC 
# MAGIC In the Silver layer, we apply technical transformations to the data that improve its accessibility without changing the information value in the data.  The Silver layer is often the focal point for Data Scientists interested in exploring new and novel interpretations of our data.  It also is the layer from which most business-aligned assets (in the Gold layer) are derived.
# MAGIC 
# MAGIC While we don't like to alter the data, we will add a timestamp to the data with which we can better assess its movement through our architecture.  This isn't important for the historical dataset but will be important when we start streaming *live* data later in the solution accelerator:

# COMMAND ----------

# DBTITLE 1,Retrieve Staged Historical Events Data
historical_events = (
  spark
    .table('staged_events')
    .filter(fn.expr("data_set='historical'")) # get historical data only
    .withColumn('ingest_ts', fn.expr("CAST('1970-01-01' AS TIMESTAMP)"))
    .withColumn('current_ts', fn.expr("CURRENT_TIMESTAMP()"))
  )

# COMMAND ----------

# DBTITLE 1,Write Historical Events Data tp Silver Layer Table
_ = (
  historical_events
    .write
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('electronics_events_silver')
  )

# COMMAND ----------

# DBTITLE 1,Display Silver Layer Events Data
display(
  spark
    .table('electronics_events_silver')
  )

# COMMAND ----------

# MAGIC %md ##Step 3: Silver to Gold
# MAGIC 
# MAGIC In the medallion architecture, the Gold layer is the principal area of focus of business-oriented analytics and operational needs. Here we may apply transformations to our data which interpret and enhance the information in ways that are agreed to by some or all of the business but which may be subject to change over time. With our dataset, we will want to present our event data as well as metrics from across different levels of granularity that will serve as the basis for various monitoring efforts as well as the generation of features for our model.
# MAGIC 
# MAGIC In thinking about the metrics/features we might derive, it seems that we might derive values for different entities and entity-combinations associated with these events.  These might include metrics for:</p>
# MAGIC 
# MAGIC * user
# MAGIC * product
# MAGIC * user-product
# MAGIC * session (cart)
# MAGIC * session-product (cart-product)
# MAGIC 
# MAGIC **NOTE** We will be using the terms *session*, *shopping session*, *cart* and *shopping cart* interchangeably throughout this accelerator.  A *session, cart, etc.* is identified by the combination of the *user_id* and *user_session* fields.  (We found that the strings used in the *user_session* were re-used between users in this dataset.)
# MAGIC 
# MAGIC The easiest ones to start with are the cart-level metrics.

# COMMAND ----------

# MAGIC %md With the cart-level metrics, we need to calculate the metrics for that cart as they would exist with each event.  While there are several ways to do this, we have found that applying aggregations across windows defined for the cart, *i.e.* the combination of the *user_id* and the *user_session*, from the beginning of the session up to the point in time of the event provides us the easiest way to compute these:

# COMMAND ----------

# DBTITLE 1,Cart-Level Metrics 
window_def = 'PARTITION BY user_id, user_session ORDER BY event_time'

cart_metrics = (
  spark
    .table('electronics_events_silver')
    .select('user_id','user_session','event_time','event_type')
    .withColumn('events',fn.expr(f"COUNT(*) OVER({window_def})"))
    .withColumn('views',fn.expr(f"COUNT_IF(event_type='view') OVER({window_def})"))
    .withColumn('carts',fn.expr(f"COUNT_IF(event_type='cart') OVER({window_def})"))
    .withColumn('purchases',fn.expr(f"COUNT_IF(event_type='purchase') OVER({window_def})"))
    .drop('event_type')
    .withColumn('view_to_events', fn.expr("views/events"))
    .withColumn('carts_to_events', fn.expr("carts/events"))
    .withColumn('purchases_to_events', fn.expr("purchases/events"))
    .withColumn('carts_to_views', fn.expr("carts/views"))
    .withColumn('purchases_to_views', fn.expr("purchases/views"))
    .withColumn('purchases_to_carts', fn.expr("purchases/carts"))
  )

_ = (
   cart_metrics
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('electronics_cart_metrics__training')
    )

display(spark.table('electronics_cart_metrics__training').orderBy('user_id','user_session','event_time'))

# COMMAND ----------

# MAGIC %md Reviewing the last cell, you can see we are keeping our features pretty simple.  You could certainly add metrics examining time between events, cart size, cart value, *etc.* but we've purposefully decided to keep things minimal on this first pass through the solution.
# MAGIC 
# MAGIC Using a very similar approach, we might now turn our attention to the cart-product metrics, which track information related to the state of a product in the context of a given shopping session:

# COMMAND ----------

# DBTITLE 1,Cart-Product Metrics
window_def = 'PARTITION BY user_id, user_session, product_id ORDER BY event_time'

cart_product_metrics = (
  spark
    .table('electronics_events_silver')
    .select('user_id','user_session','product_id','event_time','event_type')
    .withColumn('events',fn.expr(f"COUNT(*) OVER({window_def})"))
    .withColumn('views',fn.expr(f"COUNT_IF(event_type='view') OVER({window_def})"))
    .withColumn('carts',fn.expr(f"COUNT_IF(event_type='cart') OVER({window_def})"))
    .withColumn('purchases',fn.expr(f"COUNT_IF(event_type='purchase') OVER({window_def})"))
    .drop('event_type')
    .withColumn('view_to_events', fn.expr("views/events"))
    .withColumn('carts_to_events', fn.expr("carts/events"))
    .withColumn('purchases_to_events', fn.expr("purchases/events"))
    .withColumn('carts_to_views', fn.expr("carts/views"))
    .withColumn('purchases_to_views', fn.expr("purchases/views"))
    .withColumn('purchases_to_carts', fn.expr("purchases/carts"))
  )

_ = (
   cart_product_metrics
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('electronics_cart_product_metrics__training')
    )

display(spark.table('electronics_cart_product_metrics__training').orderBy('user_id','user_session','product_id','event_time'))

# COMMAND ----------

# MAGIC %md Now, we can address the remaining metrics. As we consider how we will eventually operationalize these metrics, we have to wonder if the cost of recomputing features for a user, product or user-product combination in real-time is really worthwhile. Once a user or product has been associated with a large number of events, the impact of a single event on its metrics is pretty minimal.  Certainly while a user or product is new to the system, individual events have more bearing on the metrics for that item but we suspect that most users and product metrics will achieve a relatively stable state over time.  (User-product combinations may be much more sparse and less likely to *plateau* the same way but we'll lump them into our user and product approaches for simplicity.) With this in mind, it might take a considerable load off the system if we recalculated user, product and user-product metrics on a batch cycle, maybe right after midnight each day.  While these metrics do get stale as the day progresses, we believe they should be *good enough* for inference between update cycles.
# MAGIC 
# MAGIC If we take this batch-oriented approach to creating these features, we now need to calculate user, product and user-product metrics as they would appear at midnight of each day for which we would have event data associated with these.  To help with this, we will truncate each *event_time* value to midnight at the start of that date and perform our windowing against that time, *i.e.* *batch_time*:

# COMMAND ----------

# DBTITLE 1,User Metrics
# define the window of accessible data for each event record
window_def = 'PARTITION BY user_id ORDER BY batch_time'

# calculate metrics
user_metrics = (
  spark
    .table('electronics_events_silver')
    .select('user_id','event_time','event_type')
    .withColumn('batch_time', fn.expr("DATE_TRUNC('day', DATE_ADD(event_time, 1))")) # set batch_time to midnight of following day
    .groupBy('user_id','batch_time') # get summary metrics for each date
      .agg(
        fn.expr("COUNT(*)").alias('events'),
        fn.expr("COUNT_IF(event_type='view')").alias('views'),
        fn.expr("COUNT_IF(event_type='carts')").alias('carts'),
        fn.expr("COUNT_IF(event_type='purchase')").alias('purchases')
        )
    .withColumn('events',fn.expr(f"SUM(events) OVER({window_def})")) # aggregate from prior periods
    .withColumn('views',fn.expr(f"SUM(views) OVER({window_def})"))
    .withColumn('carts',fn.expr(f"SUM(carts) OVER({window_def})"))
    .withColumn('purchases',fn.expr(f"SUM(purchases) OVER({window_def})"))
    .withColumn('view_to_events', fn.expr("views/events"))
    .withColumn('carts_to_events', fn.expr("carts/events"))
    .withColumn('purchases_to_events', fn.expr("purchases/events"))
    .withColumn('carts_to_views', fn.expr("carts/views"))
    .withColumn('purchases_to_views', fn.expr("purchases/views"))
    .withColumn('purchases_to_carts', fn.expr("purchases/carts"))
  )

# write metrics to gold-layer table
_ = (
   user_metrics
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('electronics_user_metrics__training')
    )

# display saved metrics
display(
  spark.table('electronics_user_metrics__training').orderBy('user_id','batch_time')
  )

# COMMAND ----------

# DBTITLE 1,Product Metrics
# define the window of accessible data for each event record
window_def = 'PARTITION BY product_id ORDER BY batch_time'

# calculate product-level metrics
product_metrics = (
  spark
    .table('electronics_events_silver')
    .select('product_id','event_time','event_type')
    .withColumn('batch_time', fn.expr("DATE_TRUNC('day', DATE_ADD(event_time, 1))")) # set batch_time to midnight of following day
    .groupBy('product_id','batch_time') # get summary metrics for each date
      .agg(
        fn.expr("COUNT(*)").alias('events'),
        fn.expr("COUNT_IF(event_type='view')").alias('views'),
        fn.expr("COUNT_IF(event_type='carts')").alias('carts'),
        fn.expr("COUNT_IF(event_type='purchase')").alias('purchases')
        )    
    .withColumn('events',fn.expr(f"SUM(events) OVER({window_def})")) # aggregate from prior periods
    .withColumn('views',fn.expr(f"SUM(views) OVER({window_def})"))
    .withColumn('carts',fn.expr(f"SUM(carts) OVER({window_def})"))
    .withColumn('purchases',fn.expr(f"SUM(purchases) OVER({window_def})"))
    .withColumn('view_to_events', fn.expr("views/events"))
    .withColumn('carts_to_events', fn.expr("carts/events"))
    .withColumn('purchases_to_events', fn.expr("purchases/events"))
    .withColumn('carts_to_views', fn.expr("carts/views"))
    .withColumn('purchases_to_views', fn.expr("purchases/views"))
    .withColumn('purchases_to_carts', fn.expr("purchases/carts"))
    )

# persist metrics
_ = (
   product_metrics
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('electronics_product_metrics__training')
    )

# display metrics
display(spark.table('electronics_product_metrics__training').orderBy('product_id','batch_time'))

# COMMAND ----------

# DBTITLE 1,User-Product Metrics
window_def = 'PARTITION BY user_id, product_id ORDER BY batch_time'

user_product_metrics = (
  spark
    .table('electronics_events_silver')
    .select('user_id','product_id','event_time','event_type')
    .withColumn('batch_time', fn.expr("DATE_TRUNC('day', DATE_ADD(event_time, 1))")) # set batch_time to midnight of following day
    .groupBy('user_id','product_id','batch_time') # get summary metrics for each date
      .agg(
        fn.expr("COUNT(*)").alias('events'),
        fn.expr("COUNT_IF(event_type='view')").alias('views'),
        fn.expr("COUNT_IF(event_type='carts')").alias('carts'),
        fn.expr("COUNT_IF(event_type='purchase')").alias('purchases')
        )    
    .withColumn('events',fn.expr(f"SUM(events) OVER({window_def})")) # aggregate from prior periods
    .withColumn('views',fn.expr(f"SUM(views) OVER({window_def})"))
    .withColumn('carts',fn.expr(f"SUM(carts) OVER({window_def})"))
    .withColumn('purchases',fn.expr(f"SUM(purchases) OVER({window_def})"))   
    .withColumn('view_to_events', fn.expr("views/events"))
    .withColumn('carts_to_events', fn.expr("carts/events"))
    .withColumn('purchases_to_events', fn.expr("purchases/events"))
    .withColumn('carts_to_views', fn.expr("carts/views"))
    .withColumn('purchases_to_views', fn.expr("purchases/views"))
    .withColumn('purchases_to_carts', fn.expr("purchases/carts"))
    )

_ = (
   user_product_metrics
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('electronics_user_product_metrics__training')
    )

display(spark.table('electronics_user_product_metrics__training').orderBy('user_id','product_id','batch_time'))

# COMMAND ----------

# MAGIC %md While we are creating these metrics for our initial model training, it's important to note that we will only be keeping the *current state* of these metrics as we move into the inference phase of our work.  That means that whenever our Data Scientists wish to train/re-train a model using these data, they will need to re-run the logic above to generate new inputs from the latest data in the Silver layer of our lakehouse. It's for this reason that we've named each of the tables to which these metrics are being recorded using a *\__training* suffix.  We will use a *\__inference* suffix in the next phase of this solution accelerator to keep the *current state* metrics separate from the historical state metrics used for training purposes.

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
