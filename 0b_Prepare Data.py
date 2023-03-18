# Databricks notebook source
# MAGIC %md The purpose of this notebook is to prepare the data for use in the Clickstream Propensity solution accelerator.  You may find this notebook at https://github.com/databricks-industry-solutions/clickstream-analytics

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC In this notebook, we will access the ecommerce events data around which we will train our model and simulate live user activity on a website.  The data set we will use will be divided into *historical* and *real-time* portions to be used in different parts of our work.  In addition to the partitioning of the dataset, we will spend some time examining the data so that decisions made during feature engineering and model training might be clearer. 

# COMMAND ----------

# MAGIC %md ##Step 1: Data Preparation
# MAGIC 
# MAGIC We will use the *[eCommerce events history in Electronics Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store?resource=download)* dataset available from the Kaggle website. This dataset captures data for various events taking place on an e-commerce website and is limited to those events associated with products in the electronics department.  (Other datasets associated with this include interactions with products from other departments.) 
# MAGIC 
# MAGIC Each record in the dataset captures information for the following fields:
# MAGIC </p>
# MAGIC 
# MAGIC * event_time - time when event happened at (in UTC)
# MAGIC * event_type - the type of event represented by this record, *i.e.* *view*, *cart*, or *purchase* as described above
# MAGIC * product_id - the id associated with the product
# MAGIC * category_id - the category id associated with the product
# MAGIC * category_code - the product's category taxonomy (code name) if it was possible to make it
# MAGIC * brand - the down-cased string of brand name
# MAGIC * price - the float price of a product
# MAGIC * user_id - the user's permanent id
# MAGIC * user_session - the temporary session id for this user
# MAGIC 
# MAGIC This information is not representative of a raw clickstream as would be generated from a web server but instead represents filtered and enhanced data that could be derived with some downstream processing of the raw weblogs in combination with information about products in the company's product catalog. If you are interested in implementing a solution like the one demonstrated in these notebooks, you'll need to carefully examine the information available to you to determine how you might generate a similar dataset. (Raw web server logs are rarely made available for public use so this is an on-going challenge in any demonstration of analytics on online data. That said, the more condensed nature of this data set will allow us to focus on the more immediate challenges related to model training and inference.)
# MAGIC 
# MAGIC To make this data accessible, we have provided a script to download it from Kaggle and extract it to the `events_path`. You need to set up your Kaggle credentials for the download to work - check out the `RUNME` notebook after you import these notebooks for detailed instructions. If you wish to download your data to a different file location, please change the appropriate configuration settings under the *0a* notebook.

# COMMAND ----------

# DBTITLE 1,Download and Extract Source Data
# MAGIC %run ./util/data-extract

# COMMAND ----------

# DBTITLE 1,Get Config Info
# MAGIC %run "./0a_Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as fn

from databricks import feature_store

import time

# COMMAND ----------

# DBTITLE 1,Retrieve Events Data
# event data schema
events_schema = StructType([
  StructField('event_time', StringType()),
  StructField('event_type', StringType()),  
  StructField('product_id', LongType()),  
  StructField('category_id', LongType()),
  StructField('category_code', StringType()),
  StructField('brand', StringType()),  
  StructField('price', FloatType()),  
  StructField('user_id', LongType()),
  StructField('user_session', StringType())
  ])

# read events data
events = (
  spark
    .read
    .csv(
      config['events_path'],
      header=True,
      sep=',',
      schema=events_schema
      )
    .withColumn('event_time', fn.to_timestamp('event_time','y-M-d H:m:s z'))
  )

# make events dataframe a temporary view to assist with subsequent queries
events.createOrReplaceTempView('events')

# display events data
display( 
  spark.table('events').orderBy('event_time') 
  )

# COMMAND ----------

# MAGIC %md ##Step 2: Explore the Data
# MAGIC 
# MAGIC Within this dataset, the *event_type* field is of particular interest.  It contains three distinct values, *i.e.* *view*, *cart*, and *purchase*, indicating whether the event record is associated with a page view for a product, the placement of a product into a shopping cart or the purchase of a product, respectively:

# COMMAND ----------

# DBTITLE 1,Examine Event Types
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   event_type,
# MAGIC   COUNT(*) as occurances
# MAGIC FROM events
# MAGIC GROUP BY event_type

# COMMAND ----------

# MAGIC %md With each event, we track the user's identity as well as the shopping session (as identified by the combination of the *user_id* and *user_session*) that ties all these events together around a sales motion.  To understand how the event data relates to these concepts, let's examine the event records associated with a specific shopping session:

# COMMAND ----------

# DBTITLE 1,Examine Events During a Shopping Session
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   user_id,
# MAGIC   user_session,
# MAGIC   event_time,
# MAGIC   event_type,
# MAGIC   product_id,
# MAGIC   category_code,
# MAGIC   price
# MAGIC FROM events 
# MAGIC WHERE user_id=1515915625353900095 AND user_session='12dadbda-acb5-4f26-a5e8-f71814190c04'
# MAGIC ORDER BY event_time

# COMMAND ----------

# MAGIC %md In this session, we can see a user initially views a product and abandons the session for a few hours before returning to it.  There are several interactions with the product page for product 1703800 as well as other products before product 16237 is viewed and placed in the cart.  That item, along with another item that does not appear to have been viewed (so maybe it was presented as a *checkout lane* option) are purchased a few minutes later. There are several additional views of one of the purchased items that take place following the purchase event.  These may represent errors or the site may simply configure shopping sessions to extend to interactions on the site immediately following a purchase until a customer departs for some extended period of time.
# MAGIC 
# MAGIC The flow from view to purchase is not steady and linear.  Shopping sessions (again as identified by the *user_id* and the *user_session* fields in combination) extend over quite long periods of time and don't seem to close immediately following a purchase. Multiple products are viewed and only some (if any) are purchased.  And products not viewed can suddenly appear at the time of purchase.  Without more context, it's difficult for us to explain all these patterns with certainty, but still we should have the basis for building a model to predict product purchases.

# COMMAND ----------

# MAGIC %md Now we can examine the date range associated with this dataset:

# COMMAND ----------

# DBTITLE 1,Examine Date Range of Dataset
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   MIN(cast(event_time as date)) as min_date,
# MAGIC   MAX(cast(event_time as date)) as max_date
# MAGIC FROM events

# COMMAND ----------

# MAGIC %md Our goal will be to estimate the likelihood a customer will complete a purchase based on observed events in real-time.  As such, we will need to stream some portion of this data in order to simulate that real-time activity.  However, to train our model, we will need to accumulate some amount of historical data and it doesn't make sense to stream a long backlog of event records before we can engage in that part of the process.  Therefore, we'll arbitrarily consider all data prior to February 1, 2021 as historical data that's already been accumulated in the Databricks environment and focus our streaming and prediction efforts on the events for the subsequent dates.
# MAGIC 
# MAGIC It's important to note that shopping sessions can extend over multi-day periods so that it is very likely that numerous events records in our historical dataset will be part of a session that extends into our *real-time* period. It's not clear from the dataset how this website determines when a shopping session should be considered terminated so instead of making up some rules or otherwise looking into the "future" from our historical data, we'll be a little sloppy and act as if all the data in our historical partition is complete.  A more robust approach to partitioning this data may take a look at the last session observed for a user within some extended period prior to February 1, 2021, determine which ones were closed with purchases and otherwise exclude from consideration potentially open shopping carts as of that date, but we're going to keep things simple for this iteration of the project.

# COMMAND ----------

# MAGIC %md ##Step 3: Stage Events Data for Publication
# MAGIC 
# MAGIC To support one part of our data as historical data and another as *real-time* streaming data, we will quickly persist our dataset to a staging table.  The data will be used for internal processes and should not be considered part of what would be delivered in production system: 

# COMMAND ----------

# DBTITLE 1,Reset Database
# drop feature store tables
fs = feature_store.FeatureStoreClient()
for table in ['electronics_cart_metrics__inference', 'electronics_cart_product_metrics__inference', 'electronics_user_metrics__inference', 'electronics_product_metrics__inference', 'electronics_user_product_metrics__inference']: 
  try:
    _ = fs.drop_table(table)
  except:
    pass

# reset the database
_ = spark.sql(f"DROP DATABASE IF EXISTS {config['database']} CASCADE")
_ = spark.sql(f"CREATE DATABASE IF NOT EXISTS {config['database']}")
_ = spark.catalog.setCurrentDatabase(config['database'])

# reset any checkpoint files in existance
dbutils.fs.rm(config['checkpoint_path'], recurse=True)

# COMMAND ----------

# DBTITLE 1,Stage the Events Data for Biforcated Publication
_ = (
  events
    .withColumn('data_set', fn.expr("case when event_time < '2021-02-01' then 'historical' else 'real-time' end"))
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('staged_events')
  )

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
