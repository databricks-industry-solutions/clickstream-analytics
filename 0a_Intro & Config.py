# Databricks notebook source
# MAGIC %md The purpose of this notebook is to introduce the Clickstream Propensity solution acclerator and to provide access to configuration information for the notebooks supporting it. You may find this notebook at https://github.com/databricks-industry-solutions/clickstream-analytics

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC In ecommerce platforms, the path to a purchase is not always as straightforward as we might expect.  A customer may click on a product, put it in their cart, and immediately make a purchase, but more typically, especially with more expensive or complex items, the customer will browse, compare, leave the site for a while, return to give the product more consideration, and do all this multiple times before completing the purchase.  Quite often, customers will engage in this process and never complete the pruchase, instead, abandoning the cart altogether, having either made a purchase through another channel or becoming disinterested in the product.
# MAGIC 
# MAGIC As we observe customers engage the site, we may wish to estimate the liklihood the customer will purchase the particular item with which they are engaging. A number of factors related to the product, the customer, prior interactions between the customer and the particular product, and patterns of engagement observed in the context of the user's shopping session all contribute to our understanding of the customer's intent.
# MAGIC 
# MAGIC If we see a customer has a low probability of completing the pruchase of a product, we may use this as a signal that we should suggest alternative products which may be of more interest to that customer.  If we see the customer has a high probability of completing a purchase, we may take the opportunity to suggest complementary products that will boost the size of the sale.  And somewhere in the middle, we might use discounts or promotional offers to nudge the customer forward, securing the sale.
# MAGIC 
# MAGIC In this solution accelerator, we leverage event data from an electronics ecommerce site to demonstrate how such data may be used to estimate these propensities.  We first use a historical backlog of these data to train a model.  We then implement a real-time stream to calculate features supporting inference with each page clicked.  You can think of the accelerator as divided into these two parts where in the first we focus on understanding the data and how we might approach the Data Science portion of our work.  In the second, the focus is on the mechanics of building a real-time inference layer.  Before diving into those parts (identify by the number associated with different notebooks), be sure to run notebook *0b* to stage the data required for both parts.
# MAGIC 
# MAGIC **NOTE** This accelerator uses the newly released Databricks Model Serving capability which is avaiable in only select regions.  Please consult the feature documentation to determine if your workspace is in a supported cloud and region ([Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/#--region-availability), [AWS](https://docs.databricks.com/machine-learning/model-serving/index.html#region-availability)) before deploying the accelerator.

# COMMAND ----------

# MAGIC %md ###Understanding Part 1: Model Training
# MAGIC 
# MAGIC Part 1 of this accelerator is comprised of two notebooks named *1a* and *1b*.  In the first of these notebooks, we generate point-in-time features from historical clickstream data that we've accumulated.  In setting up the data, we pretend as if that data has been accumulated through a workflow organized around a medallion architecture.  
# MAGIC 
# MAGIC The medallion architecture, explained in more depth in notebook *1a*, divides our data processing workflow into three high-level stages.  
# MAGIC </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/cs_part_1.png' width=750>
# MAGIC 
# MAGIC In the first stage, known as the Bronze layer, we accumulate data in the raw, unaltered format with which it is presented to Databricks.  We have invisioned a production architecture where data is sent from our website to an Azure Event Hub ingest layer and its from the Event Hub layer that Databricks reads the data.  Because the Bronze data would reflect the default data structure associated with the Event Hub and not necessarily the structure of the data sent to it, we skip this step in Part 1.  However, we do write data to a table representing our Silver layer in this part of the accelerator.
# MAGIC 
# MAGIC The Silver layer, the second layer of the medallion architecture, represents the data transformed for technical accessiblity but with no business-level interpretation attached.  This is often the starting point for Data Scientists looking for high-fidelity data and for Data Engineers looking to build business-aligned assets that will comprise the Gold layer of our architecture.
# MAGIC 
# MAGIC For our Gold layer, we derive 5 sets of metrics from our historical event data.  These data are landed in tables named for the feature set they are intended to represent and are assigned a *\__training* suffix to indicate these are generated to support model training and should not be used for live inference. The reason we need to separate these records from our inference records is that for training we recreate our understanding of each user, product and shopping cart's state with each individual event that occurs.  This represents the full body of knowledge we might wish to use during model training whereas during inference we only care about the latest known state for these.  
# MAGIC 
# MAGIC For those curious how these tables would be managed during retraining events, they would need to be recreated from the Silver layer data available at that time. Once we have implemented the workflows showin in Part 2, we would have a real-time process populating our Bronze and Silver layer tables so that the one-time staging of historical data in the Silver tables performed in Part 1 of the accelerator would not be repeated.

# COMMAND ----------

# MAGIC %md ###Understanding Part 2: Real-Time Inference
# MAGIC 
# MAGIC Part 2 of this accelerator is comprised to four notebooks named *2a*, *2b*, *2c* and *2d*. The first three tackle the data engineering workflow that moves data through the medallion architecture.  The fourth notebook addresses the deployment of our trained model for real-time inference.
# MAGIC </p>
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/cs_part_2.png' width=1000>
# MAGIC 
# MAGIC In order to simulate data flowing in real-time from a website, we replay the last month of the 5-month dataset (the first four of which were used in the Part 1 notebooks).  Notebook *2a* provides the replay logic and writes data to an Azure Event Hub you will need to provision in advance of running the notebook.  The data is played back in accelerated real-time so that you can replay the period in less than one month's time.
# MAGIC 
# MAGIC Notebooks *2b* and *2c* represent the streaming and batch portions of our feature engineering workflow.  As we will explain in notebook *1a*, it doesn't make much sense for us to compute absolutely all features in real-time.  By separating our logic this way, we hope to provide a suggested compromise for similar workflows that will help you reduce complexity and the cost of processing your data. In both these notebooks, we publish data to Gold layer tables (named using a *\__inference* suffix) which we then register with our feature store.  We employ both offline feature store and online feature store capabilities.  The online feature store capability facilitates real-time inference but does require you to deploy an Azure CosmosDB instance and configure a database within it before you run these notebooks
# MAGIC 
# MAGIC Notebook *2d* addresses model deployment.  In this notebook, we need to revisit the model trained in Part 1 and associate it with the feature store tables created in Part 2.  This is an atypical deployment pattern but one necessitated by the fact that we use a different set of feature tables during model training than the ones intended for model inference.

# COMMAND ----------

# MAGIC %md ## Configuration Settings
# MAGIC 
# MAGIC The following represent configuration settings used across various notebooks in this solution accelerator.  You should read through all the notebooks to understand how the configuration settings are used before making any changes to the values below.

# COMMAND ----------

# DBTITLE 1,Instantiate Config Variable
if 'config' not in locals().keys():
  config = {}

# COMMAND ----------

# DBTITLE 1,Database
# database name
config['database'] = 'clickstream'

# create database if not exists
_ = spark.sql('create database if not exists {0}'.format(config['database']))

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# DBTITLE 1,Storage (see notebook 0b for more info)
# mount point path - here we use a tmp folder to allow the notebook to run without setting up any mount point. If you would like to use your own storage account, consider setting up mount points 
# https://docs.databricks.com/dbfs/mounts.html
# https://learn.microsoft.com/en-us/azure/databricks/dbfs/mounts

config['mount_point'] ='/tmp/clickstream'

# file paths
config['events_path'] = config['mount_point'] + '/electronics'
config['checkpoint_path'] = config['mount_point'] + '/checkpoints'

# COMMAND ----------

# DBTITLE 1,Streaming Ingest (see notebook 2a for more info)
# azure event hub sas policy connection string
# here we use a secret scope to hold the connection string; see the RUNME notebook after import to find instructions for how to set up this secret 
event_hub_sas_policy_connection_string = dbutils.secrets.get("solution-accelerator-cicd", "event_hub_sas_policy_connection_string") 

# helper function to convert connection string (above) into dictionary of parameters
def split_connstring(connstring):
  conn_dict = {}
  for kv in connstring.split(';'):
    k,v = kv.split('=',1)
    conn_dict[k]=v
  return conn_dict
  
# split conn strings
eh_conn = split_connstring( event_hub_sas_policy_connection_string )

# extracted values from connection string
eh_namespace = eh_conn['Endpoint'].split('.')[0].split('://')[1] 
config['eh_kafka_topic'] = eh_conn['EntityPath']
config['eh_bootstrap_servers'] = '{0}.servicebus.windows.net:9093'.format(eh_namespace)
config['eh_sasl'] = 'kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required username=\"$ConnectionString\" password=\"Endpoint={0};SharedAccessKeyName={1};SharedAccessKey={2}\";'.format(eh_conn['Endpoint'], eh_conn['SharedAccessKeyName'], eh_conn['SharedAccessKey'])

# COMMAND ----------

# DBTITLE 1,Online Feature Store (see notebook 2b for more info)
# azure cosmosdb uri
# here we use a secret scope to hold the cosmosdb uri; see the RUNME notebook after import to find instructions for how to set up this secret 
config['cosmosdb_uri'] = dbutils.secrets.get("solution-accelerator-cicd", "cosmosdb_uri")  

# secret scopes
config['scope_readonly'] = 'clickstream-readonly'
config['scope_readwrite'] = 'clickstream-readwrite'

# secret prefixes
config['secret_prefix'] = 'onlinefs'

# COMMAND ----------

# DBTITLE 1,Model (see notebooks 1b & 2d for more info)
config['model name'] = 'clickstream'
config['inference_model_name'] = config['model name']+'__inference'

# COMMAND ----------

# DBTITLE 1,Get API token info from notebook context (see notebook 2d for more info)
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
config['databricks token'] = ctx.apiToken().getOrElse(None)
config['databricks url'] = ctx.apiUrl().getOrElse(None)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
