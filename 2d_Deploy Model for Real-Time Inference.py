# Databricks notebook source
# MAGIC %md The purpose of this notebook is to demonstrate how our trained model can be deployed for real-time model inference. This notebook was developed on a **Databricks 12.3** cluster with **Photon Acceleration** enabled.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC Many notebooks back, we trained a model to enable product-purchase propensity scoring with each website event. With our features being calculated from the live, incoming event stream, we can now turn our attention to the deployment of the model for real-time inference.
# MAGIC 
# MAGIC For real-time inference to work, we need our model to have knowledge of how to access the *real-time* features hosted in our online feature store. The typical pattern we use for this is to train our model with features pulled from feature store tables and then log that model via the feature store to mlflow.  By using this pattern, we embed with the model the instructions required to retrieve features from the feature store when nothing more than primary key elements for the feature store tables are provided, greatly simplifying the deployment and integration process.
# MAGIC 
# MAGIC But we didn't train our earlier model using feature store data.  The primary reason for this is that the historical features on which we trained our model contains information for each event record (at least for our cart and cart-product features) while our real-time features on which we wish to perform inference are only keeping up with the latest state of our data.  If we had decided to preserve each change in state with our real-time features, the concern was that our feature store might become so large (even with regular maintainance) that performance would suffer.  So, we kept our historical features separate in a series of tables named with a suffix of *\__training* while keeping our *real-time* features in a different set of tables named with a suffix of *\__inference*.
# MAGIC 
# MAGIC Given this choice, we now need to take our model training on our historical data and associate it with our feature store tables before we then deploy it to our model serving layer for real-time inference.

# COMMAND ----------

# DBTITLE 1,Install Feature Store Components
# MAGIC %pip install xgboost
# MAGIC %pip install databricks-feature-store

# COMMAND ----------

# DBTITLE 1,Get Config Info
# MAGIC %run "./CS 0a: Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from databricks import feature_store

import pyspark.sql.functions as fn

import mlflow
from xgboost import XGBClassifier

# COMMAND ----------

# MAGIC %md ##Step 1: Connect Model with Feature Store
# MAGIC 
# MAGIC To connect our model to the feature store, we first need to assemble a [training set](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/train-models-with-feature-store) representing the details of which features are employed by our model. Please note that we are generating our feature lookups in a manner similar to how we assembled our features during training.  As part of this, we are renaming our features using the pattern of *\<table_name\>\__\<feature_name\>*, leveraging the *rename_outputs* argument which is assigned a dictionary mapping original feature names to output feature names.

# COMMAND ----------

# DBTITLE 1,Connect to Feature Store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# DBTITLE 1,Define Feature Lookups
# these are columns in feature store tables we do not want to use as features
columns_to_exclude_from_features = ['user_id','user_session','product_id','ingest_ts','current_ts','event_time']

# define how records in feature tables are retrieved 
metrics_map = { # table_name, key fields
  'electronics_cart_metrics__inference':['user_id','user_session'],
  'electronics_cart_product_metrics__inference':['user_id','user_session','product_id'],
  'electronics_user_metrics__inference':['user_id'],
  'electronics_product_metrics__inference':['product_id'],
  'electronics_user_product_metrics__inference':['user_id','product_id']
  }

# attach metrics to each event record
feature_lookups = []
for table_name, key_fields in metrics_map.items():
  
  # define prefix on feature names
  prefix = prefix = table_name.split('__')[0] # prefix is everything prior to the __
  
  # define feature lookup
  feature_lookups += [
    feature_store.FeatureLookup(
      table_name = f"{config['database']}.{table_name}",
      lookup_key = key_fields,
      feature_names = [c for c in spark.table(f"{config['database']}.{table_name}").columns if c not in columns_to_exclude_from_features],
      rename_outputs = {c:f"{prefix}__{c}" for c in spark.table(f"{config['database']}.{table_name}").columns if c not in columns_to_exclude_from_features}
      )
    ]
  
# display info about set of feature lookups
feature_lookups

# COMMAND ----------

# MAGIC %md The feature lookup is used to retrieve features against a pre-defined set of keys.  It is often used first during the model training cycle against a set of keys and a label associated with our training set. Once training is completed, these keys and label are used (in combination with our feature lookup definition) to define a training set to stitch together all the metadata that ties our model to the feature store.  
# MAGIC 
# MAGIC With that in mind, let's define a sample training dataset:
# MAGIC 
# MAGIC **NOTE** We are using the sort on *event_time* to better ensure we are getting more recently observed shopping carts.

# COMMAND ----------

# DBTITLE 1,Define Sample Training Data 
training_df = ( 
  spark
    .table('electronics_events_silver')
    .select('user_id','user_session','product_id','event_type','event_time')
    .groupBy('user_id','user_session','product_id')
      .agg(
        fn.max(fn.expr("case when event_type='purchase' then 1 else 0 end")).alias('purchased'),
        fn.max('event_time').alias('event_time') 
        )
    .orderBy('event_time', ascending=False)
    .limit(10)
    .drop('event_time')
    )

display(training_df)

# COMMAND ----------

# MAGIC %md Using our sample training dataset and our feature lookups, we now define the training set that represents our linkage between incoming data and features in the feature store:

# COMMAND ----------

# DBTITLE 1,Define Training Set
training_set = fs.create_training_set(
  training_df,
  feature_lookups=feature_lookups,
  label='purchased',
  exclude_columns = columns_to_exclude_from_features
  )

# COMMAND ----------

# MAGIC %md We then need to retrieve our model, re-log it with the feature set information and move it to production:
# MAGIC 
# MAGIC **NOTE** Please note that the *log_model* method provided by the feature store does not directly support the *pyfunc_predict_fn* argument used in *CS 1b*.  However, it does allow additional arguments to be passed in using a kwarg which we are using to specify this option.

# COMMAND ----------

# DBTITLE 1,Retrieve Model
# connect to mlflow
client = mlflow.tracking.MlflowClient()

# retrieve model
model = mlflow.sklearn.load_model(model_uri=f"models:/{config['model name']}/production")

# COMMAND ----------

# DBTITLE 1,Re-Log Model with Feature Store Metadata
_ = fs.log_model(
  model,
  artifact_path='model',
  flavor=mlflow.sklearn,
  training_set=training_set,
  registered_model_name=config['model name']+'__inference',
  **{'pyfunc_predict_fn':'predict_proba'}
  )

# COMMAND ----------

# DBTITLE 1,Move Re-Logged Model to Production Status
# connect to mlflow
client = mlflow.tracking.MlflowClient()

# identify model version in registry
model_version = client.search_model_versions(f"name='{config['model name']}__inference'")[0].version

# move model version to production
client.transition_model_version_stage(
  name=config['model name']+'__inference',
  version=model_version,
  stage='production'
  )      

# COMMAND ----------

# MAGIC %md ##Step 2: Deploy Model to Serving Layer
# MAGIC 
# MAGIC With the model now registered with feature store information, we can now deploy it to the Databricks model serving infrastructure.  To do this, we need to create a Serving endpoint using [these steps](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints). Be sure to configure the Databricks UI for *Machine Learning* (instead of using the default *Data Science & Engineering* configuration) in order to more easily access the *Serving* icon in the sidebar UI.  When selecting your model, be sure to select the instance using the *\__inference* suffix.  Scale the compute per your requirements (though we used a Small configuration with *Scale to zero* deselected for our testing).
# MAGIC 
# MAGIC Pleaese wait until the endpoint is fully deployed and running before proceeding to the next step:
# MAGIC </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/cs_model_serving.PNG' width=750>

# COMMAND ----------

# MAGIC %md ##Step 3: Verify Model Deployment
# MAGIC 
# MAGIC Before moving away from the model serving endpoint UI, click the Query Endpoint button in the upper right-hand corner of the UI and copy the Python code in the resulting popup.  You can paste a copy of this code in the cell below:

# COMMAND ----------

# DBTITLE 1,Paste the Query Endpoint | Python Code Here
import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://adb-1883264859880764.4.azuredatabricks.net/serving-endpoints/clickstream/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 
'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return response.json()

# COMMAND ----------

# MAGIC %md Before you can run the code above, you need to create an environmental variable named *DATABRICKS_TOKEN* that has been assigned a valid [personal access token](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/authentication):

# COMMAND ----------

# DBTITLE 1,Set Personal Access Token to Access the Model Serving Endpoint
os.environ['DATABRICKS_TOKEN'] = 'dapic57135894fd55c541501569ba9938871'

# COMMAND ----------

# MAGIC %md Running the pasted code you have now defined a couple functions with which you can test your endpoint:

# COMMAND ----------

# DBTITLE 1,Assemble Set of Keys to Score
# assemble lookup keys to score
training_pd = (
  training_df
    .drop('purchased')
  ).toPandas()

display(training_pd)

# COMMAND ----------

# DBTITLE 1,Score the Keys
score_model(training_pd)

# COMMAND ----------

# MAGIC %md You should notice that unlike the Spark UDF registered in notebook *CS 1b*, the model serving endpoint returns the full array of probabilities, the second of which is the probability of a purchase, the positive class.

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
