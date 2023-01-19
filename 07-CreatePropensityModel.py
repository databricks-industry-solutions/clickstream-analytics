# Databricks notebook source
# MAGIC %md # Creating a Propensity Model 
# MAGIC 
# MAGIC ### Requirements
# MAGIC * Databricks Runtime 11.2 LTS for Machine Learning or above. 

# COMMAND ----------

from pyspark.sql.functions import col
from imblearn.under_sampling import RandomUnderSampler
import mlflow
from databricks.feature_store import FeatureLookup,FeatureStoreClient
from collections import Counter
import pandas as pd
import numpy as np

# COMMAND ----------

user_product_engage_metrics = 'sachin_clickstream.user_product_engage_metrics'
user_product_time_metrics  = 'sachin_clickstream.user_product_time_metrics'
lookup_key = ["product_id","user_id"]
target_col = 'is_purchase'
model_registry_name = 'propensity_model_puneet'

fs = FeatureStoreClient()

user_product_engage_f_cols = [col for col in spark.read.table(user_product_engage_metrics).columns if col not in lookup_key]
user_product_time_f_cols = [col for col in spark.read.table(user_product_time_metrics).columns if col not in lookup_key]

# COMMAND ----------

df = spark.read.table('sachin_clickstream.electronics_cs_gold_user').drop_duplicates(['user_id','product_id']).select(['user_id','product_id','end_time','is_purchase'])

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md ## Create the training set by joining the features

# COMMAND ----------

feature_lookups = [
  FeatureLookup(
    table_name = user_product_engage_metrics,
    feature_names = user_product_engage_f_cols,
    lookup_key = lookup_key,
    ),
  FeatureLookup(
    table_name = user_product_time_metrics,
    feature_names = user_product_time_f_cols,
    lookup_key = lookup_key
    )]

training_set = fs.create_training_set(df,
                           feature_lookups,
                           label = target_col,
                           exclude_columns = ['end_time'])

training = training_set.load_df()
training  = training.join(df.drop_duplicates(['user_id','product_id']).select(['user_id','product_id','end_time']),how = 'inner' , on =lookup_key)
training = training.toPandas()

# COMMAND ----------

training.columns

# COMMAND ----------

# MAGIC %md ## Create preprocessing pipeline by creating dummies and imputing missing values

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

cols = ['min_price', 'max_price', 'num_of_cart', 'num_of_clicks', 'numSessions',
       'num_of_views', 'session_start_day', 'session_start_month',
       'session_start_dayofweek', 'session_end_day', 'session_end_month',
       'session_end_time_dayofweek', 'duration']

select_pipeline =  ColumnSelector(cols)


one_hot_pipeline = Pipeline(steps=[
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore",drop='first'))
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, [ "session_start_month", "session_start_dayofweek", "session_end_month", "session_end_time_dayofweek"])]


preprocessor = ColumnTransformer(categorical_one_hot_transformers, remainder="passthrough")

# COMMAND ----------

# MAGIC %md ## Create train and test split based on the time window

# COMMAND ----------

from sklearn.model_selection import train_test_split

training['end_time'] = pd.to_datetime(training['end_time']/1000,unit='s')
training.drop(lookup_key,axis=1,inplace = True)

train = training[training['end_time'] <= '2021-01-7'].drop('end_time',axis=1)
test = training[training['end_time'] > '2021-01-7'].drop('end_time',axis=1)

train_Y = train[target_col]
train_X = train.drop(target_col,axis=1)
print("train sample :",train_Y.shape[0])

test_Y = test[target_col]
test_X = test.drop(target_col,axis=1)

# COMMAND ----------

print(train['session_start_month'].unique())
print(test['session_start_month'].unique())

# COMMAND ----------

# MAGIC %md ## Create pipeline and define the workflow

# COMMAND ----------

import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from mlflow.models.signature import infer_signature

set_config(display="diagram")

sklr_classifier = LogisticRegression(
  C=1.6645936949595288,
  penalty="l2",
  random_state=177326489,
)


model = Pipeline([
#       ("colselector",select_pipeline),
    ("preprocessor", preprocessor),
    ("classifier", sklr_classifier),
])

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(disable=False, silent=True)

with mlflow.start_run(run_name="logistic_regression") as mlflow_run:

#     rus = RandomUnderSampler(sampling_strategy = 0.10,random_state=0)
#     X_resampled, y_resampled = rus.fit_resample(train_X, train_Y)
#     print(sorted(Counter(y_resampled).items()))
    model.fit(train_X, train_Y)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    signature = infer_signature(train_X, model.predict(train_X))
    mlflow.log_param('columns',train_X.columns)
#     display(pd.DataFrame([sklr_test_metrics], index=["test"]))
    model_env = mlflow.sklearn.get_default_conda_env()
    fs.log_model(model = model,
                 artifact_path = "feature_model",
                  flavor=mlflow.sklearn,
                  training_set=training_set,
                  **{
#                     "signature" : signature,
                     "conda_env" :model_env,
                     "input_example" : train_X[:5]}
                )
    mlflow.evaluate("runs:/" + mlflow_run.info.run_id+"/model",
                  test,
                  targets =target_col,
                  model_type ="classifier")

# COMMAND ----------

# MAGIC %md ## Transition Model to production

# COMMAND ----------

from mlflow.tracking import MlflowClient

new_model_version = mlflow.register_model(f"runs:/{mlflow_run.info.run_id}/feature_model", model_registry_name)

client = MlflowClient()
latest_version_info = client.get_latest_versions(model_registry_name, stages=["Production"])
if latest_version_info == []:
  print("No model with production Tag found")
else:
  latest_production_version = latest_version_info[0].version

  transition_old = client.transition_model_version_stage(
    name=model_registry_name,
    version=latest_production_version,
    stage="Archived")

  
# Transition new model to production  
transition_new = client.transition_model_version_stage(
  name=model_registry_name,
  version=new_model_version.version,
  stage="Production")

# COMMAND ----------


