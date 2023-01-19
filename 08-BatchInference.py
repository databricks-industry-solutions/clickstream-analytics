# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC ##Deploying the model for batch inferences
# MAGIC 
# MAGIC <img style="float: right; margin-left: 20px" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_batch_inference.gif" />
# MAGIC 
# MAGIC Now that our model is available in the Registry, we can load it to compute our inferences and save them in a table to start building dashboards.
# MAGIC 
# MAGIC We will use MLFlow function to load a pyspark UDF and distribute our inference in the entire cluster. If the data is small, we can also load the model with plain python and use a pandas Dataframe.
# MAGIC 
# MAGIC If you don't know how to start, Databricks can generate a batch inference notebook in just one click from the model registry !

# COMMAND ----------

import mlflow
from databricks.feature_store import FeatureStoreClient
from collections import Counter
import pandas as pd
import numpy as np

# COMMAND ----------

fs = FeatureStoreClient()

model_registry_name = 'propensity_model_puneet'

# COMMAND ----------

df = data.drop_duplicates(['user_id','product_id']).select(['user_id','product_id'])

# COMMAND ----------

display(df)

# COMMAND ----------

prediction = fs.score_batch(model_uri= 'models:/'+ model_registry_name +"/7",df=df)


# COMMAND ----------

display(prediction.select('prediction').distinct().count())

# COMMAND ----------

data = fs.read_table('sachin_clickstream.user_product_engage_metrics')
display(data.count())

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##Deploying the model for real-time inferences
# MAGIC 
# MAGIC <img style="float: right; margin-left: 20px" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_realtime_inference.gif" />
# MAGIC 
# MAGIC Our marketing team also needs to run inferences in real-time using REST api (send a customer ID , Product ID  and get back the inference).
# MAGIC 
# MAGIC Real-time with Model Serving v2, you can deploy your Databricks Model in a single click.
# MAGIC 
# MAGIC Open the Model page and click on "Serving". It'll start your model behind a REST endpoint and you can start sending your HTTP requests!

# COMMAND ----------

! pip freeze

# COMMAND ----------


