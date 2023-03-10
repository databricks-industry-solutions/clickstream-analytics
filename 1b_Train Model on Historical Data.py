# Databricks notebook source
# MAGIC %md The purpose of this notebook is train the model for the Clickstream Propensity solution accelerator.  This notebook was developed on a **Databricks ML 12.1** cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC With our metrics in place, we now can use those values to train a model.  Our model will predict the probability that a product engaged as part of a shopping session will be purchased within that session.  Such propensity calculations are typically handled as binary classification problems but ones where there's often a class imbalance created by the fact that most products interacted with are not ultimately purchased.  To overcome this issue, we will need to explore a variety of hyperparameter settings that will force our model to home in on those limited events associated with a purchase. That exercise, referred to as hyperparameter tuning, will require us to explore a large number of model variations.  We'll take advantage of the scalability of Databricks to speed up this process.

# COMMAND ----------

# DBTITLE 1,Get Config Info
# MAGIC %run "./0a_Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import mlflow
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef

from xgboost import XGBClassifier

import pyspark.sql.functions as fn

# COMMAND ----------

# MAGIC %md ##Step 1: Retrieve Features & Labels
# MAGIC 
# MAGIC Our first step is to assemble a set of features and labels. The metrics we calculated in our last notebook will serve as our features.  For the labels, we will flag each event record with a 0 or 1 indicating whether the product associated with that event eventually led to a purchase in the current shopping session. The first step in this process is to identify each product engaged in each shopping session and flag it based on whether it was purchased or not:

# COMMAND ----------

# DBTITLE 1,Identify Product Outcome in Shopping Sesssions
# identify whether a given product in a shopping session was purchased (1) or not (0)
cart_product_outcomes = (
  spark
    .table('electronics_events_silver')
    .groupBy('user_id','user_session','product_id')
      .agg(
        fn.max(fn.expr("CASE WHEN event_type='purchase' THEN 1 ELSE 0 END")).alias('purchased')
        )
  )

display(cart_product_outcomes.orderBy('user_id','product_id'))

# COMMAND ----------

# MAGIC %md Now we will assemble the feature records, one for each event in our historical dataset.  We will link each of these events to our various metrics, as those metrics were understood at the time of the event:

# COMMAND ----------

# DBTITLE 1,Assemble Features for Each Event Instance
# identify event instances to which we need to assign feature metrics
raw_features = (
  spark
    .table('electronics_events_silver')
    .select('user_id','user_session','product_id','event_time','event_type') # include event type for validation/troubleshooting purposes
    .withColumn('batch_time', fn.expr("DATE_TRUNC('day', event_time)")) # calculate batch time as would be available at this event time
  )

# identify metrics tables and how they will join with event data
metrics_map = { # table_name, key fields
  'electronics_cart_metrics__training':['user_id','user_session','event_time'],
  'electronics_cart_product_metrics__training':['user_id','user_session','product_id','event_time'],
  'electronics_user_metrics__training':['user_id','batch_time'],
  'electronics_product_metrics__training':['product_id','batch_time'],
  'electronics_user_product_metrics__training':['user_id','product_id','batch_time']
  }

# attach metrics to each event record
for table_name, key_fields in metrics_map.items():
  
  # retrieve metrics 
  temp_features = spark.table(table_name)
  
  # identify how fields from metric table should be named in feature set
  prefix = table_name.split('__')[0] # prefix is everything prior to the __
  renamed_fields = [f"{c} as {prefix}__{c}" for c in temp_features.columns if c not in key_fields]
  
  # rename metric fields in preparation for join
  temp_features = temp_features.selectExpr(key_fields + renamed_fields)

  # join metrics to event instances
  raw_features = (
    raw_features
      .join(
        temp_features,
        on=key_fields,
        how='left'
        )
    )

# display feature data
display(raw_features)

# COMMAND ----------

# MAGIC %md Now we will flag each event record for the eventual outcome of the associated product within the shopping session (cart), assigning a label to each feature instance:

# COMMAND ----------

# DBTITLE 1,Assemble Features & Labels
raw_features_and_labels = (
  raw_features
    .join(
      cart_product_outcomes,
      on=['user_id','user_session','product_id']
      )
  )

display(raw_features_and_labels.orderBy('user_id','user_session','event_time'))

# COMMAND ----------

# MAGIC %md In our raw features and labels dataset, we've kept a number of fields in place that are helpful for validation of the data. Having had a chance to review our data and now confident in the feature set, we can remove those fields.  We can also remove those records for the actual purchase events; just doesn't make sense to predict a purchase when a purchase occurs:
# MAGIC 
# MAGIC **NOTE** We are keeping *user_id* and *user_session* so that we may split our data at a shopping cart level in the next step. Those fields will be removed once we have performed the split.

# COMMAND ----------

# DBTITLE 1,Finalize Features & Labels Set
features_and_labels = (
  raw_features_and_labels
    .filter("event_type != 'purchases'") # remove purchase events
    .drop('event_type','product_id','event_time','batch_time') # drop unnecessary fields
  )

# COMMAND ----------

# MAGIC %md ##Step 2: Define & Tune Model
# MAGIC 
# MAGIC With our features and labels assembled, need to consider how best to split our data.  The typical pattern is to treat each event instance as independent of the others and perform a random split on that set.  If we did that, separate events from the same shopping cart might land in both our training and testing sets creating the potential for contamination where knowledge about a session in training influences the prediction we make during testing.  To avoid this, we may split on the individual shopping carts and then train on the events associated with the training carts and test on the events associated with the testing carts:

# COMMAND ----------

# DBTITLE 1,Get Unique Carts
carts = (
  features_and_labels
    .select('user_id','user_session')
    .distinct()
  )

# COMMAND ----------

# DBTITLE 1,Split on Carts
holdout_ratio = 0.1
train_test_ratio = 1 - holdout_ratio

# split carts
train_test, validate = carts.randomSplit([train_test_ratio, holdout_ratio])
train, test = train_test.randomSplit([1-(holdout_ratio/train_test_ratio), holdout_ratio/train_test_ratio])

# present split set counts
print(f"Training:\t{train.count()}")
print(f"Testing:\t{test.count()}")
print(f"Validate:\t{validate.count()}")

# COMMAND ----------

# DBTITLE 1,Associate Events with Split Carts
train_test = train_test.join(features_and_labels, on=['user_id','user_session']).drop('user_id','user_session')
train = train.join(features_and_labels, on=['user_id','user_session']).drop('user_id','user_session')
test = test.join(features_and_labels, on=['user_id','user_session']).drop('user_id','user_session')
validate = validate.join(features_and_labels, on=['user_id','user_session']).drop('user_id','user_session')

# present split set counts
print(f"Training:\t{train.count()}")
print(f"Testing:\t{test.count()}")
print(f"Validate:\t{validate.count()}")

# COMMAND ----------

# MAGIC %md With our data split, we're just about ready to perform hyperparameter tuning.  One of the key things we need to consider in doing such an exercise when our approach involves a binary classification is whether or not our data includes a class imbalance:

# COMMAND ----------

# DBTITLE 1,Examine Ratio of Positive and Negative Classes at Cart Level
display(
  features_and_labels
    .select('user_id','user_session','purchased') # get distinct sessions and session labels
    .distinct()
    .withColumn('total',fn.expr("COUNT(*) OVER()"))
    .groupby('purchased')
      .agg( 
        fn.count('*').alias('instances'),
        fn.first('total').alias('total')
        )
    .withColumn('ratio',fn.expr("instances/total"))
    .drop('total')
  )

# COMMAND ----------

# MAGIC %md We can see from these results that only about 5% of carts result in a purchase.  It would be easy for us to state that no carts result in a purchase and we've have a model that's 95% accurate but clearly not useful.  So, we'll want to consider weighing the positive class instances a bit higher during our tuning exercise in order to force our model to give more consideration to positive class predictions.
# MAGIC 
# MAGIC With this in mind, we can turn our attention to our model.  We'll make use of the [XGBoostClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier) given its frequent use in these kind of exercises.
# MAGIC 
# MAGIC One of the challenges in using the XGBoostClassifier is the large number of hyperparameters with which model training is configured.  We'll define a search space for a subset of these parameters and then use [Hyperopt](http://hyperopt.github.io/hyperopt/) to intelligently explore the values in the identified ranges to see which combination of hyperparameter values give us our best model results. More details about defining Hyperopt search spaces can be found [here](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/):

# COMMAND ----------

# DBTITLE 1,Define Hyperparameter Search Space
search_space = {
    'max_depth' : hp.quniform('max_depth', 1, 20, 1)                       
    ,'learning_rate' : hp.uniform('learning_rate', 0.01, 0.40) 
    ,'scale_pos_weight' : hp.uniform('scale_pos_weight', 1.0, 100)   
    }

# COMMAND ----------

# MAGIC %md To search this space, we need to define a function.  Hyperopt will pass a set of candidate hyperparameter values to this function which will return an evaluation metric with which the Hyperopt engine will evaluate parameter performance. That metric is returned as a loss value which Hyperopt seeks to minimize.  We will be using the average precision metric for our evaluation metric, which places greater emphasis on positive class predictions.  This metric *improves* as its value increases.  For that reason, we'll return a negative average precision score from the function so that Hyperopt drives the hyperparameters in the right direction:
# MAGIC 
# MAGIC **NOTE** We've included some additional metrics in our evaluation as these are frequently used in evaluating binary classifiers with class imbalances.

# COMMAND ----------

# DBTITLE 1,Define Function for Evaluating Hyperparameter values
def evaluate_model(hyperopt_params):
  
  # accesss replicated input data
  train_input = train_pd_broadcast.value
  test_input = test_pd_broadcast.value
  
  X_train = train_input.drop('purchased', axis=1)
  y_train = train_input['purchased']
  X_test = test_input.drop('purchased', axis=1)
  y_test = test_input['purchased']
  
  # configure model parameters
  params = hyperopt_params
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
  # all other hyperparameters are taken as given by hyperopt
  
  # instantiate model with parameters
  model = XGBClassifier(**params)
  
  # train
  model.fit(X_train, y_train)
  
  # predict
  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)
  
  # eval metrics
  model_ap = average_precision_score(y_test, y_prob[:,1])
  model_ba = balanced_accuracy_score(y_test, y_pred)
  model_mc = matthews_corrcoef(y_test, y_pred)
  
  # log metrics with mlflow run
  mlflow.log_metrics({
    'avg precision':model_ap,
    'balanced_accuracy':model_ba,
    'matthews corrcoef':model_mc
    })                                       
                                             
  # invert key metric for hyperopt
  loss = -1 * model_ap
  
  # return results
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md Within our function, we are tapping into variables named *train_pd_broadcast* and *test_pd_broadcast*.  These represent replicated copies of our *train* and *test* Spark dataframes converted to pandas dataframes.  (Note that if your data is too large to be moved to a pandas dataframe, you might consider taking a random sample your datasets.) By replicating these across each worker in our cluster, we enable each call to our function, multiple of which are executed in parallel across the Databricks cluster by Hyperopt, to access a local copy of the dataset.  But to create these copies of our data, we first need to broadcast the datasets as follows:

# COMMAND ----------

# DBTITLE 1,Broadcast Training & Testing Data Sets
train_pd_broadcast = sc.broadcast(train.toPandas())
test_pd_broadcast = sc.broadcast(test.toPandas())

# COMMAND ----------

# MAGIC %md With everything in place, we can now run our hyperparameter tuning jobs and identify our optimal parameters as follows.  Please note, we have limited our evaluations to relatively small number but may find even better results by elevating that value.  Using Hyperopt in combination with the scaled-out Databricks cluster, we are in control of how much time we want to spend seeking optimal hyperparameter values:

# COMMAND ----------

# DBTITLE 1,Explore Search Space
# perform evaluation
with mlflow.start_run(run_name='tuning'):
  
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
    max_evals=50,
    trials=SparkTrials(parallelism=5),  # 50 evals with 5 evals at a time = 10 learning cycles
    verbose=True
    )
  
# separate hyperopt output from our results
print('\n')

# capture optimized hyperparameters
hyperopt_params = space_eval(search_space, argmin)
hyperopt_params

# COMMAND ----------

# MAGIC %md ##Step 4: Train & Persist Final Model
# MAGIC 
# MAGIC With our optimal hyperparameter values identified, we can train our final model using all our training and testing data as the inputs and the validation set which was withheld from the tuning exercise for a final model evaluation:

# COMMAND ----------

# DBTITLE 1,Train Final Model & Validate
with mlflow.start_run(run_name='trained'):
  
  # get data train & validate final model with
  train_test_pd = train_test.toPandas()
  validate_pd = validate.toPandas()

  # separate X & y
  X_train_test = train_test_pd.drop('purchased', axis=1)
  y_train_test = train_test_pd['purchased']
  X_validate = validate_pd.drop('purchased', axis=1)
  y_validate = validate_pd['purchased']

  # configure model parameters
  params = hyperopt_params
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
    
  # train
  model = XGBClassifier(**params)
  model.fit(X_train_test, y_train_test)
  
  # predict
  y_pred = model.predict(X_train_test)
  y_prob = model.predict_proba(X_train_test)
  
  # eval metrics
  model_ap = average_precision_score(y_train_test, y_prob[:,1])
  model_ba = balanced_accuracy_score(y_train_test, y_pred)
  model_mc = matthews_corrcoef(y_train_test, y_pred)
  
  # record metrics with mlflow run 
  mlflow.log_metrics({
    'avg precision':model_ap,
    'balanced_accuracy':model_ba,
    'matthews corrcoef':model_mc
    })  
       
  # persist wrapped model
  mlflow.sklearn.log_model(
    artifact_path='model',
    sk_model=model,
    pip_requirements=['xgboost'],
    registered_model_name=config['model name'],
    pyfunc_predict_fn='predict_proba'
    )
  
  print(model_ap)

# COMMAND ----------

# MAGIC %md Please note two things about how we are logging our model.  First, we log it under the *sklearn* API.  The XGBoosClassifier model supports an sklearn API so that while there are both *xgboost* and *pyfunc* model flavors available to us in mlflow, we are able to log this model as an sklearn variant.  
# MAGIC 
# MAGIC Why do we do that?  Well, that's the second thing we need to note.  If you look at the *log_model* method call, you will see there is an *pyfunc_predict_fn* argument that's only supported with the *sklearn* mlflow flavor.  By setting this argument to *predict_proba* instead of allowing it to default to *predict* we are telling mlflow to use this alternative method to generate predictions when we use mlflow to deploy our model to a user-defined function or under an REST API.  This function returns the underlying probabilities associated with our class predictions which are far more useful in this scenario than simple 0s and 1s.
# MAGIC 
# MAGIC Our model is now registered with the [MLFlow registry](https://mlflow.org/docs/latest/model-registry.html).  The registry supports integration with various CI/CD processes, but to facilitate our demonstration, we'll simply move our model to production status programmatically:

# COMMAND ----------

# DBTITLE 1,Move Model to Production Status
# connect to mlflow
client = mlflow.tracking.MlflowClient()

# identify model version in registry
model_version = client.search_model_versions(f"name='{config['model name']}'")[0].version

# move model version to production
client.transition_model_version_stage(
  name=config['model name'],
  version=model_version,
  stage='production'
  )      

# COMMAND ----------

# MAGIC %md ##Step 4: Examine Predictions
# MAGIC 
# MAGIC Before moving on, it would be helpful to examine our model's outputs to better understand the predictions they return.  To do this, we'll retrieve our model and deploy it as a udf:

# COMMAND ----------

# DBTITLE 1,Deploy Model as UDF
predict_udf = mlflow.sklearn.pyfunc.spark_udf(
  spark, 
  f"models:/{config['model name']}/production"
  )

# COMMAND ----------

# MAGIC %md Before using our UDF to generate predictions, its important to understand what it will return.  The *predict_proba* method we configured it to use returns a 2-dimensional array.  The first value in the array represents the probability of class 0 (not purchased) and the second value represents the probability of class 1 (purchased).  However, the UDF only returns the first of these values which means we need to subtract the returned value from 1.0 in order to get the prediction we actually want:  

# COMMAND ----------

# DBTITLE 1,Examine UDF Return Type
predict_udf.returnType

# COMMAND ----------

# MAGIC %md With that understood, we can now generate purchase predictions as follows:

# COMMAND ----------

# DBTITLE 1,Generate Predictions for a Sample Session
# get predictions for sample shopping session
sample = (
  raw_features_and_labels
    .filter(fn.expr("user_id=1515915625353900095 AND user_session='12dadbda-acb5-4f26-a5e8-f71814190c04'"))
    .withColumn('prediction', 1 - predict_udf(fn.struct([c for c in train.columns if c not in ['purchased']]))) # identify which columns to pass to the function for prediction
    .select('user_id','user_session','event_time','event_type','product_id','purchased','prediction')
    .orderBy('event_time')
  )

display(sample)

# COMMAND ----------

# MAGIC %md While every shopping experience is different, there are some interesting patterns here that are likely applicable to a large number of shopping sessions.  First, when our user first engages our products, we have a small bump in our belief they will complete the transaction.  But as the session continues on over an extended period of time, we settle into a general, low-level expectation they will make the purchase.  As we get to the end of this session, we see that we start picking up signal slowly boosts our expectation they will buy a product until we get right up to the end when our prediction moves close to certain.  
# MAGIC 
# MAGIC In this shopping session, there are some views that take place after the actual purchase which are flagged as certain of the purchase.  As was mentioned in the earlier notebook, this messiness in the data is not perfectly understood so these predictions are interesting but not too telling from a business execution standpoint.

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
