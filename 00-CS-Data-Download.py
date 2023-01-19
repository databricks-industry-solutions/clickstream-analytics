# Databricks notebook source
# MAGIC %md
# MAGIC # Download Retail Clickstream data from Kaggle

# COMMAND ----------

# MAGIC %pip install kaggle

# COMMAND ----------

download_path = "dbfs:/Users/sachin.patil@databricks.com/datasets/kaggle/cs"
dbutils.fs.mkdirs(download_path)

# COMMAND ----------

# MAGIC %sh
# MAGIC echo "{\"username\":\"schinpatil\",\"key\":\"11cc621b071e2e82363fe31f13a09df6\"}"

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC rm ~/.kaggle/kaggle.json
# MAGIC echo "API Key" > ~/.kaggle/kaggle.json

# COMMAND ----------

# MAGIC %sh
# MAGIC chmod 600 /root/.kaggle/kaggle.json

# COMMAND ----------

# MAGIC %sh
# MAGIC kaggle datasets download -d mkechinov/ecommerce-events-history-in-electronics-store  -p /dbfs/Users/sachin.patil@databricks.com/datasets/kaggle/cs

# COMMAND ----------

dbutils.fs.ls(download_path)

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip /dbfs/Users/sachin.patil@databricks.com/datasets/kaggle/cs/ecommerce-events-history-in-electronics-store.zip

# COMMAND ----------

unzip_path="dbfs:/Users/sachin.patil@databricks.com/datasets/kaggle/cs/unzipped"
dbutils.fs.mkdirs(unzip_path)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -l

# COMMAND ----------

# MAGIC %sh
# MAGIC mv events.csv /dbfs/Users/sachin.patil@databricks.com/datasets/kaggle/cs/unzipped/

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -l /dbfs/Users/sachin.patil@databricks.com/datasets/kaggle/cs/unzipped

# COMMAND ----------

# MAGIC %sh
# MAGIC head /dbfs/Users/sachin.patil@databricks.com/datasets/kaggle/cs/unzipped/events.csv

# COMMAND ----------

# MAGIC %sh
# MAGIC wc -l /dbfs/Users/sachin.patil@databricks.com/datasets/kaggle/cs/unzipped/events.csv

# COMMAND ----------


