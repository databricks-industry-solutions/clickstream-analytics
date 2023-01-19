# Databricks notebook source
# MAGIC %md
# MAGIC # Write data into Azure EventHubs using Kafka API 
# MAGIC 
# MAGIC In the notebook, we are simulating data ingestion into a message bus. In this example, I am sourcing data from file, however in real world scenario, it can be directly from the Clickstream API in near real time. We can use any message bus system. In this example, I am writing into Azure Eventhubs. However I am using Kafka API of the Eventhubs as it is more performant than native Eventhubs API. As I am using Kafka API, the same code can be leveraged for Apache Kafka, Confluent Kafka, Amazon MSK etc.

# COMMAND ----------

# MAGIC %run ./config_cs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest the clicks streams data from object stores into Message bus systems
# MAGIC Source data is in the CSV format. Ingestion includes below steps:
# MAGIC - Define the CSV schema
# MAGIC - Load data in a dataframe using readstream
# MAGIC - To write the data in message bus systems, typically key and value are passed. In this example, all the attributes are converted into a JSON structure and deinfed as value. Key is optional and not passed to the message bus system in this example.

# COMMAND ----------

# Define the frequency at which you would like to ingest data into eventstream
# In this case, we are ingesting one day's data in single batch and deinfing frequency of each batch in seconds
rate=120

# COMMAND ----------

from pyspark.sql.types import StructType, IntegerType, DateType, StringType, DoubleType

schema = (
  StructType()
  .add("event_time", StringType(), True)
  .add("event_type", StringType(), True)
  .add("product_id", IntegerType(), True)
  .add("category_id", StringType(), True)
  .add("category_code", StringType(), True)
  .add("brand", StringType(), True)
  .add("price", DoubleType(), True)
  .add("user_id", StringType(), True)
  .add("user_session", StringType(), True)
)

df_clickStream = (
  spark.read.format("csv")
  .option("delimiter", ",")
  .option("header", "true")
  .schema(schema)
  .load("dbfs:/Users/sachin.patil@databricks.com/datasets/kaggle/cs/unzipped/")
)

# COMMAND ----------

from pyspark.sql.functions import *
cs_dates= (
            df_clickStream
            .select(date_format(to_date(to_timestamp('event_time')),"yyyy-MM-dd").alias("dt_col"))
            .distinct()
            .orderBy( to_date(to_timestamp('event_time')))  # sorting of list is essential for logic below
          ).collect()

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

import time
for cs_dt in cs_dates:
  dt=cs_dt['dt_col']
  temp_cs_df= (df_clickStream \
                .filter(to_date(to_timestamp(df_clickStream.event_time))==dt) \
                .selectExpr("to_json(struct(*)) AS value") \
                .write \
                .format("kafka") \
                .option("kafka.bootstrap.servers", EH_BOOTSTRAP_SERVERS) \
                .option("kafka.sasl.mechanism", "PLAIN") \
                .option("kafka.security.protocol", "SASL_SSL") \
                .option("kafka.sasl.jaas.config", EH_SASL) \
                .option("topic", EH_KAFKA_TOPIC) \
                .save())
  time.sleep(rate)


# COMMAND ----------

dbutils.notebook.exit("End of notebook execution")

# COMMAND ----------


