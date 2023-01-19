# Databricks notebook source
# MAGIC %md
# MAGIC ## Read data from the message bus system and write into Bronze Delta table
# MAGIC In this step, let's read the Clickstream data and store in a delta table

# COMMAND ----------

# MAGIC %run ./config_cs

# COMMAND ----------

# MAGIC %md
# MAGIC Refer below link for the best practices
# MAGIC https://databricks.atlassian.net/wiki/spaces/~779106034/pages/2615510716/Streaming+to+from+Event+Hubs+-+Performance+Tuning

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

# COMMAND ----------

from pyspark.sql.functions import from_json
from pyspark.sql.functions import col
df_raw_cs=spark.readStream \
          .format("kafka") \
          .option("subscribe", EH_KAFKA_TOPIC) \
          .option("kafka.bootstrap.servers", EH_BOOTSTRAP_SERVERS) \
          .option("kafka.sasl.mechanism", "PLAIN") \
          .option("kafka.security.protocol", "SASL_SSL") \
          .option("kafka.sasl.jaas.config", EH_SASL) \
          .option("startingOffsets","latest") \
          .load() 

# COMMAND ----------

write_delta= df_raw_cs.writeStream \
             .format("delta") \
             .outputMode("append") \
             .option("checkpointLocation", "/tmp/delta/click/_checkpoints/electronics_cs_bronze/") \
             .toTable("sachin_clickstream.electronics_cs_bronze")

# COMMAND ----------

dbutils.notebook.exit("End of notebook execution")

# COMMAND ----------

# MAGIC %sql
# MAGIC truncate table sachin_clickstream.electronics_cs_bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from sachin_clickstream.electronics_cs_bronze

# COMMAND ----------

dbutils.fs.rm("/tmp/delta/click/_checkpoints/electronics_cs_bronze/", True)

# COMMAND ----------


