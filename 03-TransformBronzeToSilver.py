# Databricks notebook source
# MAGIC %md
# MAGIC Data in the eventbus is maintained in the JSON format. As part of the bronze to silver data transformation, data is first parsed and persisted in the silver delta table

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
csBronzeDF = spark \
  .readStream \
  .table("sachin_clickstream.electronics_cs_bronze") \
  .select(from_json(col("value").cast("string"), schema).alias("parsed_cs")) \
          .select("parsed_cs.*")

# COMMAND ----------

csBzDF2=csBronzeDF.selectExpr("product_id", "to_timestamp(event_time) as event_time", "event_type", "category_id", "category_code", "brand","price", "user_id", "user_session") 

# COMMAND ----------

write_delta= csBzDF2.writeStream \
             .format("delta") \
             .outputMode("append") \
             .option("checkpointLocation", "/tmp/delta/click/_checkpoints/electronics_cs_silver/") \
             .toTable("sachin_clickstream.electronics_cs_silver")

# COMMAND ----------

dbutils.notebook.exit("End of notebook execution")

# COMMAND ----------

# MAGIC %sql
# MAGIC delete from sachin_clickstream.electronics_cs_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC -- truncate table sachin_clickstream.electronics_cs_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from sachin_clickstream.electronics_cs_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from sachin_clickstream.electronics_cs_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from sachin_clickstream.electronics_cs_bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1)  from sachin_clickstream.electronics_cs_gold

# COMMAND ----------

# MAGIC %sql
# MAGIC truncate table sachin_clickstream.electronics_cs_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_bronze

# COMMAND ----------

dbutils.fs.rm("/tmp/delta/click/_checkpoints/electronics_cs_silver/", True)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_gold_session

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_gold_user

# COMMAND ----------


