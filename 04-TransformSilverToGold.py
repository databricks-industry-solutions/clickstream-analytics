# Databricks notebook source
from pyspark.sql.functions import expr
from pyspark.sql.functions import window

csSilverDF = spark \
  .readStream \
  .table("sachin_clickstream.electronics_cs_silver")

csGoldDF = csSilverDF \
.withWatermark("event_time", "10 minutes") \
.groupBy(window("event_time", "10 minutes"), csSilverDF.product_id, csSilverDF.user_id, csSilverDF.user_session) \
.agg(expr("max(event_time) as start_time"), expr("min(event_time) as end_time"), \
     expr("count(product_id) as num_of_clicks"),expr("max(price) as max_price"), expr("min(price) as min_price"), \
     expr("sum(case when event_type='view' then 1 else	0 end) as num_of_views"), \
     expr("sum(case when event_type='cart' then 1 else	0 end) as num_of_cart"), \
     expr("max(case when event_type='view' then 0 when	event_type='cart' then 0 else	1 end) as is_purchase") \
    ) 

csGoldDF2=csGoldDF.selectExpr("product_id", "user_id", "user_session", "start_time", "end_time", "num_of_clicks", "max_price", "min_price", "num_of_views", "num_of_cart","is_purchase") 

# csGoldDF2.writeStream \
#     .format("delta") \
#     .outputMode("append") \
#     .option("checkpointLocation", "/tmp/delta/click/_checkpoints/electronics_cs_session_gold/") \
#     .toTable("sachin_clickstream.electronics_cs_session_gold")


# foreachbatch
# optimizewrite 32 mb
# flatmapwithstate

# COMMAND ----------

from pyspark.sql.functions import expr
from pyspark.sql.functions import window

csSilverDF = spark \
  .readStream \
  .table("sachin_clickstream.electronics_cs_silver")

csGoldDF2 = csSilverDF \
.groupBy(csSilverDF.product_id, csSilverDF.user_id, csSilverDF.user_session) \
.agg(expr("max(event_time) as start_time"), expr("min(event_time) as end_time"), \
     expr("count(product_id) as num_of_clicks"),expr("max(price) as max_price"), expr("min(price) as min_price"), \
     expr("sum(case when event_type='view' then 1 else	0 end) as num_of_views"), \
     expr("sum(case when event_type='cart' then 1 else	0 end) as num_of_cart"), \
     expr("max(case when event_type='view' then 0 when	event_type='cart' then 0 else	1 end) as is_purchase") \
    ) 

# csGoldDF2=csGoldDF.selectExpr("product_id", "user_id", "user_session", "start_time", "end_time", "num_of_clicks", "max_price", "min_price", "num_of_views", "num_of_cart","is_purchase") 


# COMMAND ----------

from pyspark import Row
  
# Function to upsert `microBatchOutputDF` into Delta table using MERGE
def upsertToDelta(microBatchOutputDF, batchId): 
  # Set the dataframe to view name
  microBatchOutputDF.createOrReplaceTempView("updates")

  # ==============================
  # Supported in DBR 5.5 and above
  # ==============================

  # Use the view name to apply MERGE
  # NOTE: You have to use the SparkSession that has been used to define the `updates` dataframe
  microBatchOutputDF._jdf.sparkSession().sql("""
    MERGE INTO sachin_clickstream.electronics_cs_session_gold t
    USING updates s
    ON s.product_id = t.product_id and s.user_id=t.user_id and s.user_session=t.user_session
    WHEN MATCHED THEN UPDATE SET t.num_of_clicks=s.num_of_clicks+t.num_of_clicks
    WHEN NOT MATCHED THEN INSERT *
  """)

# Setting # partitions to 1 only to make this demo faster.
# Not recommended for actual workloads.
spark.conf.set("spark.sql.shuffle.partitions", "1")

# Reset the output aggregates table
# spark.createDataFrame([ Row(key=0, count=0) ]).write \
#   .format("delta").mode("overwrite").saveAsTable("sachin_clickstream.electronics_cs_session_gold")

# Define the aggregation
# aggregatesDF = spark.readStream \
#   .format("rate") \
#   .option("rowsPerSecond", "1000") \
#   .load() \
#   .selectExpr("value % 100 as key") \
#   .groupBy("key") \
#   .count()

# Start the query to continuously upsert into aggregates tables in update mode
csGoldDF2.writeStream \
  .format("delta") \
  .foreachBatch(upsertToDelta) \
  .outputMode("update") \
  .start() 

# COMMAND ----------

dbutils.notebook.exit("End of notebook execution")

# COMMAND ----------

display(csGoldDF)

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table sachin_clickstream.electronics_cs_gold

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_gold

# COMMAND ----------

# MAGIC %sql
# MAGIC create table hive_metastore.sachin_clickstream.clicks_features_electornics
# MAGIC as
# MAGIC select product_id, user_id, 
# MAGIC count(1) as num_of_clicks, 
# MAGIC count(distinct user_session) as num_of_sessions, 
# MAGIC max(price) as max_price,
# MAGIC min(price) as min_price,
# MAGIC sum(case when event_type='view' then 1
# MAGIC else	0 end) as num_of_views,
# MAGIC sum(case when event_type='cart' then 1
# MAGIC else	0 end) as num_of_cart,
# MAGIC max(case when event_type='view' then 0
# MAGIC when	event_type='cart' then 0 
# MAGIC else	1 end) as is_purchase, 
# MAGIC max(to_timestamp(event_time)) as last_time,
# MAGIC min(to_timestamp(event_time)) as start_time,
# MAGIC timestampdiff(SECOND,min(to_timestamp(event_time)), max(to_timestamp(event_time))) as duration,
# MAGIC extract(year from max(to_timestamp(event_time))) as year,
# MAGIC extract (month from max(to_timestamp(event_time))) as month,
# MAGIC extract (DAYOFWEEK from max(to_timestamp(event_time))) as day_of_week,
# MAGIC extract(HOUR from max(to_timestamp(event_time))) as hour,
# MAGIC case when extract (DAYOFWEEK from max(to_timestamp(event_time))) = 1 or extract (DAYOFWEEK from max(to_timestamp(event_time)))=7 
# MAGIC then 1 else 0 end as weekend_flag
# MAGIC from hive_metastore.sachin_clickstream.electronics_cs_bz_1
# MAGIC group by product_id, user_id

# COMMAND ----------

sensorStreamDF = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "host1:port1,host2:port2") \
  .option("subscribe", "tempAndPressureReadings") \
  .load()

sensorStreamDF = sensorStreamDF \
.withWatermark("eventTimestamp", "10 minutes") \
.groupBy(window(sensorStreamDF.eventTimestamp, "10 minutes")) \
.avg(sensorStreamDF.temperature,
     sensorStreamDF.pressure)

sensorStreamDF.writeStream
  .format("delta")
  .outputMode("append")
  .option("checkpointLocation", "/delta/events/_checkpoints/temp_pressure_job/")
  .start("/delta/temperatureAndPressureAverages")

# COMMAND ----------

# MAGIC %sql
# MAGIC truncate table sachin_clickstream.electronics_cs_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC describe table sachin_clickstream.electronics_cs_gold13

# COMMAND ----------

# MAGIC %sql
# MAGIC describe detail sachin_clickstream.electronics_cs_gold13

# COMMAND ----------

dbutils.fs.rm("/tmp/delta/click/_checkpoints/csGold/")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from sachin_clickstream.electronics_cs_gold

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_gold

# COMMAND ----------

# MAGIC %sql
# MAGIC select product_id, user_id, min(window.start), max(window.end), sum(num_of_clicks), max(max_price), min(min_price), sum(num_of_views), sum(num_of_views), sum(num_of_cart), max(is_purchase)  
# MAGIC from sachin_clickstream.electronics_cs_gold
# MAGIC group by product_id, user_id
# MAGIC limit 5

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table sachin_clickstream.electronics_cs_session_gold

# COMMAND ----------

dbutils.fs.rm("/tmp/delta/click/_checkpoints/electronics_cs_session_gold/", True)

# COMMAND ----------

dbutils.fs.ls("/tmp/delta/click/_checkpoints/electronics_cs_session_gold/*")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_session_gold
# MAGIC -- where user_id=1515915625521929824

# COMMAND ----------

# MAGIC %sql
# MAGIC truncate table sachin_clickstream.electronics_cs_session_gold

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_silver_backup
# MAGIC where user_id=1515915625521929824

# COMMAND ----------

# MAGIC %sql
# MAGIC create table sachin_clickstream.electronics_cs_silver_backup
# MAGIC as 
# MAGIC select * from sachin_clickstream.electronics_cs_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC truncate table sachin_clickstream.electronics_cs_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_silver_backup
# MAGIC where user_id=1515915625521929824

# COMMAND ----------

# MAGIC %sql
# MAGIC insert into sachin_clickstream.electronics_cs_silver
# MAGIC select * from sachin_clickstream.electronics_cs_silver_backup
# MAGIC where user_id=1515915625521929824

# COMMAND ----------

# MAGIC %sql
# MAGIC select product_id, user_id, 
# MAGIC count(1) as num_of_clicks, 
# MAGIC count(distinct user_session) as num_of_sessions, 
# MAGIC max(price) as max_price,
# MAGIC min(price) as min_price,
# MAGIC sum(case when event_type='view' then 1
# MAGIC else	0 end) as num_of_views,
# MAGIC sum(case when event_type='cart' then 1
# MAGIC else	0 end) as num_of_cart,
# MAGIC max(case when event_type='view' then 0
# MAGIC when	event_type='cart' then 0 
# MAGIC else	1 end) as is_purchase, 
# MAGIC max(to_timestamp(event_time)) as last_time,
# MAGIC min(to_timestamp(event_time)) as start_time,
# MAGIC timestampdiff(SECOND,min(to_timestamp(event_time)), max(to_timestamp(event_time))) as duration,
# MAGIC extract(year from max(to_timestamp(event_time))) as year,
# MAGIC extract (month from max(to_timestamp(event_time))) as month,
# MAGIC extract (DAYOFWEEK from max(to_timestamp(event_time))) as day_of_week,
# MAGIC extract(HOUR from max(to_timestamp(event_time))) as hour,
# MAGIC case when extract (DAYOFWEEK from max(to_timestamp(event_time))) = 1 or extract (DAYOFWEEK from max(to_timestamp(event_time)))=7 
# MAGIC then 1 else 0 end as weekend_flag
# MAGIC from hive_metastore.sachin_clickstream.electronics_cs_bz_1
# MAGIC group by product_id, user_id
