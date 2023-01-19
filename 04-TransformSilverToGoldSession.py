# Databricks notebook source
# MAGIC %md
# MAGIC For any retail websites, the user session is determined by the pre-defined inactivity period. In the given dataset, the session length varies widely and the maximum session length is 155 days which is not very reliable. Hence we are creating session definition using the custom logic based on the predefined inactivity period. Based on your requirement, you can change the configuration for inactivity period.
# MAGIC 
# MAGIC In session windows are natively supported in Spark 3.2 and above. A session window closes when there's no input received within the gap duration after receiving the latest input. This enables you to group events until there are no new events for a specified time duration (inactivity).
# MAGIC 
# MAGIC https://www.databricks.com/blog/2021/10/12/native-support-of-session-window-in-spark-structured-streaming.html

# COMMAND ----------

#Let's define the inactivity period here. You can change the definition for your website requirement

inactivity_period="4 minutes"

# COMMAND ----------

from pyspark.sql.functions import expr
from pyspark.sql.functions import window, session_window

csSilverDF = spark \
  .readStream \
  .table("sachin_clickstream.electronics_cs_silver")

csGoldDFSession = csSilverDF \
.withWatermark("event_time", "10 minutes") \
.groupBy(session_window("event_time", inactivity_period), csSilverDF.product_id, csSilverDF.user_id) \
.agg(expr("max(event_time) as end_time"), expr("min(event_time) as start_time"), \
     expr("count(product_id) as num_of_clicks"),expr("max(price) as max_price"), expr("min(price) as min_price"), \
     expr("sum(case when event_type='view' then 1 else	0 end) as num_of_views"), \
     expr("sum(case when event_type='cart' then 1 else	0 end) as num_of_cart"), \
     expr("max(case when event_type='view' then 0 when	event_type='cart' then 0 else	1 end) as is_purchase") \
    ) 

csGoldDFSessKey=csGoldDFSession.selectExpr("product_id", "user_id", "concat('Session-',cast(end_time as STRING)) as session_id", "start_time", "end_time", "num_of_clicks", "max_price", "min_price", "num_of_views", "num_of_cart","is_purchase") 

csGoldDFSessKey.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/delta/click/_checkpoints/electronics_cs_gold_session/") \
    .toTable("sachin_clickstream.electronics_cs_gold_session")


# COMMAND ----------

dbutils.notebook.exit("End of notebook execution")

# COMMAND ----------

# MAGIC %sql
# MAGIC delete from sachin_clickstream.electronics_cs_gold_session

# COMMAND ----------

dbutils.fs.rm('/tmp/delta/click/_checkpoints/electronics_cs_gold_session/', True) 

# COMMAND ----------

display(csGoldDFSessKey)

# COMMAND ----------

display(csGoldDFSession)

# COMMAND ----------

display(csSilverDF)

# COMMAND ----------

# MAGIC %sql
# MAGIC select concat("Session-",cast(event_time as STRING)) from sachin_clickstream.electronics_cs_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_silver

# COMMAND ----------

display(csGoldDF)

# COMMAND ----------

# MAGIC %sql
# MAGIC truncate table sachin_clickstream.electronics_cs_gold_session

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table sachin_clickstream.electronics_cs_gold_session

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_gold

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
# MAGIC select * from sachin_clickstream.electronics_cs_gold_session

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table sachin_clickstream.electronics_cs_session_gold

# COMMAND ----------

dbutils.fs.rm("/tmp/delta/click/_checkpoints/", True)

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
# MAGIC select * from sachin_clickstream.electronics_cs_gold_session

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from sachin_clickstream.electronics_cs_gold_session

# COMMAND ----------

# MAGIC %sql
# MAGIC truncate table sachin_clickstream.electronics_cs_gold_session

# COMMAND ----------

dbutils.fs.rm("/tmp/delta/click/_checkpoints/", True)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC truncate table sachin_clickstream.electronics_cs_silver
