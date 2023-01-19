# Databricks notebook source
import time
import pandas as pd

from pyspark.sql.functions import col

from pyspark.sql.types import (
    StringType,
    IntegerType,
    StructType,
    LongType,
    TimestampType,
    BooleanType,
    StructField,
    DoubleType,
)

# Set timeout such that the user journey will be expired if no data received for xx mins (parameter in milliseconds)
user_journey_timeout=14*60*1000

def sessionization_func(key, pdf_iter, state):
  import pandas as pd
  import calendar
  
  product_id_s=key[0]
  user_id_s=key[1]
  
  if state.hasTimedOut:
    prev_state = state.get
    numSessions_s = prev_state[0]
    start_time_s = prev_state[1]
    end_time_s = prev_state[2]
    num_of_clicks_s = prev_state[3]
    num_of_views_s = prev_state[4]
    num_of_cart_s = prev_state[5]
    max_price_s = prev_state[6]
    min_price_s = prev_state[7]
    is_purchase_s = prev_state[8]
    
    finalUpdate = { 'product_id': product_id_s, 'user_id': user_id_s, 'numSessions': numSessions_s, 'start_time': start_time_s, 'end_time': end_time_s, 'num_of_clicks': num_of_clicks_s, 'num_of_views': num_of_views_s, 'num_of_cart': num_of_cart_s, 'max_price': max_price_s, 'min_price': min_price_s, 'is_purchase': is_purchase_s, 'expired': True }
    state.remove()
    yield pd.DataFrame.from_dict([finalUpdate])
  else:
    # Set up proper default values
    numSessions_s = 0
    start_time_s = 2 ** 63 - 1
    end_time_s = -2 ** 63
    num_of_clicks_s = 0
    num_of_views_s = 0
    num_of_cart_s = 0
    max_price_s = 0
    min_price_s = 0
    is_purchase_s = 0

    prev_state = state.getOption
    if prev_state:
      numSessions_s = prev_state[0]
      start_time_s = prev_state[1]
      end_time_s = prev_state[2]
      num_of_clicks_s = prev_state[3]
      num_of_views_s = prev_state[4]
      num_of_cart_s = prev_state[5]
      max_price_s = prev_state[6]
      min_price_s = prev_state[7]
      is_purchase_s = prev_state[8]
    
    for pdf in pdf_iter:
      # Update start and end timestamps in session
      numSessions_s += len(pdf)
      
      # Python produces epoch as second precision in integer part and sub-second precision in decimal part,
      # hence have to take care of it - ideal approach is to bring up millisecond precision to integer part.
      #
      # If someone doesn't want to struggle with the semantic of timestamp and its timezone (it could be headache
      # if you are going back and forth with localized timestamp vs UTC based timestamp), or if someone
      # is not convenient with dealing with the model and class of Python time, you can convert the column of
      # timestamp type to long type before passing to Python UDF. The value will be microsecond precision.
      pser = pdf['start_time'].apply(lambda dt: calendar.timegm(dt.utctimetuple()) * 1000 + int(dt.microsecond / 1000))
      start_time_s = min(min(pser), start_time_s)
      psmr = pdf['end_time'].apply(lambda dt: calendar.timegm(dt.utctimetuple()) * 1000 + int(dt.microsecond / 1000))
      end_time_s = max(max(psmr), end_time_s)
      
      num_of_clicks_s += sum(pdf['num_of_clicks'])
      num_of_views_s += sum(pdf['num_of_views'])
      num_of_cart_s += sum(pdf['num_of_cart'])
      max_price_s = max(max(pdf['max_price']), max_price_s)
      
      if (min_price_s == 0 or min_price_s is None):
        min_price_s=min(pdf['min_price'])
      else:
        min_price_s = min(min(pdf['min_price']), min_price_s)
                        
      is_purchase_s = max(max(pdf['is_purchase']), is_purchase_s)
    
    state.update((numSessions_s, start_time_s, end_time_s, num_of_clicks_s, num_of_views_s, num_of_cart_s, max_price_s, min_price_s, is_purchase_s, ))

#     # Set timeout such that the user journey will be expired if no data received for specific time
    state.setTimeoutDuration(user_journey_timeout)
   
    session_update = { 'product_id': product_id_s, 'user_id': user_id_s, 'numSessions': numSessions_s, 'start_time': start_time_s, 'end_time': end_time_s, 'num_of_clicks': num_of_clicks_s, 'num_of_views': num_of_views_s, 'num_of_cart': num_of_cart_s, 'max_price': max_price_s, 'min_price': min_price_s, 'is_purchase': is_purchase_s, 'expired': False }
    print(session_update)
    print(type(session_update))
    yield pd.DataFrame.from_dict([session_update])


df_session=spark.readStream.table("sachin_clickstream.electronics_cs_gold_session")


output_type = StructType(
  [
    StructField("product_id", IntegerType()),
    StructField("user_id", StringType()),
    StructField("numSessions", LongType()),
    StructField("start_time", LongType()),
    StructField("end_time", LongType()),
    StructField("num_of_clicks", LongType()),
    StructField("num_of_views", LongType()),
    StructField("num_of_cart", LongType()),
    StructField("max_price", DoubleType()),
    StructField("min_price", DoubleType()),
    StructField("is_purchase", IntegerType()),
    StructField("expired", BooleanType()),
  ]
)
state_type = StructType(
  [
    StructField("numSessions", LongType()),
    StructField("start_time", LongType()),
    StructField("end_time", LongType()),
    StructField("num_of_clicks", LongType()),
    StructField("num_of_views", LongType()),
    StructField("num_of_cart", LongType()),
    StructField("max_price", DoubleType()),
    StructField("min_price", DoubleType()),
    StructField("is_purchase", IntegerType()),
  ]
)


# Using Append mode since Delta sink does not support Update mode and we don't want to focus on how to upsert the result from the example.
#   df_session_temp.selectExpr("product_user", "count(session_id) as numSessions", "sum(num_of_clicks) as num_of_clicks", "min(start_time) as start_time", "max(end_time) as end_time")
#   .groupBy("product_user")
q = (
  df_session.selectExpr("product_id", "user_id", "session_id", "start_time", "end_time", "num_of_clicks", "max_price", "min_price", "num_of_views", "num_of_cart", "is_purchase")
  .groupBy("product_id", "user_id")
  .applyInPandasWithState(
    sessionization_func, output_type, state_type, "Append", "ProcessingTimeTimeout"
  )
  .writeStream
  .option("checkpointLocation", "/tmp/delta/click/_checkpoints/electronics_cs_gold_user/")
  .outputMode("append")
  .trigger(processingTime='1 second')
  .toTable("sachin_clickstream.electronics_cs_gold_user")
)

q.awaitTermination()

# COMMAND ----------

dbutils.notebook.exit("End of notebook execution")

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table sachin_clickstream.electronics_cs_gold_user

# COMMAND ----------

dbutils.fs.rm('/tmp/delta/click/_checkpoints/electronics_cs_gold_user/', True) 

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
# MAGIC select product_id, user_id, min(window.start) as start_time, max(window.end) as last_time,, sum(num_of_clicks), max(max_price), min(min_price), sum(num_of_views), sum(num_of_views), sum(num_of_cart), max(is_purchase), timestampdiff(SECOND,min(window.start), max(window.end)) as duration,  extract(year from max(window.end))
# MAGIC from sachin_clickstream.electronics_cs_gold
# MAGIC group by product_id, user_id
# MAGIC limit 5

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_gold_session

# COMMAND ----------

# MAGIC %sql
# MAGIC create table sachin_clickstream.electronics_cs_gold_session_2
# MAGIC as
# MAGIC select * from sachin_clickstream.electronics_cs_gold_session

# COMMAND ----------

# MAGIC %sql
# MAGIC truncate table sachin_clickstream.electronics_cs_gold_session

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sachin_clickstream.electronics_cs_gold_session_2
# MAGIC where product_id=3830671

# COMMAND ----------


