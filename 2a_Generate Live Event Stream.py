# Databricks notebook source
# MAGIC %md The purpose of this notebook is to generate the live stream of event data for use in the Clickstream Propensity solution accelerator.  You may find this notebook at https://github.com/databricks-industry-solutions/clickstream-analytics

# COMMAND ----------

# MAGIC %md ##Important Note
# MAGIC 
# MAGIC Please note that while a Databricks ML cluster was used in prior notebooks, this and all subsequent notebooks will make use of a **standard Databricks cluster with Photon Acceleration enabled**.  The [Photon engine](https://www.databricks.com/product/photon) supports high-performance Data Engineering scenarios such as the one explored in this and subsequent notebooks.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC In notebook *0b* we partitioned our data into *historical* and *real-time* subsets.  The historical subset represented event data already processed that would be used to train a model that we would apply to "future" records streaming into the environment in "real-time".  The *real-time* subset, representing the last 1 month of the approximately 5 month overall dataset, is the portion of the dataset we will focus on in this notebook.
# MAGIC 
# MAGIC To simulate our *real-time* data arriving from a web server, we will stream it to a data ingest layer.  We'll use Azure Event Hubs because they are easy to setup and expose themselves to downstream systems using a standard Kafka API.  So while the transmission of events data to the Event Hub will take on some service-specific elements, the downstream portions of our work found in subsequent notebooks should be broadly recognizable to those familiar with Apache Kafka.
# MAGIC 
# MAGIC To setup the Azure Event Hub, you'll need to follow [these steps](https://learn.microsoft.com/en-us/azure/event-hubs/event-hubs-create). Be sure to configure a Shared Access Policy with Send and Listen policies on your event hub and record the connection string for that policy in notebook *0a* before running this and subsequent notebooks. Be sure to set the pricing tier of your event hub to Standard or above for use with this demo as the Kafka interface is [not supported](https://learn.microsoft.com/en-us/azure/event-hubs/event-hubs-quickstart-kafka-enabled-event-hubs) in the lower level Basic tier.
# MAGIC 
# MAGIC **NOTE** This notebook should be running as you run any of the remaining notebooks in this accelerator.

# COMMAND ----------

# DBTITLE 1,Get Config Info
# MAGIC %run "./0a_Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

import time

# COMMAND ----------

# MAGIC %md ##Step 1: Setup Ingest Layer
# MAGIC 
# MAGIC To setup your own Azure Event Hub, please follow [these steps](https://learn.microsoft.com/en-us/azure/event-hubs/event-hubs-create). Once you have completed those steps, create a *SAS Policy* with *Send* and *Listen* permissions for your eventhub and copy its connection string associated with the secondary key to the appropriate spot in the *0a* notebook.  Logic in there will break the connection string into the parts needed to connect to the Kafka API endpoint.

# COMMAND ----------

# MAGIC %md ##Step 2: Access Events Data
# MAGIC 
# MAGIC We will now retrieve our *real-time* events data separated from the *historical* records in notebook *0b*:

# COMMAND ----------

# DBTITLE 1,Retrieve Real-Time Events Data
realtime_events = (
  spark
    .table('staged_events')
    .filter(fn.expr("data_set='real-time'"))
    .drop('data_set')
  )

# cache data to minimize retrieval overhead in the loop that follows
realtime_events = realtime_events.repartition('event_time').cache()

display(realtime_events.orderBy('event_time'))

# COMMAND ----------

# MAGIC %md ##Step 3: Send Events Data to Ingest Layer
# MAGIC 
# MAGIC We will now playback the real-time events data, looping through each unique timestamp in the dataset and sending the records associated with that timestamp to the ingest layer. The time between events will be used to delay transmittal of the next set of records, though a *speed_factor* variable can be used to accelerate the playback.  With one month's worth of data, a speed factor of 10 will allow our playback logic to transmit all data in about 3 days.  An extended loop is needed to allow you to observe real-time inference in the notebooks that follow:
# MAGIC 
# MAGIC **NOTE** Our playback logic does not account for the overhead associated with retrieving and transmitting specific records from our dataset.  As a result, the playback will not be perfectly aligned with the *speed_factor* variable, especially as it is elevated to higher values: 

# COMMAND ----------

# DBTITLE 1,Set Playback Speed
speed_factor=100

# COMMAND ----------

# DBTITLE 1,Get Unique Timestamps
# get unique timestamps
timestamps = (
   realtime_events
      .select('event_time')
      .distinct()
      .orderBy('event_time')
      .toPandas()['event_time']
      .to_list()
    )

print( f'First Event: {timestamps[0]} | Last Event: {timestamps[-1]}' )

# COMMAND ----------

# MAGIC %md **NOTE** We are adding a timestamp column called *ingest_ts* to our events payload.  This value reflects the current date and time as recognized by Databricks.  As data flows through our streaming workflows, we will extract this value and calculate the current timestamp at different stages to understand the speed with which data moves through our workflow. 

# COMMAND ----------

# DBTITLE 1,Push Data to Event Hub
# initialize last ts
last_ts = timestamps[0]

# for each set of events
for i, ts in enumerate(timestamps):
  
  # notify of date change
  if (last_ts.date() != ts.date()) or (i==0):
    print(f"Current Date: {ts.date().strftime('%Y-%m-%d')}", end='\r')
  
  # push events to event hub
  start_time = time.process_time()
  ts_events = (
    realtime_events
      .filter(fn.expr(f"event_time = '{ts}'")) # get relevant events
      .withColumn('ingest_ts', fn.expr("CURRENT_TIMESTAMP()")) # insert our local timestamp into payload for perf monitoring
      .selectExpr('to_json(struct(*)) as value') # convert each event to json document
      .write # write to kafka endpoint
        .format('kafka')
        .option('kafka.bootstrap.servers', config['eh_bootstrap_servers'])
        .option('kafka.sasl.mechanism', 'PLAIN')
        .option('kafka.security.protocol', 'SASL_SSL')
        .option('kafka.sasl.jaas.config', config['eh_sasl'])
        .option('topic', config['eh_kafka_topic'])
        .save()
    )
  end_time = time.process_time()
  duration = end_time - start_time
  
  # sleep required seconds
  sleep_seconds = max(((ts-last_ts).seconds / speed_factor**1.25) - duration, 0)
  time.sleep( sleep_seconds )
  
  # set last ts
  last_ts = ts

# COMMAND ----------

# MAGIC %md With the events flowing into the ingest layer, you can now proceed to the next notebook.  (Please be sure to leave this notebook running as you move forward.)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
