![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-Azure-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

# Clickstream Analytics

## Intro
In ecommerce platforms, the path to a purchase is not always as straightforward as we might expect.  A customer may click on a product, put it in their cart, and immediately make a purchase, but more typically, especially with more expensive or complex items, the customer will browse, compare, leave the site for a while, return to give the product more consideration, and do all this multiple times before completing the purchase.  Quite often, customers will engage in this process and never complete the pruchase, instead, abandoning the cart altogether, having either made a purchase through another channel or becoming disinterested in the product.

As we observe customers engage the site, we may wish to estimate the liklihood the customer will purchase the particular item with which they are engaging. A number of factors related to the product, the customer, prior interactions between the customer and the particular product, and patterns of engagement observed in the context of the user's shopping session all contribute to our understanding of the customer's intent.

If we see a customer has a low probability of completing the pruchase of a product, we may use this as a signal that we should suggest alternative products which may be of more interest to that customer.  If we see the customer has a high probability of completing a purchase, we may take the opportunity to suggest complementary products that will boost the size of the sale.  And somewhere in the middle, we might use discounts or promotional offers to nudge the customer forward, securing the sale.

In this solution accelerator, we leverage event data from an electronics ecommerce site to demonstrate how such data may be used to estimate these propensities.  We first use a historical backlog of these data to train a model.  We then implement a real-time stream to calculate features supporting inference with each page clicked.  You can think of the accelerator as divided into these two parts where in the first we focus on understanding the data and how we might approach the Data Science portion of our work.  In the second, the focus is on the mechanics of building a real-time inference layer.  Before diving into those parts (identify by the number associated with different notebooks), be sure to run notebook *0b* to stage the data required for both parts.

**NOTE** This accelerator uses the newly released Databricks Model Serving capability which is avaiable in only select regions.  Please consult the feature documentation to determine if your workspace is in a supported region ([Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/#--region-availability) before deploying the accelerator.

### Understanding Part 1: Model Training

Part 1 of this accelerator is comprised of two notebooks named *1a* and *1b*.  In the first of these notebooks, we generate point-in-time features from historical clickstream data that we've accumulated.  In setting up the data, we pretend as if that data has been accumulated through a workflow organized around a medallion architecture.  

The medallion architecture, explained in more depth in notebook *1a*, divides our data processing workflow into three high-level stages.  
</p>

<img src='https://brysmiwasb.blob.core.windows.net/demos/images/cs_part_1.png' width=750>

In the first stage, known as the Bronze layer, we accumulate data in the raw, unaltered format with which it is presented to Databricks.  We have invisioned a production architecture where data is sent from our website to an Azure Event Hub ingest layer and its from the Event Hub layer that Databricks reads the data.  Because the Bronze data would reflect the default data structure associated with the Event Hub and not necessarily the structure of the data sent to it, we skip this step in Part 1.  However, we do write data to a table representing our Silver layer in this part of the accelerator.

The Silver layer, the second layer of the medallion architecture, represents the data transformed for technical accessiblity but with no business-level interpretation attached.  This is often the starting point for Data Scientists looking for high-fidelity data and for Data Engineers looking to build business-aligned assets that will comprise the Gold layer of our architecture.

For our Gold layer, we derive 5 sets of metrics from our historical event data.  These data are landed in tables named for the feature set they are intended to represent and are assigned a *\__training* suffix to indicate these are generated to support model training and should not be used for live inference. The reason we need to separate these records from our inference records is that for training we recreate our understanding of each user, product and shopping cart's state with each individual event that occurs.  This represents the full body of knowledge we might wish to use during model training whereas during inference we only care about the latest known state for these.  

For those curious how these tables would be managed during retraining events, they would need to be recreated from the Silver layer data available at that time. Once we have implemented the workflows showin in Part 2, we would have a real-time process populating our Bronze and Silver layer tables so that the one-time staging of historical data in the Silver tables performed in Part 1 of the accelerator would not be repeated.

### Understanding Part 2: Real-Time Inference

Part 2 of this accelerator is comprised to four notebooks named *2a*, *2b*, *2c* and *2d*. The first three tackle the data engineering workflow that moves data through the medallion architecture.  The fourth notebook addresses the deployment of our trained model for real-time inference.
</p>
<img src='https://brysmiwasb.blob.core.windows.net/demos/images/cs_part_2.png' width=1000>

In order to simulate data flowing in real-time from a website, we replay the last month of the 5-month dataset (the first four of which were used in the Part 1 notebooks).  Notebook *2a* provides the replay logic and writes data to an Azure Event Hub you will need to provision in advance of running the notebook.  The data is played back in accelerated real-time so that you can replay the period in less than one month's time.

Notebooks *2b* and *2c* represent the streaming and batch portions of our feature engineering workflow.  As we will explain in notebook *1a*, it doesn't make much sense for us to compute absolutely all features in real-time.  By separating our logic this way, we hope to provide a suggested compromise for similar workflows that will help you reduce complexity and the cost of processing your data. In both these notebooks, we publish data to Gold layer tables (named using a *\__inference* suffix) which we then register with our feature store.  We employ both offline feature store and online feature store capabilities.  The online feature store capability facilitates real-time inference but does require you to deploy an Azure CosmosDB instance and configure a database within it before you run these notebooks

Notebook *2d* addresses model deployment.  In this notebook, we need to revisit the model trained in Part 1 and associate it with the feature store tables created in Part 2.  This is an atypical deployment pattern but one necessitated by the fact that we use a different set of feature tables during model training than the ones intended for model inference.

___
Sachin Patil (sachin.patil@databricks.com), Puneet Jain (puneet.jain@databricks.com) & Bryan Smith (bryan.smith@databricks.com)


___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| kaggle                                | kaggle API      | Apache 2.0        | https://github.com/Kaggle/kaggle-api                      |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
