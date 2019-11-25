# hops-yarn-ML

This project collects useful metrics from InfluxDB coming from Resource Manager, Node Manager, Spark through Graphite and Telegraf and also from MySQL cluster and then processes the data to feed into Tensorflow to build a fully connected Feedforward Neural Network for predicting memory and CPU utilization for applications at the container level.

The data processing, cleansing and aggregation is done using pyspark and machine learning model is built on Tensorflow. The data is read in batches from the databases and offset is used to keep track of the subsequent batches till it reads all the data.

The credentials information of databases if read from a congif.txt file


Input parameters:

Start timestamp: time1 <br />
End timestamp:        time2 <br />
Time is in seconds but we need to provide in the following format e.g 1501758105000000000

How to use: <br />
/srv/hops/spark-2.1.0-bin-without-hadoop/bin/spark-submit yarn_machine_learning.py

Note: <br />
At the moment it is designed to predict one label that is 'PCpuUsagePercentAvgPercents', average cpu utilization in percentage. 
