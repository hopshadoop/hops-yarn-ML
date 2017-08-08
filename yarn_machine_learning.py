from pyspark import SparkConf, SparkContext
from influxdb import InfluxDBClient
from copy import deepcopy
import tensorflow as tf
import os
import re
from operator import add
import numpy as np
import argparse
import mysql.connector
import csv
import random
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class InfluxTensorflow():

    def __init__(self, hostname, port, username, password, db, mysql_host, mysql_user, mysql_db, mysql_port, t1, t2):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.db = db
        self.len_features = 5

        self.query_rns = 'select * from'

        self.query_t_cpu = 'select * from cpu'
        self.query_t_mem = 'select * from mem'

        self.mysql_host = mysql_host
        self.mysql_user = mysql_user
        self.mysql_db = mysql_db
        self.mysql_port = mysql_port

        self.t1 = t1
        self.t2 = t2

        self.limit = 10000

        self.rep_None = -1
        self.col_len_spark = 8

        self.hostname_lookup = {'vagrant':1}

    def query_batch(self, query, db, epoch='s'):
        """
        get results of a query from database

        :param query:
        :param db:
        :param epoch:
        :return
        """
        cli = InfluxDBClient(self.hostname, self.port, self.username, self.password)

        while True:
            res = cli.query(query, database=db, epoch=epoch)
            if not res:
                break
            #yield res
            return res

            offset += limit
            break

    def training_step(self, i, update_test_data, update_train_data, X, Y_, Y, data_train, data_test, train_step, sess,
                      col_length, batch_size, labels, labels_test, cross_entropy, accuracy):
        """
        traininig the machine learning model on specific iterations

        :param i: iteration count
        :param update_test_data: contains updated testing data
        :param update_train_data: contains updated training data
        :param XX: data
        :param Y_: one hot encoding
        :param Y1: Model to train
        :param data_train: data used for training
        :param train_step: step size during training
        :param sess: session
        :param col_length: num of features
        :param batch_size: rows per batch
        :param labels:
        :param cross_entropy: cost function
        :return: cost of training and testing lists
        """

        print "\r", i,

        ####### evaluating model performance for printing purposes
        train_c = []
        test_c = []
        train_a = []
        test_a = []

        # feed values include Python scalars, strings, lists, or numpy ndarray
        # the backpropagation training step
        sess.run(train_step, feed_dict={X: data_train, Y_: labels})

        if update_train_data:
            a, c = sess.run([accuracy,cross_entropy], feed_dict={X: data_train, Y_: labels})
            train_a.append(a)
            train_c.append(c)

        if update_test_data and len(data_test) > 0:
            a, c = sess.run([accuracy,cross_entropy], feed_dict={X: data_train, Y_: labels_test})
            test_a.append(a)
            test_c.append(c)

        return (train_c, test_c, train_a, test_a)

    def initialize_session(self):
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        return sess

    def train_model(self, data, data_test, labels, labels_test):
        """
        train the machine learning Model

        :param: rdd_join: Resilient distributed datasets
        :return: result for ML model
        """

        total_data_size = len(data) # total rows of data
        batch_size = 10
        epoch_size = 5000;

        training_iter = total_data_size / batch_size

    #     batch_size = total_data_size / training_iter
        col_length = len(data[0])

        data_train = data
        data_train = np.array(data, dtype=np.float32)
    #     data_train /= np.std(data_train, axis=0)
        data_test = np.array(data_test, dtype=np.float32)
    #     data_test /= np.std(data_test, axis=0)
    #     print (data_train.shape)
    #     labels = np.array(labels, dtype=np.float32)
    #     print (labels.shape);

        # 1. Define Variables and Placeholders
        X = tf.placeholder(tf.float32, [batch_size, col_length], name='X') #the first dimension (None) will index the images
        Y_ = tf.placeholder(tf.float32, [batch_size,], name='Y_') # placeholder for correct answers

        X = tf.nn.batch_normalization(
        X,
        50, #mean
        0.8, #variance
        5, #offset
        3, #scale
        0.005, #variance_epsilon
        )

        # Weights initialised with small random values between -0.2 and +0.2
        W1 = tf.Variable(tf.truncated_normal([col_length, 6], stddev=0.09))
        B1 = tf.Variable(tf.zeros([1]))
        W2 = tf.Variable(tf.truncated_normal([6, 3], stddev=0.08))
        B2 = tf.Variable(tf.zeros([1]))
        W3 = tf.Variable(tf.truncated_normal([3, 1], stddev=0.2))
        B3 = tf.Variable(tf.ones([1]))

        W4 = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
        B4 = tf.Variable(tf.zeros([1]))
        W5 = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
        B5 = tf.Variable(tf.zeros([1]))
        W6 = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
        B6 = tf.Variable(tf.zeros([1]))
        W7 = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
        B7 = tf.Variable(tf.zeros([1]))
        W8 = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
        B8 = tf.Variable(tf.zeros([1]))
        W9 = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
        B9 = tf.Variable(tf.zeros([1]))
        W10 = tf.Variable(tf.truncated_normal([1, 1], stddev=0.1))
        B10 = tf.Variable(tf.zeros([1]))

        # 2. Define the model

        ######## SIGMOID activation func #######
    #     Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
    #     Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
    #     Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
    #     Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
    #     Y5 = tf.nn.sigmoid(tf.matmul(Y4, W5) + B5)

        ######## ReLU activation func #######
        Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    #     Y1 = tf.nn.l2_normalize(Y1, 0, epsilon=1e-12, name=None)
        Y1 = tf.nn.dropout(Y1, 0.5, noise_shape=None, seed=None,name='dropoutY1')
        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
        Y2 = tf.nn.dropout(Y2, 0.5, noise_shape=None, seed=None,name='dropoutY2')
        Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
        Y3 = tf.nn.dropout(Y3, 0.5, noise_shape=None, seed=None,name='dropoutY3')
        Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
    #     Y4 = tf.nn.dropout(Y4, 0.3, noise_shape=None, seed=None,name='dropoutY4')
        Y5 = tf.nn.relu(tf.matmul(Y4, W5) + B5)
    #     Y5 = tf.nn.dropout(Y5, 1, noise_shape=None, seed=None,name='dropoutY5')
        Y6 = tf.nn.relu(tf.matmul(Y5, W6) + B6)
    #     Y6 = tf.nn.dropout(Y6, 0.5, noise_shape=None, seed=None,name='dropoutY6')
        Y7 = tf.nn.relu(tf.matmul(Y6, W7) + B7)
    #     Y7 = tf.nn.dropout(Y7, 0.5, noise_shape=None, seed=None,name='dropoutY7')
        Y8 = tf.nn.relu(tf.matmul(Y7, W8) + B8)
    #     Y8 = tf.nn.dropout(Y8, 0.5, noise_shape=None, seed=None,name='dropoutY8')
        Y9 = tf.nn.relu(tf.matmul(Y8, W9) + B9)
    #     Y9 = tf.nn.dropout(Y9, 0.5, noise_shape=None, seed=None,name='dropoutY9')
        Y10 = tf.nn.relu(tf.matmul(Y9, W10) + B10)
    #     Y10 = tf.nn.dropout(Y10, 0.5, noise_shape=None, seed=None,name='dropoutY10')

        Y = Y3

        cross_entropy = tf.reduce_sum(tf.pow(Y - Y_, 2))/(2*batch_size) # reduce_mean
    #     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_) )

        # Loss function using L2 Regularization
    #     regularizer = tf.nn.l2_loss(W1); beta = 0.2
    #     cross_entropy = tf.reduce_mean(cross_entropy + beta * regularizer)

        is_correct = tf.equal(tf.argmax(Y,0), tf.argmax(Y_,0))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        # 5. Define an optimizer
    #     optimizer = tf.train.GradientDescentOptimizer(0.5)
        optimizer = tf.train.AdamOptimizer(0.003)  ## do not use gradient descent 0.005
        train_step = optimizer.minimize(cross_entropy)

        # initialize and train
        sess = self.initialize_session()
        # 6. Train and test the model, store the accuracy and loss per iteration

        train_c = []
        test_c = []
        train_a = []
        test_a = []

        for k in range(50):
            for i in range(training_iter):
                test = False
                if i % epoch_size == 0:
                    test = True
                c, tc, a, ta = self.training_step(k*training_iter+i, test, test, X, Y_, Y, data_train[i*batch_size:batch_size*(i+1)],
                    data_test[i*batch_size:batch_size*(i+1)], train_step, sess, col_length, batch_size,
                    labels[i*batch_size:batch_size*(i+1)],labels_test[i*batch_size:batch_size*(i+1)],
                                      cross_entropy, accuracy)
                train_c += c
                test_c += tc
                train_a += a
                test_a += ta
        print ('Train Cost',train_c)
        print ('Test Cost', test_c)
        return (train_c, test_c, train_a, test_a, training_iter, epoch_size)

    def train_model_test(self, rdd_join):
        """
        only for test purpose
        """
        data = rdd_join.collect()
        batch_size = len(data) # total rows of data
        col_length = len(data[0])

        training_data = np.array(data)

        #if n_features != self.n_input_features_:
        #    raise ValueError("X shape does not match training shape")

        x = tf.placeholder(tf.float32, shape=(batch_size, col_length))

        y = tf.matmul(tf.reshape(x, [batch_size, col_length]), x)

        data_initializer = tf.placeholder(dtype=tf.float32,
                                shape=[batch_size, col_length])
        input_data = tf.Variable(data_initializer, trainable=False, collections=[])
        with tf.Session() as sess:
            print (sess.run(input_data.initializer, feed_dict={x: training_data}))

        """# Specify that all features have real-value data
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=5)]

        # Build 3 layer DNN with 10, 20, 10 units respectively.
        classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3)
        # Fit model.
        classifier.fit(x=training_data, y=training_data, steps=10)

        # Evaluate accuracy.
        accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
        print ('Accuracy: {0:f}'.format(accuracy_score))"""

        return []

    def train_model_lstm(self, data):
        """
        training the LSTM ML Model

        :param data: input data with features
        :return: result of LSTM ML model
        """
        num_steps = 1
        data = data.collect()
        batch_size = len(data) # total rows of data
        col_length = len(data[0])
        #data = np.array(data)
        print ("data",data)

        lstm_size = col_length
        # Placeholder for the inputs in a given iteration.
        #words = tf.placeholder(tf.float32, [batch_size, num_steps])

        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        print ('lstm_size',lstm.state_size)
        # Initial state of the LSTM memory.
        initial_state = state = tf.zeros([col_length, lstm.state_size]) #lstm.state_size])
        probablilities = []
        loss = 0.0

        #for i in range(num_steps):
        if True:
            # The value of state is updated after processing each batch of words.
            output, state = lstm(data, state)

            # The LSTM output can be used to make next word predictions
            logits = tf.matmul(output, softmax_w) + softmax_b
            probabilities.append(tf.nn.softmax(logits))
            loss += loss_function(probabilities, target_words)

        final_state = state
        return final_state

    def load_data_into_tensorflow(self, data, data_test, labels, labels_test):
        return self.train_model(data, data_test, labels, labels_test)
        # return self.train_model_lstm(data)

    def get_appid_from_cont(self,cont_id):
        return "application_" + re.search('container_.*?_(.*?_.*?)_',cont_id).group(1)

    def join_rdd(self, rdd1_, rdd2_):
        """
        Join two spark RDD

        :param rdd1_: first RDD
        :param rdd2_: second RDD
        :return: result after joining two RDD
        """
        """if rdd2_.collect():
            print 'nonnonononon'
            rdd_ = rdd1_.leftOuterJoin(rdd2_)
            print ('join yes no',rdd_.collect())
        else:
            print 'yesyesyes'"""
        rdd_ = rdd1_.join(rdd2_) #.collectAsMap() # .reduceByKey(lambda x,y : x+y)
        if rdd_:
            return rdd_.map(lambda x : (x[0],sum(x[1],()))) # adding multiples tuples
        else:
            return rdd1_

    def join_graphite_metrics(self, results_g, sc, source):
        """
        Convert graphite db results into RDD

        :param results_g: query results from graphite db
        :param sc: sparkcontext
        :return: RDD
        """
        if results_g:
            for count, res in enumerate(results_g.raw['series'][0:]):
                values_g = res['values']
                name_g = res['name']
                x = res['columns'];
                print ('Columns',name_g, x)
                #print ('Columns',name_g, x[0:1] + x[97:98] + x[39:40] + x[44:45] + x[51:52] + x[57:59] +\
                       # x[64:65] + x[100:101] + ['app id'] )
                #print ('values',values_g)

                rdd1 = sc.parallelize(values_g)
                rdd1_ = rdd1 # remove after testing
                if source == 'nm':
                    #print ('Columns',name_g, x[0:1] + x[97:98] + x[39:40] + x[44:45] + x[51:52] + x[57:59] +\
                    # x[64:65] + x[100:101] + ['app id'] )

                    rdd1_ = rdd1.map(lambda x: x[0:1] + x[97:98] + x[39:40] + x[44:45] + x[51:52] + x[57:59] + x[64:65] +\
                    x[100:101] + [ self.get_appid_from_cont(x[100]) ] ) # x[97]:hostname
                    rdd1_ = rdd1_.map(lambda x: [ self.rep_None if a == None else a for a in x])

                    # converted to tuple for join/union operation to work properly
                    rdd1_ = rdd1_.map(lambda x: ((x[0], (x[1]).replace('Hostname=','')), tuple(x[2:])))
                elif source == 'spark':
                    #print ('Columns',name_g, x[0:1] + x[5:6] + x[8:9] + x[17:18] + x[27:28] + x[29:30] + x[52:57] + x[58:63] )

                    rdd1_ = rdd1.map(lambda x: x[0:1] + x[5:6] + x[8:9] + x[17:18] + x[27:28] + x[29:30] + x[52:57] + x[58:63])
                    rdd1_ = rdd1_.map(lambda x: [ self.rep_None if a == None else a for a in x])
                    rdd1_ = rdd1_.map(lambda x: x[0:1] + [float(x[1].replace('application_','').replace('_',''))] + x[2:])
                elif source == 'rm':
                    print ('Columns',name_g, x[0:1] + x[88:89] + x[32:33] + x[42:43] )

                    rdd1_ = rdd1.map(lambda x: x[0:1] + x[88:89] + x[32:33] + x[42:43]) # x[88]:hostname
                    rdd1_ = rdd1_.map(lambda x: [ self.rep_None if a == None else a for a in x])
                    # converted to tuple for join/union operation to work properly
                    rdd1_ = rdd1_.map(lambda x: ((x[0], (x[1]).replace('Hostname=','')), tuple(x[2:])))
                else:
                    pass

                #print ('nm', count, name_g, rdd1_.collect())
                if count == 0:
                    rdd_join = rdd1_
                else:
                    rdd_join = rdd_join.union(rdd1_)
                    #rdd_join = self.join_rdd(rdd_join, rdd1_)
                    pass

            return rdd_join
        else:
            return []

    def join_telegraf_metrics(self, results_t, sc, source):
        """
        Convert telegraf db results into RDD

        :param results_t: query results from telegraf db
        :param sc: sparkcontext
        :return: RDD
        """

        for count, res_t in enumerate(results_t.raw['series'][0:2]):
            """ There are no tags at host """
            values_t = res_t['values']
            name_t = res_t['name']; # print ('name_t',name_t)
            x = res_t['columns']; #print ("columns", name_t, x[0:1] + x[8:9] + x[3:4])
            #print (values_t)

            rdd1 = sc.parallelize(values_t)
            rdd1 = rdd1.map(lambda x: [ self.rep_None if a == None else a for a in x])

            if source == 'cpu': # for host cpu info
                rdd1_ = rdd1.map(lambda x: ((x[0], x[3]), tuple(x[7:9] + x[13:16]))) # x[3]hostname
            elif source == 'mem': # for host mem info
                rdd1_ = rdd1.map(lambda x: ((x[0], x[8]), tuple(x[3:4] ))) # hardcoded time need to be replaced after

            rdd1_ = rdd1_.map(lambda x: [ 0 if a == None else a for a in x])
            #print ("tele rdd1_", count, name_t, rdd1_.collect())
            if count == 0:
                rdd_join = rdd1_
            else:
                rdd_join = self.join_rdd(rdd_join, rdd1_)
        return rdd_join

    def join_mysql_metrics(self, results_mysql, sc):
        rdd1 = sc.parallelize(results_mysql)
        rdd1 = rdd1.map(lambda x: (x[3], tuple(x[0:1] + x[8:11])) ) # 8 am_memory, 9 am_Vcores, 10 execution duration
        return rdd1

    def get_results_from_mysql_cluster(self):
        cnx = mysql.connector.connect(user=self.mysql_user, password=self.mysql_user, host=self.mysql_host,
              database=self.mysql_db, port=self.mysql_port);
        cursor = cnx.cursor()
        res = cursor.execute(("select * from jobs_history"))
        return cursor.fetchall()

    def get_results_from_graphite(self, time1, time2):
        query = "{0} where time > {1} and time < {2} group by /time/".format(self.query_g, time1, time2)
        return self.query_batch(query, db="graphite")

    def get_results_from_telegraf_cpu(self, time1, time2, offset):
        query = "{0} where time > {1} and time < {2} and cpu =~ /cpu-total/ limit {3} offset {4}".\
                 format(self.query_t_cpu, time1, time2, self.limit, offset)
        #query = "{0} where cpu =~ /cpu-total/ limit {1} offset {2}".\
        #         format(self.query_t_cpu, self.limit, offset)
        return self.query_batch(query, db="telegraf")

    def get_results_from_telegraf_mem(self, time1, time2, offset):
        query = "{0} where time > {1} and time < {2} limit {3} offset {4}".format(self.query_t_mem, time1, time2, self.limit, offset)
        #query = "{0} limit {1} offset {2}".format(self.query_t_mem, self.limit, offset)
        return self.query_batch(query, db="telegraf")

    def get_results_from_graphite_nm(self, time1, time2, offset):
        query = "{0} nodemanager where source =~ /container.*$/ and time > {1} and time < {2} limit {3} offset {4}".\
                 format(self.query_rns, time1, time2, self.limit, offset) # group by /time/,/cpu/,/source/
        #query = "{0} nodemanager where source =~ /container.*$/ limit {1} offset {2}".\
        #        format(self.query_rns, self.limit, offset)
        return self.query_batch(query, db="graphite")

    def get_results_from_graphite_rm(self, time1, time2, offset):
        query = "{0} resourcemanager where service =~ /yarn.*$/ and source =~ /ClusterMetrics.*$/ and time > {1} and time < {2} limit {3} offset {4}".\
                 format(self.query_rns, time1, time2, self.limit, offset)
        return self.query_batch(query, db="graphite")

    def get_results_from_graphite_spark(self, time1, time2, offset):
        query = "{0} spark where source =~ /jvm/ and service =~ /driver/ and time > {1} and time < {2} limit {3} offset {4}".\
                 format(self.query_rns, time1, time2, self.limit, offset)
        return self.query_batch(query, db="graphite")

    def remv_app_s(self, string):
        return float(string.replace('application_','').replace('_',''))

    def remv_cont_s(self, string):
        return float(string.replace('ContainerResource_container_e','').replace('_',''))

    def conv_numbers_to_app_id(self, s):
        str = re.search('(.*)([0-9]{4}$)',s)
        return 'application_' + str.group(1) + '_' + str.group(2)

    def get_data_from_influxdb(self):
        time1 = self.t1
        time2 = self.t2

        results_mysql = self.get_results_from_mysql_cluster()

        offset0 = 0 # for telegraf, rm DB
        cc = 0 # count to execute sparkcontext only once
        while (1): # data from telegraf is fetched in batches
            results_t_cpu = self.get_results_from_telegraf_cpu(time1, time2, offset0);
            if results_t_cpu:
                len_cpu = len(results_t_cpu.raw['series'][0]['values']);
                print ('len_cpu',len_cpu);

                results_t_mem = self.get_results_from_telegraf_mem(time1, time2, offset0);
                print ("result_telegraf_cpu", len(results_t_cpu));
                print ("result_telegraf_mem", len(results_t_mem));

                results_g_rm = self.get_results_from_graphite_rm(time1, time2, offset0);
                results_g_spark = self.get_results_from_graphite_spark(time1, time2, offset0)
                #print "results_g_rm",results_g_rm;

                if cc == 0:
                   sc = SparkContext()

                offset = 0 # for node manager DB
                while (1): # data from graphite is fetched in batches
                    results_g_nm = self.get_results_from_graphite_nm(time1, time2, offset)
                    if results_g_nm:
                        len_nm = len(results_g_nm.raw['series'][0]['values'])
                        #print "result_g_nm",results_g_nm

                        rdd_join_tele_cpu = self.join_telegraf_metrics(results_t_cpu, sc, 'cpu');
                        #print ("tele_cpu",rdd_join_tele_cpu.collect());return
                        rdd_join_tele_mem = self.join_telegraf_metrics(results_t_mem, sc, 'mem');
                        #print ("tele_mem",rdd_join_tele_mem.collect()); return
                        rdd_join_t = self.join_rdd(rdd_join_tele_cpu, rdd_join_tele_mem);
                        #print ('rdd_join_tele_cpu_mem',rdd_join_t.collect())"""

                        rdd_join_g_nm = self.join_graphite_metrics(results_g_nm, sc, 'nm');
                        #print ("rdd_join_g_nm",rdd_join_g_nm.collect()[:2]);

                        rdd_join_g_rm = self.join_graphite_metrics(results_g_rm, sc, 'rm')
                        #print ("rdd_join_g_rm",rdd_join_g_rm.collect()[:2]);

                        rdd_join_g_spark = self.join_graphite_metrics(results_g_spark, sc, 'spark');
                        #print ('rdd_join_g_spark',rdd_join_g_spark);
                        """if not rdd_join_g_spark:
                            rdd_spark = [([0] * self.col_len_nm)]*len(rdd_join_g_nm.collect())
                            rdd_spark = sc.parallelize(rdd_spark)
                            rdd_join = rdd_spark.map(lambda x: (x[0], tuple(x[1:]))); print rdd_join
                        else:
                            rdd_join = self.join_rdd(rdd_join_g_nm, rdd_join_g_spark)"""

                        rdd_join_g_nm_t = self.join_rdd(rdd_join_t, rdd_join_g_nm) #join with time & hostname
                        #print ('join_nm_g_t', rdd_join_g_nm_t.collect());

                        rdd_join_g_nm_t_rm = self.join_rdd(rdd_join_g_nm, rdd_join_g_rm);
                        #print ("rdd_join_g_nm_t_rm",rdd_join_g_nm_t_rm.collect());

                        rdd_mysql = self.join_mysql_metrics(results_mysql, sc)
                        #print ('rdd_mysql',rdd_mysql.collect()[0]); print '\n'

                        rdd_join_g_nm_t_rm = rdd_join_g_nm_t.map(lambda x: (x[1][-1], tuple(x[0:]))) # join with app id for MySQL cluster
                        #print ('rdd_join_nm_appid', rdd_join.collect()[0]);

                        rdd_join_g_nm_t_rm_mysql = self.join_rdd(rdd_join_g_nm_t_rm, rdd_mysql)
                        print ('rdd_join_g_nm_t_rm_mysql',rdd_join_g_nm_t_rm_mysql.collect()[:1])

                        rdd_join = rdd_join_g_nm_t_rm_mysql.map(lambda x : [x[1][0][0]] + list(x[1][1]) + list(x[1][2:]) ) 
                        # x[0][0] time hostname
                        # also removing the redundant app id
                        print ('rdd_final',rdd_join.collect()[:1])
                        labels_rdd = rdd_join.map(lambda x: int(x[9]) ) # labels are indexed at 9 and 11

                        j = -6 # index of container id
                        """ Remove labels from data """
                        rdd_join = rdd_join.map(lambda x : x[0:5] + [self.hostname_lookup[x[5]]] + x[6:9] + x[9:j] +\
                                   x[j+2:j+3] )
                                   #[self.remv_cont_s(x[j])] + [self.remv_app_s(x[j+1])] + x[j+2:] )

                        #opf = csv.writer(open('data2.csv', 'w'), delimiter=',')
                        #for row in data:
                        #    opf.writerow(row)

                        labels = labels_rdd.collect()
                        data = rdd_join.collect()

                        data_train = data[:len(data)/2]
                        labels_train = labels[:len(data)/2]
                        data_test = data[len(data)/2:]
                        labels_test = labels[len(data)/2:]

                        #print rdd_join.coalesce(1).glom().collect()   # .glom()  # coalesce to reduce  no of partitions

                        train_c, test_c, train_a, test_a, training_iter, epoch_size = \
                            self.load_data_into_tensorflow(data_train, data_test, labels_train, labels_test)

                        if len_nm < self.limit:
                            break
                        offset += self.limit
                    else:
                        break

                if len_cpu < self.limit:
                    break
                offset0 += self.limit
            else:
                break
            cc += 1


    def get_data_from_csv(self):
        with open('data1.csv', 'rb') as f:
            try:
                file_reader = csv.reader(f, delimiter=',')
            except IOError:
                print "Error Reading csv File", f
                sys.exit()
            data = list(file_reader)

        print len(data)

        labels = [ d[8] for d in data ]
        data = [ x[0:8] + x[9:-3] + [self.remv_cont_s(x[-3])] + [self.remv_app_s(x[-2])] + [x[-1]] for x in data ]
        #print data
        #print labels
        result = self.load_data_into_tensorflow(data, labels)

    def main(self):
        self.get_data_from_influxdb()
        #self.get_data_from_csv()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Optional arguments for InfluxDB')
    parser.add_argument('--host', type=str, required=False,
                        default='localhost',
                        help='hostname of InfluxDB http API')
    parser.add_argument('--port', type=int, required=False, default=8086,
                        help='port of InfluxDB http API')
    parser.add_argument('--configfile', type=str, required=False, default='/home/vagrant/yarnml/config.txt',
                        help='path to config file containing username & password')
    parser.add_argument('--time1', type=int, required=False, default=1501758105000000000,
                        help='time to fetch data from influxdb from')
    parser.add_argument('--time2', type=int, required=False, default=1502196168000000000,
                        help='time to fetch data from influxdb from')


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    f = open(args.configfile, 'rb')
    info = (f.read()).split("\n")
    username = info[0]
    password = info[1]
    mysql_host = info[2]
    mysql_user = info[3]
    mysql_db = info[4]
    mysql_port = info[5]
    indbtf = InfluxTensorflow(args.host, args.port, username, password, 'graphite', mysql_host, mysql_user, mysql_db, mysql_port,
             args.time1, args.time2)
    indbtf.main()

