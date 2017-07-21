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

    def __init__(self, hostname, port, username, password, db, mysql_host, mysql_user, mysql_db, mysql_port):
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

        self.rep_None = -1
        self.col_len_spark = 8

    def query_batch(self, query, db, epoch='ns'):
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

    def training_step(self, i, update_test_data, update_train_data, X, Y_, Y, data_train, train_step, sess, col_length,
                      batch_size, labels, cross_entropy):
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

        # feed values include Python scalars, strings, lists, or numpy ndarray
        # the backpropagation training step
        sess.run(train_step, feed_dict={X: data_train, Y_: labels})

        if update_train_data:
            c = sess.run(Y_, feed_dict={X: data_train, Y_: labels})
            train_c.append(c)

        """if update_test_data:
            c = sess.run([cross_entropy], feed_dict={X: data_test, Y_: labels})
            test_c.append(c)"""

        return (train_c, test_c)

    def initialize_session(self):
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        return sess

    def train_model(self, data, labels):
        """
        train the machine learning Model

        :param: rdd_join: Resilient distributed datasets
        :return: result for ML model
        """

        total_data_size = len(data) # total rows of data
        batch_size = 10
        training_iter = total_data_size / batch_size

        col_length = len(data[0])

        data_train = data
        # data_train = np.array(data, dtype=np.float32)
        # print (data_train.shape)
        # labels = np.array(labels, dtype=np.float32)
        # print (labels.shape);

        # 1. Define Variables and Placeholders
        X = tf.placeholder(tf.float32, [batch_size, col_length], name='X') #the first dimension (None) will index the images
        Y_ = tf.placeholder(tf.float32, [batch_size,], name='Y_') # placeholder for correct answers

        # Weights initialised with small random values between -0.2 and +0.2
        W1 = tf.Variable(tf.truncated_normal([col_length, batch_size], stddev=0.1))
        B1 = tf.Variable(tf.zeros([1]))
        W2 = tf.Variable(tf.truncated_normal([batch_size, batch_size-2], stddev=0.1))
        B2 = tf.Variable(tf.zeros([1]))
        W3 = tf.Variable(tf.truncated_normal([batch_size-2, batch_size-4], stddev=0.1))
        B3 = tf.Variable(tf.zeros([1]))
        W4 = tf.Variable(tf.truncated_normal([batch_size-4, batch_size-6], stddev=0.1))
        B4 = tf.Variable(tf.zeros([1]))
        W5 = tf.Variable(tf.truncated_normal([batch_size-6, 1], stddev=0.1))
        B5 = tf.Variable(tf.zeros([1]))

        # 2. Define the model

        ######## SIGMOID activation func #######
        #Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
        #Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
        #Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
        #Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
        #Y5 = tf.nn.sigmoid(tf.matmul(Y4, W5) + B5)
        ######## ReLU activation func #######
        Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
        Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
        Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
        Y5 = tf.nn.relu(tf.matmul(Y4, W5) + B5)

        Y = Y5
        Y = tf.nn.softmax(Y5) # Ylogits
        # Y_ = tf.reshape(Y, [-1, -1])
        cross_entropy = tf.reduce_sum(tf.pow(Y - Y_, 2))/(2*batch_size) # reduce_mean

        # 5. Define an optimizer
        # optimizer = tf.train.GradientDescentOptimizer(0.5)
        optimizer = tf.train.AdamOptimizer(0.005)  ## do not use gradient descent 0.005
        train_step = optimizer.minimize(cross_entropy)

        # initialize and train
        sess = self.initialize_session()
        # 6. Train and test the model, store the accuracy and loss per iteration

        train_c = []
        test_c = []

        epoch_size = 10;

        for a in range(1):
            for i in range(training_iter):
                test = False
                if i % epoch_size == 0:
                    test = True
                c, tc = self.training_step(i, test, test, X, Y_, Y, data_train[i*batch_size:batch_size*(i+1)], train_step,
                sess, col_length, batch_size, labels[i*batch_size:batch_size*(i+1)], cross_entropy)
                train_c += c
                test_c += tc
        print ('train Cost',train_c)

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

    def load_data_into_tensorflow(self, data, labels):
        return self.train_model(data, labels)
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
                #print ('Columns',name_g, x)
                #print ('Columns',name_g, x[0:1] + x[97:98] + x[39:40] + x[44:45] + x[51:52] + x[57:59] \
                #+ x[64:65] + x[100:101] + ['app id'] )
                #print ('values',values_g)

                rdd1 = sc.parallelize(values_g)
                rdd1_ = rdd1 # remove after testing
                if source == 'nm':
                    #src = float((res['tags']['source']).replace('ContainerResource_container_e','').replace('_',''))

                    rdd1_ = rdd1.map(lambda x: x[0:1] + x[97:98] + x[39:40] + x[44:45] + x[51:52] + x[57:59] + x[64:65] +\
                    x[100:101] + [ self.get_appid_from_cont(x[100]) ] )
                    rdd1_ = rdd1_.map(lambda x: [ self.rep_None if a == None else a for a in x])
                elif source == 'spark':
                    rdd1_ = rdd1.map(lambda x: x[0:1] + x[5:6] + x[8:9] + x[17:18] + x[27:28] + x[29:30] + x[52:57] + x[58:63])
                    rdd1_ = rdd1_.map(lambda x: [ self.rep_None if a == None else a for a in x])
                    rdd1_ = rdd1_.map(lambda x: x[0:1] + [float(x[1].replace('application_','').replace('_',''))] + x[2:])
                else:
                    pass
                # converted to tuple for join/union operation to work properly
                rdd1_ = rdd1_.map(lambda x: ((x[0], (x[1]).replace('Hostname=','')), tuple(x[2:]))) 

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

    def get_results_from_telegraf_cpu(self, time1, time2):
        #query = "{0} where time > {1} and time < {2} and cpu =~ /cpu-total/ order by time desc"\
        #         .format(self.query_t_cpu, time1, time2)
        query = "{0} where cpu =~ /cpu-total/ order by time desc".format(self.query_t_cpu)
        return self.query_batch(query, db="telegraf")

    def get_results_from_telegraf_mem(self, time1, time2):
        #query = "{0} where time > {1} and time < {2} order by time desc".format(self.query_t_mem, time1, time2)
        query = "{0} order by time desc".format(self.query_t_mem)
        return self.query_batch(query, db="telegraf")

    def get_results_from_graphite_nm(self, time1, time2):
        #query = "{0} nodemanager where source =~ /container.*$/ and time > {1} and time < {2} order by time "+\
        #        "desc limit 100000".format(self.query_rns, time1, time2) # group by /time/,/cpu/,/source/
        query = "{0} nodemanager where source =~ /container.*$/ order by time desc".format(self.query_rns)
        return self.query_batch(query, db="graphite")

    def get_results_from_graphite_spark(self, time1, time2):
        query = "{0} spark where source =~ /jvm/ and service =~ /driver/ and time > {1} and time < {2} limit 10"\
                    .format(self.query_rns, time1, time2)
        return self.query_batch(query, db="graphite")

    def get_results_from_graphite_rm(self, time1, time2):
        query = "{0} nodemanager where service =~ /yarn.*$/ order by time desc limit 10".format(self.query_rns)
        return self.query_batch(query, db="graphite")

    def remv_app_s(self, string):
        return float(string.replace('application_','').replace('_',''))

    def remv_cont_s(self, string):
        return float(string.replace('ContainerResource_container_e','').replace('_',''))

    def conv_numbers_to_app_id(self, s):
        str = re.search('(.*)([0-9]{4}$)',s)
        return 'application_' + str.group(1) + '_' + str.group(2)

    def get_data_from_influxdb(self):
        time1 = 1499704325000000000
        time2 = 1499958768000000000

        results_mysql = self.get_results_from_mysql_cluster()

        results_t_cpu = self.get_results_from_telegraf_cpu(time1, time2);
        results_t_mem = self.get_results_from_telegraf_mem(time1, time2);
        #print ("result_telegraf_cpu", results_t_cpu);
        #print ("result_telegraf_mem", results_t_mem)

        results_g_nm = self.get_results_from_graphite_nm(time1, time2)
        #print "result_g_nm",results_g_nm

        results_g_spark = self.get_results_from_graphite_spark(time1, time2)
        result_g_rm = self.get_results_from_graphite_rm(time1, time2)
        if True: # results_t

            sc = SparkContext()

            rdd_join_tele_cpu = self.join_telegraf_metrics(results_t_cpu, sc, 'cpu');
            #print ("tele_cpu",rdd_join_tele_cpu.collect()); return
            rdd_join_tele_mem = self.join_telegraf_metrics(results_t_mem, sc, 'mem');
            #print ("tele_mem",rdd_join_tele_mem.collect()); return
            rdd_join_t = self.join_rdd(rdd_join_tele_cpu, rdd_join_tele_mem);
            #print ('rdd_join_tele_cpu_mem',rdd_join_t.collect())

            rdd_join_g_nm = self.join_graphite_metrics(results_g_nm, sc, 'nm');
            #print ("nm",rdd_join_g_nm.collect());return

            #rdd_join_g_spark = self.join_graphite_metrics(results_g_spark, sc, 'spark')

            #rdd_join_g_rm = self.join_graphite_metrics(results_g_rm, sc, 'rm')

            """if not rdd_join_g_spark:
                rdd_spark = [([0] * self.col_len_nm)]*len(rdd_join_g_nm.collect())
                rdd_spark = sc.parallelize(rdd_spark)
                rdd_join = rdd_spark.map(lambda x: (x[0], tuple(x[1:]))); print rdd_join
            else:
                rdd_join = self.join_rdd(rdd_join_g_nm, rdd_join_g_spark)"""

            rdd_join_g_nm_t = self.join_rdd(rdd_join_t, rdd_join_g_nm) #join with time & hostname
            #print ('join_nm_g_t', rdd_join_g_nm_t.collect());

            #rdd_join = self.join_graphite_metrics(results_t, sc, '')
            #rdd_join = self.join_telegraf_metrics(results_t, rdd_join, sc)

            rdd_mysql = self.join_mysql_metrics(results_mysql, sc)
            #print ('rdd_mysql',rdd_mysql.collect()); print '\n'

            rdd_join = rdd_join_g_nm_t.map(lambda x: (x[1][-1], tuple(x[0:]))) # join with app id
            #print ('rdd_join', rdd_join.collect());

            rdd_join = self.join_rdd(rdd_join, rdd_mysql)
            #print ('rdd t g nm mysql', rdd_join.collect()[0]);return

            rdd_join = rdd_join.map(lambda x : [x[1][0][0]] + list(x[1][1]) + list(x[1][2:]) ) # x[0][0] time hostname
            labels_rdd = rdd_join.map(lambda x: int(x[9]) ) # labels are indexed at 9 and 11
            """          exclude label from data        """
            j = -6 # index of container id
            rdd_join = rdd_join.map(lambda x : x[0:9] + x[9:j] + [self.remv_cont_s(x[j])] + [self.remv_app_s(x[j+1])] + x[j+2:] )

            labels = labels_rdd.collect()
            data = rdd_join.collect()
            print len(data)
            #print ("joined results", data[0])
            #print ("labels",labels)

            #opf = csv.writer(open('data2.csv', 'w'), delimiter=',')
            #for row in data:
            #    opf.writerow(row)

            #print rdd_join.coalesce(1).glom().collect()   # .glom()  # coalesce to reduce  no of partitions
            #result = self.load_data_into_tensorflow(data, labels)

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
    indbtf = InfluxTensorflow(args.host, args.port, username, password, 'graphite', mysql_host, mysql_user, mysql_db, mysql_port)
    indbtf.main()

