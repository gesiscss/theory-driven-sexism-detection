#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

from sklearn.base import BaseEstimator

from sklearn.preprocessing import OneHotEncoder

import numpy as np
import re
import nltk
from nltk.corpus import stopwords




def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    stops = set(stopwords.words('english'))
    stops.discard('not')
    #stops.discard('as')
    #stops.discard('more')
    
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    #check stopwords
    

    string_words = string.split(" ")
#     string_words = list(set(string_words) - set(stops)) #doesn't this mess u with word order?
    string_words = list(word for word in string_words if word not in stops)
    ##print(string_words)
    string = " ".join(string_words)

    return string.strip().lower()


# In[9]:


def preprocess_cnn(X):
    # Data Preparation
    # ==================================================
    X_ = list(map(clean_str, X))
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in X_])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(X_)))
    return x, vocab_processor

def split_train_dev(X, y, dev_sample_percentage):

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    return x_train, y_train, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev, 
            check_dir,
            allow_soft_placement,
            log_device_placement,
            embedding_dim,
            num_filters,
            filter_sizes,
            l2_reg_lambda,
            num_checkpoints,
            dropout_keep_prob,
            evaluate_every,
            checkpoint_every,
            batch_size, 
            num_epochs):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=allow_soft_placement,
          log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            
            
            
            
            out_dir = os.path.abspath(os.path.join(check_dir, "runs", timestamp))
            #print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch, dropout_keep_prob):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)), batch_size, num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch, dropout_keep_prob)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    #print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    #print("")
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #print path
                    #print("Saved model checkpoint to {}\n".format(path))
    return path




class CharCNN(BaseEstimator):
    def __init__(self, checkpoint_dir='../utils/temp/cnn_test',
                dev_sample_percentage = .1,
                embedding_dim = 128 ,
                filter_sizes = "3,4,5",
                num_filters = 128 ,
                dropout_keep_prob = 0.5,
                l2_reg_lambda = 0.,
                batch_size = 64 ,
                num_epochs = 400 ,
                evaluate_every = 200 ,
                checkpoint_every = 200 ,
                num_checkpoints = 1 ,
                allow_soft_placement = True ,
                log_device_placement = False ,
                eval_train = False,
                ):
        self.checkpoint_dir = checkpoint_dir
        self.dev_sample_percentage = dev_sample_percentage
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.num_checkpoints = num_checkpoints
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.eval_train = eval_train
    def fit(self, X, y):
        X_, self.vocab_ = preprocess_cnn(X)
        self.y_encoder_ = OneHotEncoder(sparse=False, categories='auto')
        
        y_ = self.y_encoder_.fit_transform(y.reshape(-1, 1))
        x_train, y_train, x_dev, y_dev = split_train_dev(X_, y_, self.dev_sample_percentage)
        self.model_checkpoint_ = train(x_train, y_train, self.vocab_, x_dev, y_dev, self.checkpoint_dir,
                                    self.allow_soft_placement,
                                    self.log_device_placement,
                                    self.embedding_dim,
                                    self.num_filters,
                                    self.filter_sizes,
                                    self.l2_reg_lambda,
                                    self.num_checkpoints,
                                    self.dropout_keep_prob,
                                    self.evaluate_every,
                                    self.checkpoint_every,
                                    self.batch_size, 
                                    self.num_epochs)
        
        return self
    def predict(self, X):
        X_ = list(map(clean_str, X))
        vocab_processor = self.vocab_
        x_test = np.array(list(vocab_processor.transform(X_)))

        checkpoint_file = tf.train.latest_checkpoint(os.path.dirname(self.model_checkpoint_))
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
            # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                batches = batch_iter(list(x_test), self.batch_size, 1, shuffle=False)

                # Collect the predictions here
                all_predictions = []

                for x_test_batch in batches:
                    batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
        return list(self.y_encoder_.categories_[0][int(i)] for i in all_predictions)


# In[11]:


def main():
    cn = CharCNN()

    import pandas as pd

    train_ = pd.read_csv("../sexist_data/split_data_others/sexist_nonsexist_callme_trains_1.csv", sep = "\t") 
    test_ = pd.read_csv("../sexist_data/split_data_others/sexist_nonsexist_tests_1.csv", sep = "\t")


    X, y=train_.text.values, train_.sexist_binary.values


    cn.fit(X, y)

    from sklearn.metrics import classification_report

    X, y=test_.text.values, test_.sexist_binary.values

    y_pred = cn.predict(X)
    print( classification_report( y, y_pred,))


# In[15]:


# print(datetime.datetime.now().isoformat())
# main()


# print(datetime.datetime.now().isoformat())


# In[ ]:




