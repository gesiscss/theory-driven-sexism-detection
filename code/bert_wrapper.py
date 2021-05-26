#!/usr/bin/env python
# coding: utf-8

# In[1]:


import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

from tensorflow import keras
import os
import re

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

from sklearn.metrics import classification_report

from sklearn.base import BaseEstimator

from config import TEMP_DATA_DIR
# In[3]:

# MAX_SEQ_LENGTH = 128
MAX_SEQ_LENGTH = 50
# Compute train and warmup steps from batch size
	# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
	# Warmup is a period of time where hte learning rate 
	# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
	# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 500

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


DATA_COLUMN = 'text'
LABEL_COLUMN = 'sexist_binary'

OUTPUT_DIR = "checkpoints/bert_test"
os.environ['TFHUB_CACHE_DIR'] = '/bigdata/sexism/utils/temp/tfhub_cache/'


# In[4]:


def create_tokenizer_from_hub_module():
	"""Get the vocab file and casing info from the Hub module."""
	with tf.Graph().as_default():
		bert_module = hub.Module(BERT_MODEL_HUB)
		tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
		with tf.Session() as sess:
			vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
											tokenization_info["do_lower_case"]])
		
	return bert.tokenization.FullTokenizer(
		vocab_file=vocab_file, do_lower_case=do_lower_case)

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
				 num_labels):
	"""Creates a classification model."""

	bert_module = hub.Module(
		BERT_MODEL_HUB,
		trainable=True)
	bert_inputs = dict(
		input_ids=input_ids,
		input_mask=input_mask,
		segment_ids=segment_ids)
	bert_outputs = bert_module(
		inputs=bert_inputs,
		signature="tokens",
		as_dict=True)

	# Use "pooled_output" for classification tasks on an entire sentence.
	# Use "sequence_outputs" for token-level output.
	output_layer = bert_outputs["pooled_output"]

	hidden_size = output_layer.shape[-1].value

	# Create our own layer to tune for politeness data.
	output_weights = tf.get_variable(
		"output_weights", [num_labels, hidden_size],
		initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
		"output_bias", [num_labels], initializer=tf.zeros_initializer())

	with tf.variable_scope("loss"):

	# Dropout helps prevent overfitting
		output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

		logits = tf.matmul(output_layer, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		log_probs = tf.nn.log_softmax(logits, axis=-1)

		# Convert labels into one-hot encoding
		one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

		predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
		# If we're predicting, we want predicted labels and the probabiltiies.
		if is_predicting:
			return (predicted_labels, log_probs)

		# If we're train/eval, compute loss between predicted and actual label
		per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
		loss = tf.reduce_mean(per_example_loss)
		return (loss, predicted_labels, log_probs)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
										 num_warmup_steps):
	"""Returns `model_fn` closure for TPUEstimator."""
	def model_fn(features, labels, mode, params):	# pylint: disable=unused-argument
		"""The `model_fn` for TPUEstimator."""

		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		label_ids = features["label_ids"]

		is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
		
		# TRAIN and EVAL
		if not is_predicting:

			(loss, predicted_labels, log_probs) = create_model(
				is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

			train_op = bert.optimization.create_optimizer(
					loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

			# Calculate evaluation metrics. 
			def metric_fn(label_ids, predicted_labels):
				accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
				f1_score = tf.contrib.metrics.f1_score(
						label_ids,
						predicted_labels)
				auc = tf.metrics.auc(
						label_ids,
						predicted_labels)
				recall = tf.metrics.recall(
						label_ids,
						predicted_labels)
				precision = tf.metrics.precision(
						label_ids,
						predicted_labels) 
				true_pos = tf.metrics.true_positives(
						label_ids,
						predicted_labels)
				true_neg = tf.metrics.true_negatives(
						label_ids,
						predicted_labels)	 
				false_pos = tf.metrics.false_positives(
						label_ids,
						predicted_labels)	
				false_neg = tf.metrics.false_negatives(
						label_ids,
						predicted_labels)
				return {
						"eval_accuracy": accuracy,
						"f1_score": f1_score,
						"auc": auc,
						"precision": precision,
						"recall": recall,
						"true_positives": true_pos,
						"true_negatives": true_neg,
						"false_positives": false_pos,
						"false_negatives": false_neg
				}

			eval_metrics = metric_fn(label_ids, predicted_labels)

			if mode == tf.estimator.ModeKeys.TRAIN:
				return tf.estimator.EstimatorSpec(mode=mode,
					loss=loss,
					train_op=train_op)
			else:
					return tf.estimator.EstimatorSpec(mode=mode,
						loss=loss,
						eval_metric_ops=eval_metrics)
		else:
			(predicted_labels, log_probs) = create_model(
				is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

			predictions = {
					'probabilities': log_probs,
					'labels': predicted_labels
			}
			return tf.estimator.EstimatorSpec(mode, predictions=predictions)

	# Return the actual model function in the closure
	return model_fn


# In[15]:


def init_checkpoint_dir(output_dir, delete_existing=False):
    if delete_existing:
        try:
            tf.gfile.DeleteRecursively(output_dir)
        except:
        # Doesn't matter if the directory didn't exist
            pass
    tf.io.gfile.makedirs(output_dir)
    print('***** Model output directory: {} *****'.format(output_dir))


# In[13]:


class FinetunedBertClassifier(BaseEstimator):
    def __init__(self,         
                batch_size=BATCH_SIZE,
                num_train_epochs = NUM_TRAIN_EPOCHS,
                warmup_proportion=WARMUP_PROPORTION,
                max_seq_length=MAX_SEQ_LENGTH,  
                model_dir=OUTPUT_DIR,
                save_summary_steps=SAVE_SUMMARY_STEPS,
                save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
                learning_rate=LEARNING_RATE,
                delete_existing_model_dir=True):
        # This is a path to an uncased (all lowercase) version of BERT
        self.tokenizer = create_tokenizer_from_hub_module()
        
        self.batch_size=batch_size
        self.num_train_epochs =num_train_epochs
        self.warmup_proportion=warmup_proportion
        self.max_seq_length=max_seq_length
        self.model_dir=model_dir
        self.save_summary_steps=save_summary_steps
        self.save_checkpoints_steps=save_checkpoints_steps
        self.learning_rate=learning_rate
        self.delete_existing_model_dir = delete_existing_model_dir
        
        init_checkpoint_dir(self.model_dir, delete_existing=self.delete_existing_model_dir)

    def fit(self, X, y):
        self.labels_ = sorted(set(y))

        train_InputExamples = [run_classifier.InputExample(guid=None, text_a = x_, text_b = None, label = y_) for (x_, y_) 
                               in zip(X, y)] 

        # Convert our train and test features to InputFeatures that BERT understands.
        train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, 
                                                                          self.labels_,
                                                                          self.max_seq_length, 
                                                                          self.tokenizer)


        # Compute # train and warmup steps from batch size
        num_train_steps = int(len(train_features) / self.batch_size * self.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        # Specify outpit directory and number of checkpoint steps to save
        run_config = tf.estimator.RunConfig(
            model_dir=self.model_dir,
            save_summary_steps=self.save_summary_steps,
            save_checkpoints_steps=self.save_checkpoints_steps)

        model_fn = model_fn_builder(
                num_labels=len(self.labels_),
                learning_rate=self.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps)

        self.estimator_ = tf.estimator.Estimator(
                model_fn=model_fn,
                config=run_config,
                params={"batch_size": self.batch_size})


        # Create an input function for training. drop_remainder = True for using TPUs.
        train_input_fn = bert.run_classifier.input_fn_builder(
            features=train_features,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=False)


        print('Beginning Training!')
        current_time = datetime.now()
        self.estimator_.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print("Training took time ", datetime.now() - current_time)

        
        return self
    def predict(self, X):

        input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in X] # here, "" is just a dummy label
        tokenizer = self.tokenizer
        input_features = run_classifier.convert_examples_to_features(input_examples, self.labels_, self.max_seq_length, tokenizer)
        predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=self.max_seq_length,
                                                           is_training=False, drop_remainder=False)
        predictions = self.estimator_.predict(predict_input_fn)
        return [self.labels_[prediction['labels']] for prediction in predictions]


# In[16]:


if __name__ == "__main__":
	print("HI!")
	print(OUTPUT_DIR)
	#FinetunedBertClassifier()
    # train = pd.read_csv("../sexist_data/split_data_others/sexist_nonsexist_callme_trains_1.csv", sep = "\t") 
    # test = pd.read_csv("../sexist_data/split_data_others/sexist_nonsexist_tests_1.csv",	sep = "\t")

    # from config import TEMP_DATA_DIR

    # b = FinetunedBertClassifier(num_train_epochs=1, model_dir=os.path.join(TEMP_DATA_DIR, 'bert_test/'), )

    # train.head()

    # X, y=train[:100].text.values, train[:100].sexist_binary.values

    # b.fit(X, y)

    # X, y=test.text.values, test.sexist_binary.values

    # y_pred = b.predict(X)

    # print( classification_report( y, y_pred,))


# In[ ]:




