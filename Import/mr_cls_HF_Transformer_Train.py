# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:01:15 2020


Class for training a new HuggingFace Distilbert Transformer.

Simplified class - training, testing, saving.

"""

import pandas as pd
import numpy as np
import scipy
import nltk
import spacy
import glob
import csv
import sklearn

from sklearn.model_selection import LeaveOneOut,KFold,train_test_split,cross_val_score
from sklearn.feature_selection import chi2
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import re
from scipy import sparse
import tensorflow_datasets as tfds
import tensorflow as tf
import collections
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import transformers
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, TFBertForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification



# Custom imports
# Some of those functions can probably be incorporated as methods in the class
#from mr_generic_scripts import *

class MR_transformer:
    
    def __init__(self, text_cols, age_list, class_names, max_len):
        
        # Initialize the core variables
        
        # The current classifier 
        self.mr_c = None
        
        # The current tokenizer model
        self.mr_t = None
               
        # Initialize model variables
        
        self.mr_set_model_vars(text_cols, age_list, class_names, max_len)
    
    # Function that sets model variables
    # Input: list of questions, list of ages, size of vocabulary, max len of sentence
    # Also includes certain pre-build variables for truncating
    # Also includes certain pre-built variables for dataset creation (batch size, shuffle buffer)
    def mr_set_model_vars(self, text_cols, age_list, class_names, max_len,
                          model_name = 'distilbert-base-uncased', batch_size = 6, 
                          l_rate = 8e-5, train_iter = 4):
        
        # List of questions
        self.q_list = text_cols
        
        # List of ages
        self.age_list = age_list
        
        # Size of the vocabulary
        self.class_names = class_names
        
        # Padding length
        self.max_len = max_len
        
        # Transformer to use
        self.model_name = model_name
        
        # Batch size
        self.batch_size = batch_size
        
        # Learning rate
        self.l_rate = l_rate
        
        # Number of training iteratioins
        self.train_iter = train_iter
    

    # Function that sets evaluation variables
    def mr_set_eval_vars(self, eval_q, eval_age, return_err = False):
        
        # Whether or not to perform evaluation by question
        self.eval_q = eval_q
        
        # Whether or not to perform evaluation by age
        self.eval_age = eval_age
        
        # Whether or not to return wrong predictions
        self.return_err = return_err

    # Function that trains the classifier
    # Input - a train set, and a val set
    def mr_train(self, train_df, val_df=None):
        
        # Reset the model at the start of each training
        
        self.mr_t = DistilBertTokenizer.from_pretrained(self.model_name)
        
        self.mr_c = TFDistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels = len(self.class_names))
        
        # Preproceess the training
        train_data = self.mr_t(train_df["Answer"].tolist(), truncation=True, padding = True) 
        train_dataset = tf.data.Dataset.from_tensor_slices((
                dict(train_data),
                train_df["Score"].tolist()
                ))
        
        if val_df is not None:
            val_data = self.mr_t(val_df["Answer"].tolist(), truncation=True, padding = True)
            val_dataset = tf.data.Dataset.from_tensor_slices((
                    dict(val_data),
                    val_df["Score"].tolist()
                    ))            
            
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.l_rate)
        self.mr_c.compile(optimizer=optimizer, loss=self.mr_c.compute_loss,metrics=['accuracy'])
        
        if val_df is not None:
            self.mr_c.fit(train_dataset.shuffle(1000).batch(self.batch_size), 
                      epochs=self.train_iter, 
                      batch_size=self.batch_size, 
                      validation_data=val_dataset.batch(self.batch_size)
                      )
        else:
            self.mr_c.fit(train_dataset.shuffle(1000).batch(self.batch_size), 
                      epochs=self.train_iter, 
                      batch_size=self.batch_size
                      )            


        
      
        
    # Function that evaluates the model on a test set
    # Input - test set
    def mr_test(self, test_df):
        
        # Initialize output vars
        acc_scores = []
        f1_scores = []
        
        X_test = test_df['Answer'].tolist()
        y_test = test_df['Score'].tolist()
        
        # Preproceess the testing
        test_data = self.mr_t(X_test, truncation=True, padding = True) 
        test_dataset = tf.data.Dataset.from_tensor_slices((
                dict(test_data),
                y_test
                ))        
        
        print("Testing the model on the test set:")
               
        # Get the actual predictions of the model for the test set
        y_pred = self.mr_c.predict(test_dataset.batch(1))
        y_pred = np.argmax(y_pred.logits, axis=1).tolist()
        
        # Calculate accuracy
        test_acc = accuracy_score(y_test, y_pred)
        
        # Calculate macro F1
        macro_score = sklearn.metrics.f1_score(y_test, y_pred, average='macro')
        
        print('Test Accuracy: {} \n'.format(round(test_acc,2)))
        print('Test Macro F1: {} \n'.format(round(macro_score,2)))
        
        # Add the results to the output
        acc_scores.append(round(test_acc,2))
        f1_scores.append(round(macro_score,2))
        
        # Test by question (if requested)
        # Add the scores to the output
        # Otherwise add empty list
        if self.eval_q:
            qa_scores, qf_scores = self.mr_eval_col(test_df,"Question",self.q_list)
            
            acc_scores.append(qa_scores)
            f1_scores.append(qf_scores)
        else:
            acc_scores.append([])
            f1_scores.append([])
            
        # Test by age (if requested)
        # Add the scores to the output
        # Otherwise add empty list    
        if self.eval_age:
            aa_scores, af_scores = self.mr_eval_col(test_df,"Age",self.age_list)
            
            acc_scores.append(aa_scores)
            f1_scores.append(af_scores)
        else:
            acc_scores.append([])
            f1_scores.append([])
            
        return(acc_scores,f1_scores)
            
            
    # Function that evaluates the model by a specific column
    # Can also return the actual wrong predictions
    # Input - test set, column, values
    def mr_eval_col(self, test_df, col_name, col_vals):
        # Initialize output
        acc_scores = []
        f1_scores = []
        
        # Initialize output for wrong predictions, if needed
        if self.return_err:
            wrong_pred = []
        
        # Loop through all values
        for col_val in col_vals:
            
            # Initialize output for wrong predictions, if needed
            if self.return_err:
                cur_wrong = []
            
            # Get only the entries for the current value
            cur_q = test_df[test_df[col_name] == col_val].copy()
            
            # Convert dataframe to dataset
            X_test = cur_q['Answer'].tolist()
            y_test = cur_q['Score'].tolist()

            # Preproceess the testing
            test_data = self.mr_t(X_test, truncation=True, padding = True) 
            test_dataset = tf.data.Dataset.from_tensor_slices((
                    dict(test_data),
                    y_test
                    ))       
            
            print("Evaluating column {} with value {}".format(col_name,col_val))
                      
            # Get the actual predictions of the model for the test set
            y_pred = self.mr_c.predict(test_dataset.batch(1))
            y_pred = np.argmax(y_pred.logits, axis=-1).tolist()
            
            # Calculate accuracy
            test_acc = accuracy_score(y_test, y_pred)
            
            # Calculate macro F1
            macro_score = sklearn.metrics.f1_score(y_test, 
                                                   y_pred,
                                                   average='macro')
            
            print('Accuracy: {} \n'.format(round(test_acc,2)))
            print('Macro F1: {} \n'.format(round(macro_score,2)))    
            
            # Add the results to the output
            acc_scores.append(round(test_acc,2))
            f1_scores.append(round(macro_score,2))
            
            if self.return_err:
                # Loop through all predictions and keep the incorrect ones
                # cur_q["Answer"], y_test, and y_pred are all matched, since they
                # are not shuffled (shuffle only applies to the test_dataset)
                for c_text,c_gold,c_pred in zip(cur_q["Answer"],y_test,
                                                y_pred):
                    if c_pred != c_gold:
                        cur_wrong.append([c_text,c_gold,c_pred])
                wrong_pred.append(cur_wrong)
            
        # Return the output
        if self.return_err:
            return(acc_scores,f1_scores, wrong_pred)
        else:
            return(acc_scores, f1_scores)

        
    # Function for a dummy one run on train-test
    # Input - full df, ratio for splitting on train/val/test, return errors or not
    def mr_one_train_test(self, full_df, test_r, val_r=0):
        
        # Split train and test
        train_df, test_df = train_test_split(full_df, test_size = test_r)
        
        # Check if we also need val
        if val_r > 0:
            train_df, val_df = train_test_split(train_df, test_size = val_r)
        else:
            # If not, validation is same as test
            val_df = test_df
            
        # Train the classifier
        self.mr_train(train_df, val_df)
        
        # Test the classifier
        return(self.mr_test(test_df))
    
    # Function that saves the model
    def mr_save_model(self, out_path):
        
        self.mr_c.save_pretrained(out_path)

