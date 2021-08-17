# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:54:09 2021

@author: Venelin
"""

import pandas as pd
import numpy as np
import scipy
import nltk
import spacy
import gensim
import glob
import csv
import sklearn
import re
from sklearn.model_selection import LeaveOneOut,KFold,train_test_split,cross_val_score
from sklearn.feature_selection import chi2
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from scipy import sparse
import tensorflow_datasets as tfds
import tensorflow as tf
import collections
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import argparse
from pathvalidate import sanitize_filepath
import time

import transformers
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, TFBertForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification


## Custom imports
sys.path.append('Import/')
from mr_general_imports import *
from mr_cls_HF_Transformer_Train import *

# Import config
from mr_config import *
 
# Program body
if __name__ == "__main__":
    # Initialize argparser to process command line args
    my_parser = argparse.ArgumentParser(description='Retraining a mindreading classifier, given a training set and a path to save the model.')

    # Add the arguments
    my_parser.add_argument('Action',
                           metavar='cur_action',
                           type=str,
                           help='Action - either "eval" or "train"')    
    
    # Add the arguments
    my_parser.add_argument('Train',
                           metavar='train_file',
                           type=str,
                           help='the path to the training data')
    
    my_parser.add_argument('Model',
                           metavar='model_path',
                           type=str,
                           help='the path to save the ML model')



    # Read the arguments            
    args = my_parser.parse_args()
    
    if args.Action.lower() not in ["train","eval"]:
        print("Invalid action, must be either train or eval!")
        exit(1)

    # Pass arguments to variables
    # Sanitize input to avoid funny business
    save_path = save_model_dir + sanitize_filepath(args.Model)
    train_path = train_input_dir + sanitize_filepath(args.Train)
    
    # Load raw data for training
    # Path to csv; X columns
    raw_data = mr_read_new_data(train_path,text_cols)

    # Split the data per questions, needed for the ML alg
    # raw data; X columns, Y columns, Other columns to keep, list of Qs
    qa_dataset = mr_create_qa_train_data(raw_data,text_cols,rate_cols,misc_cols,questions)
    
    # create a new transformer, take variables from config
    tr_cls = MR_transformer(text_cols,r_ages,[0,1,2],m_r_len)
    tr_cls.mr_set_eval_vars(EvalQ,EvalAge,False)
    
    if args.Action.lower() == "eval":
        tr_cls.mr_one_train_test(qa_dataset,0.2)
    else:
        tr_cls.mr_train(qa_dataset)
        tr_cls.mr_save_model(save_path)