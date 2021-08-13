# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:02:13 2021

@author: Venelin
"""

## Import section

import pandas as pd
import numpy as np
import scipy
import nltk
import spacy
import gensim
import glob
import csv
import sklearn
from sklearn.model_selection import cross_val_score
import sklearn.model_selection
import sklearn.pipeline
import re
from sklearn import svm
from sklearn import *
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
import tensorflow_datasets as tfds
import tensorflow as tf
import collections
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import LeaveOneOut,KFold,train_test_split
import ktrain
from ktrain import text
from sklearn.metrics import accuracy_score
import sys
import argparse
from pathvalidate import sanitize_filepath
import time

## Custom imports
sys.path.append('Import/')
from mr_general_imports import *
from mr_cls_Transformer import *

# Import config
from mr_config import *

# Function for creating a new model
# Takes path to model
# column names, age lisits, labels, truncation size are predefined, can be 
# overwritten
def imp_mind_tr(fpath, inp_cols, age_list, m_len, y_lab = [0,1,2]):
    
    # Create new empty classifier
    t_cl = MR_transformer(inp_cols, age_list, y_lab, m_len)
    
    # Set output parameters
    # params:  Eval by Question, Eval by Age, Return Errors (not implemented)
    t_cl.mr_set_eval_vars(True,False,False)
    
    t_cl.mr_load_model(fpath)
    
    return(t_cl)

# Program body
if __name__ == "__main__":
    # Initialize argparser to process command line args
    my_parser = argparse.ArgumentParser(description='Automatic scoring of mindreading given a model and an input csv')

    # Add the arguments
    my_parser.add_argument('Model',
                           metavar='model_path',
                           type=str,
                           help='the path to the ML model')

    my_parser.add_argument('CSV',
                           metavar='csv_path',
                           type=str,
                           help='the path to the csv data')

    # Read the arguments            
    args = my_parser.parse_args()

    # Pass arguments to variables
    # Sanitize input to avoid funny business
    model_path = model_dir + sanitize_filepath(args.Model)
    csv_path = input_dir + sanitize_filepath(args.CSV)
      
    # Load transformer with predefined parameters
    # path to model, X columns, age range, max len
    # values are in the config
    tr_cls = imp_mind_tr(model_path, text_cols, r_ages, m_r_len)

    # Load raw data for test
    # Path to csv; X columns
    raw_data = mr_read_new_data(csv_path,text_cols)
    
    # Split the data per questions, needed for the ML alg
    # raw data; X columns, Y columns, Other columns to keep, list of Qs
    qa_dataset = mr_create_qa_data(raw_data,text_cols,misc_cols,questions)


    # Score the dataset 
    scored_df = tr_cls.mr_score_data(qa_dataset)
    
    # Reformat the dataframe to mix all the questions
    processed_df = out_mind_tr(scored_df, misc_cols, text_cols)

    # Save the processed file
    # Processing folder is in the config
    out_fpath = out_dir + "processed_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
    processed_df.to_csv(out_fpath,index=False)

#print(processed_df)