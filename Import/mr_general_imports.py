# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:54:16 2021

@author: Venelin
"""

## Import section

import pandas as pd
import numpy as np
import scipy
import nltk
import glob
import csv

# Dummy function for tokenization
# Def a funciton for basic preprocessing
# This is only for the DL experiments!
# Input is a string
def mr_tok_nltk(response):
    # Tokenization (nltk for NNs, we only tokenizee)
    # Discard nans as they break nltk
    if pd.isna(response) or response=="":
        tok_res = []
    else:
        try:
            tok_res = nltk.word_tokenize(response)
        except:
            tok_res = []
       
    # Return a string
    
    return " ".join(tok_res)


# Function that reads the original csv file and applies preprocessing
# Expected input:
# inp_fpath - path to csv file
# X_cols - columns to be tokenized
# anon - whether to reset the child IDs or not
# id_col = the name of the child id col to reset
# r_col = by default set to FALSE, if we need to rename columns, we also
# need to suply r_dict to do the renaming
def mr_read_new_data(inp_fpath, X_cols, mr_anon=False, id_col = "ID",
                     r_col=False, r_dict={}):
    # Read the file
    new_df = pd.read_csv(inp_fpath, index_col=None, header=0)
    
    # Check if we need to rename columns
    if r_col:
        new_df.rename(columns=r_dict,inplace=True)
    
    # Tokenize the data, we use nltk
    new_df[X_cols] = new_df[X_cols].applymap(mr_tok_nltk)
    
    # Check if we need to anonymize
    if mr_anon:
        new_df[id_col] = new_df.index
        
    return(new_df)
    

# Function for splitting input per question (expected ML format)
# Expected input:
# full_df - the original dataset
# X_cols - column names of X values
# misc_cols - other column names that need to be kept
# list_qs - list of questions corresponding to X_cols
# drop_na - whether to drop missing columns or not
def mr_create_qa_data(full_df, X_cols, misc_cols, list_qs, drop_na=True):
    
    # Initialize a list of datasets
    q_datasets = []

    # Loop through all answer / value pairs
    for cur_X, cur_q in zip(X_cols, list_qs):
        
        # List of columns to keep - misc + current X
        cols_to_keep = misc_cols + [cur_X]
        
        # Get the data
        cur_df = full_df[cols_to_keep].copy()
        
   
        # Drop missing data and -99s
        # N.B.: this might result in minor difference between input/output (!)
        # It is generally recommended nevertheless
        if drop_na:
            cur_df[cur_X].replace('', np.nan, inplace=True)
            cur_df.loc[:,misc_cols].fillna('-99',inplace=True)
            cur_df.dropna(inplace=True)
        # If we want to keep data integrity
        else:
            cur_df[cur_X].replace('', '-99', inplace=True)
            cur_df.fillna('-99',inplace=True)
            
        
        # Add column for the current question (this is needed later on)
        # Since we rename all X columns to "answer", we need to keep track
        # of the original question
        cur_df["Question"] = cur_X
        
        # Rename columns for consistency
        cur_df.rename(columns={cur_X:'Answer'},inplace=True)
        
        # Add the question to the answer -> improves performance
        cur_df['Answer'] = cur_q + " " + cur_df['Answer'].astype(str)

        # Put the data in the list
        q_datasets.append(cur_df)

    # Get a full dataset with all the data
    # In this version of the code we don't keep the separate datasets per
    # question - it is not necessary
    full_dataset = pd.concat(q_datasets)

    # Return the full dataset
    return(full_dataset)

# Function for splitting input per question (expected ML format)
# Expected input:
# full_df - the original dataset
# X_cols - column names of X values
# y_cols - column names of Y values
# misc_cols - other column names that need to be kept
# list_qs - list of questions corresponding to X_cols
# drop_na - whether to drop missing columns or not
def mr_create_qa_train_data(full_df, X_cols, y_cols, misc_cols, list_qs, drop_na=True):
    
    # Initialize a list of datasets
    q_datasets = []

    # Loop through all answer / value pairs
    for cur_X, cur_y, cur_q in zip(X_cols, y_cols, list_qs):
        
        # List of columns to keep - misc + current X
        cols_to_keep = misc_cols + [cur_X] + [cur_y]
        
        # Get the data
        cur_df = full_df[cols_to_keep].copy()
        
   
        # Drop missing data and -99s
        # N.B.: this might result in minor difference between input/output (!)
        # It is generally recommended nevertheless
        if drop_na:
            cur_df[cur_X].replace('', np.nan, inplace=True)
            cur_df[cur_y].replace(-99, np.nan, inplace=True)
            cur_df.loc[:,misc_cols].fillna('-99',inplace=True)
            cur_df.dropna(inplace=True)
        # If we want to keep data integrity
        else:
            cur_df[cur_X].replace('', '-99', inplace=True)
            cur_df.fillna('-99',inplace=True)
            
        
        # Add column for the current question (this is needed later on)
        # Since we rename all X columns to "answer", we need to keep track
        # of the original question
        cur_df["Question"] = cur_X
        
        # Rename columns for consistency
        cur_df.rename(columns={cur_X:'Answer'},inplace=True)
        cur_df.rename(columns={cur_y:'Score'},inplace=True)
        
        # Add the question to the answer -> improves performance
        cur_df['Answer'] = cur_q + cur_df['Answer'].astype(str)

        # Put the data in the list
        q_datasets.append(cur_df)

    # Get a full dataset with all the data
    # In this version of the code we don't keep the separate datasets per
    # question - it is not necessary
    full_dataset = pd.concat(q_datasets)

    # Return the full dataset
    return(full_dataset)
    
# Function for analyzing results
def mr_proc_results(raw_results):
  # Process the results from the 10 runs
  # result format: [acc, acc per q, acc per age], [f1, f1 per q, f1 per age], [acc, acc per q, acc per age] (for wp1), [f1, f1 per q, f1 per age] (for wp1)
  # Ignore ages as they seem to be mostly consistent with global average
  # Ignore accs per question and age as averaging them seems to be consistent with global average
  # Report global acc, global macro f1, average of macro f1 per question; same for wp1
  pr_results = [[[acc_score, f1_score,round(sum(qf_s)/11,2)],[acc_score_wp1, f1_score_wp1,round(sum(qf_s_wp1)/11,2)]] 
                for ([acc_score, qa_s, aa_s], [f1_score, qf_s, af_s],
                     [acc_score_wp1, qa_s_wp1, aa_s_wp1], [f1_score_wp1, qf_s_wp1, af_s_wp1]) in raw_results]

  # Throw the list in an np array
  pr_arr = np.array(pr_results)

  # Print the results
  import pprint
  pp = pprint.PrettyPrinter(indent=4)

  pp.pprint(pr_results)
  pp.pprint(np.mean(pr_arr,axis=0))
  
  

# Function for preparing the output
# Takes scored dataframe, list of out cols, list of data columns
# Joins together multiple dataframes for the same participant
def out_mind_tr(scr_df, out_cols, inp_cols):
    
    # Create the full list of columns for the output
    # Start by initializigint it to out_cols
    o_cols = out_cols.copy()
    # Each text column has two entries in the outptut - response and score
    for cur_col in inp_cols:
        o_cols.append(cur_col)
        o_cols.append(cur_col[:-4] + 'Rating')
        
    # Create emtpy dataframe for the output
    processed_df = pd.DataFrame(columns = o_cols)
    
    # Loop through all entries of the scored df
    # N.B.: the scored dataframe contains all questions as separate lines
    # We loop through all of them and recombine them
    for cur_index, cur_line in scr_df.iterrows():
        # Current question and rating
        cur_q = cur_line["Question"]
        cur_r = cur_q[:-4] + 'Rating'
        
        # Check if we have entry for this participant already
        if cur_line["ID"] in processed_df['ID'].values:
            # If so, just update current question and its rating
            processed_df[cur_q] = np.where(processed_df['ID'] == cur_line["ID"],
                                           cur_line["Answer"],processed_df[cur_q])
            processed_df[cur_r] = np.where(processed_df['ID'] == cur_line["ID"],
                                           cur_line["Score"],processed_df[cur_r])
            
        # If there is no entry, we have to create it
        else:
            # Initialize dictionary
            inp_dict= {}
            # Add all the out_columns
            for o_col in out_cols:
                inp_dict[o_col] = cur_line[o_col]
            # Add the answer and score
            inp_dict[cur_q] = cur_line["Answer"]
            inp_dict[cur_r] = cur_line["Score"]
            
            # Create a dataframe with the dictionary
            inp_df = pd.DataFrame(inp_dict,index=[0])
            
            # Add the data to the processed dataframe
            # Fill all missing values (other questions) with -99
            processed_df = pd.concat([processed_df,inp_df],ignore_index=True).fillna(-99)
            
    return(processed_df)
    