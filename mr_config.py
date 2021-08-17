# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 14:02:44 2021

@author: Venelin
"""

# Separate columns by type

# Input columns
text_cols = ['SFQuestion_1_Text', 'SFQuestion_2_Text', 'SFQuestion_3_Text', 
             'SFQuestion_4_Text', 'SFQuestion_5_Text', 'SFQuestion_6_Text', 
             'SS_Brian_Text', 'SS_Peabody_Text', 'SS_Prisoner_Text', 
             'SS_Simon_Text', 'SS_Burglar_Text']

# Score columns
rate_cols = ['SFQ1_Rating', 'SFQ2_Rating', 'SFQ3_Rating', 
             'SFQ4_Rating', 'SFQ5_Rating', 'SFQ6_Rating', 
             'SS_Brian_Rating', 'SS_Peabody_Rating', 'SS_Prisoner_Rating',
             'SS_Simon_Rating', 'SS_Burglar_Rating']

# Other columns
# Commented line - example in case we want to control for more variables
#misc_cols = ['ID', 'CID', 'Gender', 'AgeR', 'AgeG', 'AgeGroup', 'MHVS', 'SEND']
misc_cols = ['ID']

# Add the questions
questions = ['Why did the men hide ? ', 'What does the woman think ? ', 'Why did the driver lock Harold in the van ? ', 
            'What is the deliveryman feeling and why ? ', 'Why did Harold pick up the cat ? ', 'Why does Harold fan Mildred ? ',
            'Why does Brian say this ? ', 'Why did she say that ? ', 'Why did the prisoner say that ? ',
            'Why will Jim look in the cupboard for the paddle ? ', 'Why did the burglar do that ? ']

# Age groups
r_ages = [8,9,10,11,12,13]

# Max len for truncating
m_r_len = 35

# Path to input
input_dir = 'Raw_Data/'

# Path to models
model_dir = 'Models/'

# Path to output
out_dir = 'Scored/'

# Path to training data
train_input_dir = "Train_Data/"

# Path to save re-trained models
save_model_dir = "Custom_Models/"

# Parameters for evaluation, whether to print evaluation by Question and Age
EvalQ = True
EvalAge = False
