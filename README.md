# Repository for installing python script for automatic scoring of mindreading.

# Installation instructions and content of the repo

# 1.	Download the files from this github repo  

-	Using a git client (command line git, github desktop)
-	Go to the repository -> “code” -> “download zip”

# 2.	Installation

-	If you are only going to use the code for scoring, you do not need to setup a GPU
-	If you are going to re-train the model GPU is recommended

-	These installation instructions are based on command line python and pip, you can also use anaconda if you prefer
for ubuntu you must install pip via:
apt-get install python3-pip

-	For detailed instructions on installing python and TensorFlow with GPU support, please see https://www.tensorflow.org/install/gpu . If you have NVIDIA graphic cart, you will likely have to install CUDA and cuDNN.

-	Installing necessary python packages (n.b.: if the versions of transformers and/or tensorflow are different, you might need to re-train the transformer)

pip install pandas scipy nltk spacy gensim sklearn tensorflow tensorflow_datasets pathvalidate transformers

N.B.: if a standard DigitalOcean server with 1 GB ram is used, you MUST set swap in order to install tensorflow

sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

-	Installing nltk dat – “popular” (requires for the tokenization)
https://www.nltk.org/data.html# 

# 3.	Contents of the repository

-	The “Import” folder – contains definitions of classes and functions, you should not modify it

-	The “Models” folder – this is where you need to put the models that the system will be using for scoring. 
o	N.B.: after training a new model, you need to copy it from “Custom_Models” to “Models”
o	N.B.: our best model is available for download per request. Please get in touch if you need to download the model

-	The “Custom_models” folder – this is where new models are generated after training

-	The “Raw_Data” folder – this is where you put data that needs to be scored. See below for expected format

-	The “Scored” folder – this is where the script will put the “raw data” after scoring it

-	The “Training_Data” folder – this is where you put data for training a new model

N.B.: some of these folders may be missing from the repository (if there are currently no files in them). You should create them.

# 4.	mr_config

-	The file “mr_config.py” contains the configuration for both training and scoring

-	The _dir variables indicate the path to specific folders, you should not change those

-	 The “text_cols” variable indicates the expected names of the columns for child responses

-	The “rate_cols” variable indicates the expected names of score columns. The ORDER of the rate cols must correspond to the order of “text cols”. 
e.g.: If “SFQuestion_1_Text” is the first in text cols, the first column in “rate cols” should be “SFQ1_Rating”

-	The “questions” is the list of questions on the test. The ORRDER of the “questions” must correspond to the order of “text cols”.
e.g.: If “SFQuestion_1_Text” is the first in text cols, the first column in questions should be “Why did the men hide ?”

-	The “misc_cols” variable indicates the expected “other” columns that need to be kept by the system. Those columns might include things like age, gender, or MHVC scores. By default, the only “other” column kept is ID – the identificatory of the child. ID column should NOT be removed.

-	The “r_ages” variable indicates the expected age groups. This is only used when printing evaluation by age. Evaluation by age is disabled by default. It can only be carried out if corresponding Age column exists in the data. Using age scoring might result in some code errors, it is suggested that you disable it for the moment.

# 5.	score_transformer.py

-	The script “score transformer” is the main functionality of the program

-	Before running the script make sure that 
o	you have installed all necessary packages
o	you have checked the mr_config.py
o	you have put your model in Models
o	you have put your excel file in Raw_Data

-	Execute the file from command line, by running 
python score_transformer.py MODEL_NAME DATA_NAME

MODEL_NAME – the name of the folder in Models, where your model is
DATA_NAME – the filename of the data you want scored in Raw_Data


# 6.	retrain_transformer.py

-	The script “retrain transformer” has two functions – to test how model can perform after training on new data; to create a new model

-	Execute the file from command line by running
python retrain_transformer.py train TRAIN_DATA MODEL_NAME

TRAIN_DATA – the filename of the data in Train_Data folder
MODEL_NAME – the name under which the new model will be saved in “Custom_Models”

N.B.: after training a new model, you need to copy it from Custom_Models to Models in order to be able to use it for scoring

You can also “check” the trained model by executing
python retrain_transformer.py eval TRAIN_DATA MODEL_NAME

Instead of saving a model, the “eval” splits the train data 80/20 train/test and reports the accuracy and f1.
 
# 7.	Models and corpora

-	We can provide our training data per request
-	We can provide our best performing system per request
