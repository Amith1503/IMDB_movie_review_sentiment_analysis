# import required packages
from train_NLP import text_cleaner
import pickle as pkl
from keras.models import load_model
import os
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np




# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__": 
	# 1. Load your saved model
	NLP_model=load_model("models/20859891_NLP_model.hdf5")
	# NLP_model=load_model("/home/amith/Downloads/assignment_3/wemodel_new_64_1_lay_callback.hdf5")

	# 2. Load your testing data
	# TEST DATA
	Test_data_uncleaned=[]
	Test_data_cleaned=[]
	y_tst_labels=[]

	pos_tst_path="data/aclImdb/test/pos"
	neg_tst_path= "data/aclImdb/test/neg"
	pos_tst_data=sorted(os.listdir(pos_tst_path))
	neg_tst_data=sorted(os.listdir(neg_tst_path))
	for pos in pos_tst_data:
	#     print(pos)
	    with open (os.path.join(pos_tst_path,pos)) as f:

	        data_tst_pos= f.readlines()
	        Test_data_uncleaned.append(data_tst_pos)
	        
	        data_pos_tst_clean=text_cleaner(str(data_tst_pos))
	        Test_data_cleaned.append(data_pos_tst_clean)
	        y_tst_labels.append(1)

	for neg in neg_tst_data:
	#     print(neg)
	    with open (os.path.join(neg_tst_path,neg)) as f:

	        data_tst_neg= f.readlines()
	        Test_data_uncleaned.append(data_tst_neg)
	        
	        data_neg_tst_clean=text_cleaner(str(data_tst_neg))
	        Test_data_cleaned.append(data_neg_tst_clean)
	        y_tst_labels.append(0)


	df_test=pd.DataFrame({"Reviews":Test_data_cleaned,"Labels":y_tst_labels})

	#LOAD THE SAME TOKINIZER FOR TRAIN
	with open('models/tokens.pkl', 'rb') as f:
	   		tokens=pkl.load(f)

	encoded_docs_test = tokens.texts_to_sequences(df_test["Reviews"])

	max_length = 450
	padded_docs_test = pad_sequences(encoded_docs_test,maxlen=max_length, padding='pre')
	

	y_test= np.array(df_test["Labels"])

	# 3. Run prediction on the test data and print the test accuracy

	NLP_evaluate= NLP_model.evaluate(padded_docs_test,y_test)
	print("TESTING ACCURACY : {0:2%}".format(NLP_evaluate[1]))


