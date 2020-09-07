# import required packages
import os
import re
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import nltk
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Input, Dense, Embedding, Dropout, BatchNormalization, Activation, Bidirectional,Flatten, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pickle as pkl


# PREPROCESSING DONE


stop_words = set(stopwords.words('english')) 

# REMOVING STOPWORDS,SPECIAL CHARACTERS
def text_cleaner(text):
    processeing_text = text.lower()
    processeing_text = re.sub(r'\([^)]*\)', '', processeing_text)
    processeing_text = re.sub('"','', processeing_text)

    processeing_text = re.sub(r"'s\b","",processeing_text)
    processeing_text = re.sub("[^a-zA-Z]", " ", processeing_text) 

    tokens = [word for word in processeing_text.split() if not word in stop_words]
    return ( " ".join(tokens)).strip()


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__": 
	# 1. load your training data

	Train_data_uncleaned=[]
	Train_data_cleaned=[]
	y_labels=[]
	pos_path="data/aclImdb/train/pos"
	neg_path= "data/aclImdb/train/neg"
	pos_data=sorted(os.listdir(pos_path))
	neg_data=sorted(os.listdir(neg_path))
	for pos in pos_data:
	    with open (os.path.join(pos_path,pos)) as f:

	        data_pos= f.readlines()
	        Train_data_uncleaned.append(data_pos)
	        
	        data_pos_clean=text_cleaner(str(data_pos))
	        Train_data_cleaned.append(data_pos_clean)
	        y_labels.append(1)

	for neg in neg_data:
	    with open (os.path.join(neg_path,neg)) as f:

	        data_neg= f.readlines()
	        Train_data_uncleaned.append(data_neg)
	        
	        data_neg_clean=text_cleaner(str(data_neg))
	        Train_data_cleaned.append(data_neg_clean)
	        y_labels.append(0)



	df = pd.DataFrame({"Reviews":Train_data_cleaned,"Labels":y_labels})


	# 2. Train your network

	# Training the word2vec model

	word_sentences=[nltk.word_tokenize(sentence) for sentence in df["Reviews"]]
	W2v= Word2Vec(word_sentences, size=400, window=10, min_count=10)
	embedding_vectors=W2v.wv.vectors

	# USING KERAS PREPROCESSING FOr CREATING INTO VECTORS
	tokens=Tokenizer(num_words=embedding_vectors.shape[0])
	tokens.fit_on_texts(df["Reviews"])
	pkl.dump(tokens,open("models/tokens.pkl", "wb"))

	encoded_docs_train = tokens.texts_to_sequences(df["Reviews"])
	max_length = 450
	padded_docs = pad_sequences(encoded_docs_train,maxlen=max_length, padding='pre')


	y_train= np.array(df["Labels"])

	embedding_layer = Embedding(input_dim =embedding_vectors.shape[0], output_dim = embedding_vectors.shape[1],weights= [embedding_vectors],trainable=True,input_length=450)

	# MODEL

	we_model_new=Sequential()
	we_model_new.add(embedding_layer)
	we_model_new.add(LSTM(16,return_sequences=True))
	we_model_new.add(LSTM(4))
	we_model_new.add(Dense(1,activation="sigmoid"))
	we_model_new.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	print(we_model_new.summary())
	# callback = ModelCheckpoint("wemodel_new_64_1_lay_callback.hdf5", monitor='val_accuracy', save_best_only=True)
	we_model_new.fit(padded_docs, y_train, validation_split=0.2,epochs=5,callbacks=[callback])







	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy

	# 3. Save your model
	we_model_new.save("models/20859891_NLP_model.hdf5")