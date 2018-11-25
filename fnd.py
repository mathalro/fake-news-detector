"""	
This program is a news classifier. Given a news dataset, 
	the program train a neural network to predict if that 
	news is a fake news or a real news. 
"""

import pandas as pd 											#used to manipulate files
import matplotlib.pyplot as plt
import glob
import chardet
import csv
import io
import random
import numpy as np

from sklearn import metrics 
from sklearn.feature_extraction.text import CountVectorizer		# used to vectorize the sentences
from sklearn.model_selection import train_test_split			# used to split the training and test set
from sklearn.linear_model import LogisticRegression				# used to logistic regression
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


"""
	Show Confusion Matrix, Accuracy, Precision, Recal and F1 score of a model
"""
def print_metrics(model, X, y, lr=False):
	y_pred = []
	y_true = []
	pred = model.predict(X)

	for i in range(len(pred)):
		if lr:
			y_pred.append(round(pred[i]))
		else:
			y_pred.append(round(pred[i][0]))
		y_true.append(y[i])

	print("Confusion matrix:")
	print(pd.DataFrame(metrics.confusion_matrix(y_true, y_pred, labels=[1, 0]), index=['true:fake', 'true:real'], columns=['pred:fake', 'pred:real']))
	print("\nAccuracy:", metrics.accuracy_score(y_true, y_pred))
	print("Precision:", metrics.precision_score(y_true, y_pred))
	print("Recall:", metrics.recall_score(y_true, y_pred))
	print("F1 score:", metrics.f1_score(y_true, y_pred))
	print("\n==================================================================================\n\n")


"""
	Plot history of loss and accuracy of a model CV and Train
"""
def plot_history(history, title):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure()
    plt.title(title)
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


"""
	Create a divided in Train and Test dataset
"""
def train_test_ds(df):
	for source in df['source'].unique():
		# dividing the dataset
		df_yelp = df[df['source'] == source]
		sentences = df_yelp['Body'].values
		y = df_yelp['Label'].values
		X_train, X_test, y_train, y_test = train_test_split(
			sentences, y, test_size=0.30, random_state=999)

		return [X_train, X_test, y_train, y_test]


"""
	Logistic Regression Classifier
"""
def lr_classifier(X_train, X_test, y_train, y_test):

	vectorizer = CountVectorizer()
	vectorizer.fit(X_train)

	X_train = vectorizer.transform(X_train)
	X_test = vectorizer.transform(X_test)

	# logistic regression
	model = LogisticRegression()
	model.fit(X_train, y_train)
	print("\n\nLOGISTIC REGRESSION RESULTS\n")
	print_metrics(model, X_test, y_test, lr=True)

"""
	Simple Neural Network Classifier
"""
def ann_classifier(X_train, X_test, y_train, y_test):
	vectorizer = CountVectorizer()
	vectorizer.fit(X_train)

	X_train = vectorizer.transform(X_train)
	X_test = vectorizer.transform(X_test)

	input_dim = X_train.shape[1]
	model = Sequential()
	model.add(layers.Dense(25, input_dim=input_dim, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	history = model.fit(X_train, y_train, epochs=3, verbose=True,
						validation_data=(X_test, y_test), batch_size = 10)
	loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
	print("\n\nRNA RESULTS\n")
	print_metrics(model, X_test, y_test)
	#plot_history(history, "Common Neural Network")


"""
	Convolutional neral network classifier
"""
def cnn_classifier(X_train, X_test, y_train, y_test):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(X_train)

	X_train = tokenizer.texts_to_sequences(X_train)
	X_test = tokenizer.texts_to_sequences(X_test)

	vocab_size = len(tokenizer.word_index) + 1
	maxlen = 300
	embedding_dim = 20

	X_train = pad_sequences(X_train, padding='post', truncating='post', maxlen=maxlen)
	X_test = pad_sequences(X_test, padding='post', truncating='post', maxlen=maxlen)

	model = Sequential()
	model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
	model.add(layers.Conv1D(10, kernel_size=5, activation='relu'))
	model.add(layers.MaxPool1D())
	model.add(layers.Flatten())
	model.add(layers.Dense(10, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	model.summary()
	history = model.fit(X_train, y_train, epochs=3, verbose=True, validation_data=(X_test, y_test), batch_size=10)
	print("\n\nCNN RESULTS\n")
	print_metrics(model, X_test, y_test)
	#plot_history(history, "Common Neural Network")


#filepath_dict = {'kaggle': 'data/fake-news/data1.csv', 'george': 'data/fake-news/data2.csv'}
filepath = 'data/br-fake-news/fake/brfake.txt'

filepath_dict1 = {'brnews': 'data/br-fake-news/preprocessed/brnews.csv'}
filepath_dict2 = {'brnews': 'data/br-fake-news/brnews.txt'}

df_list = []
stop_words = []

for source, filepath in filepath_dict2.items():
	df = pd.read_csv(filepath, sep="\t", encoding='utf-8')
	df['source'] = source
	df_list.append(df)

# concat all the read dataset
df = pd.concat(df_list)
#print(df)
data = train_test_ds(df)

lr_classifier(data[0], data[1], data[2], data[3])
ann_classifier(data[0], data[1], data[2], data[3])
cnn_classifier(data[0], data[1], data[2], data[3])