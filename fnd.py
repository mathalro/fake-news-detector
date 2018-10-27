"""
	This program is a news classifier. Given a news dataset, 
	the program train a neural network to predict if that 
	news is a fake news or a real news. 
"""

import pandas as pd 											#used to manipulate files
import matplotlib.pyplot as plt
from sklearn import metrics 
from sklearn.feature_extraction.text import CountVectorizer		# used to vectorize the sentences
from sklearn.model_selection import train_test_split			# used to split the training and test set
from sklearn.linear_model import LogisticRegression				# used to logistic regression
from keras.models import Sequential
from keras import layers

def plot_history(history, base):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure()
    #plt.subplot(1, 2, 1)
    #plt.plot(x, acc, 'b', label='Training acc')
    #plt.plot(x, val_acc, 'r', label='Validation acc')
    #plt.title('Training and validation accuracy')
    #plt.legend()
    #plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    #plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training cost - Base '+base)
    plt.legend()
    plt.show()

"""
	This method is used to train and test a set of sentences using logistic regression
	The method is separated in three steps
		1 - Divide the data in training set and test set (25% for test)
		2 - Vectorize the sentences
		3 - Training and test
"""
def lr_classifier(df):
	for source in df['source'].unique():
		# dividing the dataset
		df_yelp = df[df['source'] == source]
		sentences = df_yelp['Body'].values
		y = df_yelp['Label'].values
		sentences_train, sentences_test, y_train, y_test = train_test_split(
			sentences, y, test_size=0.25, random_state=1000)

		# vectorizing sentences
		vectorizer = CountVectorizer()

		for i in range(len(sentences_train)):
			sentences_train[i] = str(sentences_train[i])
		for i in range(len(sentences_test)):
			sentences_test[i] = str(sentences_test[i])

		vectorizer.fit(sentences_train)

		X_train = vectorizer.transform(sentences_train)
		X_test = vectorizer.transform(sentences_test)

		# logistic regression
		classifier = LogisticRegression()
		classifier.fit(X_train, y_train)
		predictions = classifier.predict(X_train)
		f1 = metrics.f1_score(y_train, predictions)
		print(f1)

"""
	This method is used to train and test a set of sentences using neural network
	The method is separated in three steps
		1 - Divide the data in training set and test set (25% for test)
		2 - Vectorize the sentences
		3 - Training and test
"""
def ann_classifier(df):
	for source in df['source'].unique():
		# dividing the dataset
		df_yelp = df[df['source'] == source]
		sentences = df_yelp['Body'].values
		y = df_yelp['Label'].values
		sentences_train, sentences_test, y_train, y_test = train_test_split(
			sentences, y, test_size=0.25, random_state=1000)

		# vectorizing sentences
		vectorizer = CountVectorizer()

		for i in range(len(sentences_train)):
			sentences_train[i] = str(sentences_train[i])
		for i in range(len(sentences_test)):
			sentences_test[i] = str(sentences_test[i])

		if source == 'george':
			for i in range(len(y_train)):
				if y_train[i] == "REAL":
					y_train[i] = 0
				else:
					y_train[i] = 1
			for i in range(len(y_test)):
				if y_test[i] == "REAL":
					y_test[i] = 0
				else:
					y_test[i] = 1

		vectorizer.fit(sentences_train)

		X_train = vectorizer.transform(sentences_train)
		X_test = vectorizer.transform(sentences_test)


		input_dim = X_train.shape[1]
		model = Sequential()
		model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
		model.add(layers.Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.summary()
		history = model.fit(X_train, y_train, epochs=20, verbose=False,
							validation_data=(X_test, y_test), batch_size = 10)
		loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
		print("Training Accuracy: {:.4f}".format(accuracy))
		loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
		print("Test Accuracy: {:.4f}".format(accuracy))
		plot_history(history, source)


filepath_dict = {'kaggle': 'data/fake-news/data1.csv', 'george': 'data/fake-news/data2.csv'}

df_list = []
# read all the datasets and save in a list. The columns are (sentence, label, source)
for source, filepath in filepath_dict.items():
	df = pd.read_csv(filepath, sep=',')
	df['source'] = source
	df_list.append(df)

# concat all the read dataset
df = pd.concat(df_list)
ann_classifier(df)
