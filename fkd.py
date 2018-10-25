import pandas as pd 											#used to manipulate files
from sklearn.feature_extraction.text import CountVectorizer		# used to vectorize the sentences
from sklearn.model_selection import train_test_split			# used to split the training and test set
from sklearn.linear_model import LogisticRegression				# used to logistic regression
from keras.models import Sequential
from keras import layers

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
		sentences = df_yelp['sentence'].values
		y = df_yelp['label'].values
		sentences_train, sentences_test, y_train, y_test = train_test_split(
			sentences, y, test_size=0.25, random_state=1000)

		# vectorizing sentences
		vectorizer = CountVectorizer()
		vectorizer.fit(sentences_train)
		X_train = vectorizer.transform(sentences_train)
		X_test = vectorizer.transform(sentences_test)

		# logistic regression
		classifier = LogisticRegression()
		classifier.fit(X_train, y_train)
		score = classifier.score(X_test, y_test)
		print("Accuracy for {} data: {:.4f}".format(source, score))

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
		sentences = df_yelp['sentence'].values
		y = df_yelp['label'].values
		sentences_train, sentences_test, y_train, y_test = train_test_split(
			sentences, y, test_size=0.25, random_state=1000)

		# vectorizing sentences
		vectorizer = CountVectorizer()
		vectorizer.fit(sentences_train)
		X_train = vectorizer.transform(sentences_train)
		X_test = vectorizer.transform(sentences_test)

		input_dim = X_train.shape[1]
		model = Sequential()
		model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
		model.add(layers.Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.summary()
		history = model.fit(X_train, y_train, epochs=100, verbose=False,
							validation_data=(X_test, y_test), batch_size = 10)
		loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
		print("Training Accuracy: {:.4f}".format(accuracy))
		loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
		print("Test Accuracy: {:.4f}".format(accuracy))


filepath_dict = {'yelp': 'data/examples/yelp_labelled.txt',
				 'amazon': 'data/examples/amazon_cells_labelled.txt',
				 'imdb': 'data/examples/imdb_labelled.txt'}

df_list = []
# read all the datasets and save in a list. The columns are (sentence, label, source)
for source, filepath in filepath_dict.items():
	df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
	df['source'] = source
	df_list.append(df)

# concat all the read dataset
df = pd.concat(df_list)
#lr_classifier(df)
ann_classifier(df)
