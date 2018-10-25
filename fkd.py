import pandas as pd 											#used to manipulate files
from sklearn.feature_extraction.text import CountVectorizer 	# used to vectorize the sentences
from sklearn.model_selection import train_test_split 			# used to split the training and test set
from sklearn.linear_model import LogisticRegression 			# used to logistic regression

"""
	This method is used to train and test a set of sentences using logistic regression
	The method is separated in three steps
		1 - Divide the data in training set and test set (25% for test)
		2 - Vectorize the sentences
		3 - Training and predict
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
lr_classifier(df)
