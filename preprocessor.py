"""
	This programm is responsible by clear the news content, removing stopwords.
"""

import numpy as np
import pandas as pd
import csv
import re
import nltk

from nltk.stem import RSLPStemmer
from collections import defaultdict

def rem_noise(text):
	# removing noise characters
	text = re.sub('[0-9\[\](){}=.,:;+?/!\*<>_\-§%\$\'\"]', '', text)
	# removing urls
	text = re.sub('(www|http)\S+', '', text)
	return text

"""
	Read the stopwords file and remove them from the news
"""
def cleanning(data):
	print("Removing stopwords and noise characters")
	filepath_sw = 'data/stopwords/stopwords.txt'
	file = open(filepath_sw)
	stop_words = []

	for row in file:
		stop_words.append(row.replace('\n', '').replace(' ', ''))

	for i, body in enumerate(data):
		body = rem_noise(body)
		words = body.split()
		for s in stop_words:
			for j, w in enumerate(words):
				if (w.lower() == s):
					words.pop(j)
		data[i] = " ".join(words)

	return np.asarray(data)

def stemming(tokens):
	stemmer = RSLPStemmer()
	result = []
	for word in tokens:
		result.append(stemmer.stem(word.lower()))
	return result

"""
	Remove the words that occurs less than 10 times in all the news (rare words not differenciate two news)
"""
def rem_lowfreq(data):
	print("Removing low frequence words")
	dic = defaultdict(int)
	for i in data:
		words = i.split()
		for w in words:
			dic[w] += 1

	for i, body in enumerate(data):
		words = body.split()
		for j, w in enumerate(words):
			if dic[w] < 10:
				words.pop(j)
		words = stemming(words)
		data[i] = " ".join(words)

	return np.asarray(data)


rfilepath = 'data/br-fake-news/brnews.txt'
wfilepath = 'data/br-fake-news/preprocessed/brnews.csv'

df = pd.read_csv(rfilepath, sep="\t", encoding='utf-8', names=['Body', 'Label'])
new_df = pd.DataFrame({'Body': cleanning(df['Body'].values), 'Label': df['Label'].values})
new_df = pd.DataFrame({'Body': rem_lowfreq(df['Body'].values), 'Label': df['Label'].values})
new_df.to_csv(wfilepath, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, index=False)