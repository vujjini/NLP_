"""This is a sample file for hw2. 
It contains the function that should be submitted,
except all it does is output a random value.
- Dr. Licato"""

# Name: Sriram Vujjini
# UID: U59519443

import string # importing python string for removing punctuations
# commenting all the downloads
# import nltk
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer   # importing lemmatizer
from sklearn.feature_extraction.text import CountVectorizer  # importing CountVectorizer
from sklearn.naive_bayes import MultinomialNB # importing Multinomial NB model
import json # importing json

lemmatizer = WordNetLemmatizer() #initializing the lemmatizer object as lemmatizer()
	
# Creating a global vectorizer_ class so that an object declared using the CountVectorizer() and its features such as the n gram counts, etc in a function can be used
# seamlessly in other functions.
class vectorizer_:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer # initialising the vectorizer object with self.vectorizer(), self.ngram_counts and self.ngrams as features 
        self.ngram_counts = {}
        self.ngrams = []

    def transform(self, documents):
        # Get n-gram counts for the input documents
        X = self.vectorizer.fit_transform(documents)  # using transform
        self.ngrams = self.vectorizer.get_feature_names_out() # getting ngram names
        counts = X.toarray().sum(axis=0)  # Sum the counts across documents
        self.ngram_counts = {i: int(j) for i, j in dict(zip(self.ngrams, counts)).items()} # getting counts for each each n gram
		

	
V = vectorizer_(CountVectorizer(ngram_range=(1, 2), stop_words="english"))  # initialisng global vectorizer object with unigrams and bigrams and stopword detection for the english language words.

def calcNGrams_train(trainFile):

	with open(trainFile, 'r', encoding='utf-8') as file:
		document = file.read()  # getting the data
	
	document = document.translate(str.maketrans('', '', string.punctuation))  # using the python str library to remove punctuation
	document = document.split() # converting document into a list

	lemmatized_words = [lemmatizer.lemmatize(token) for token in document] # lemmatizing all the words in document
	lemmatized_doc = " ".join(lemmatized_words) # combining all of them into a single document string.
	V.transform([lemmatized_doc]) # using V.transform() to fit the data into vectorizer V.

	"""
	trainFile: a text file, where each line is arbitratry human-generated text
	Outputs n-grams (n=2, or n=3, your choice). Must run in under 120 seconds
	"""
	pass #don't return anything from this function!

def calcNGrams_test(sentences):

	test_vectorizer = vectorizer_(CountVectorizer(ngram_range=(2, 2), stop_words="english")) # using a test_vectorizer to retrieve n grams and their respective counts from the test sentences.

	probabilities = []  # to be used at end when finding minimum of probabilities
	for sentence in sentences:
		sentence = sentence.translate(str.maketrans('', '', string.punctuation)) # using the python str library to remove punctuation
		sentence = sentence.split() # converting document into a list
		lemmatized_doc = " ".join([lemmatizer.lemmatize(token) for token in sentence]) # lemmatizing the test sentences
		test_vectorizer.transform([lemmatized_doc]) # fitting the test data in the test vectorizer
		probability = 0
		for i in test_vectorizer.ngrams:  # summing up probabilities of all n grams (using formula from slides) of the test sentence. 
			if i in V.ngram_counts:
				probability+=((V.ngram_counts[i])/(V.ngram_counts[" ".join(i.split()[:2])]))
			else:
				probability+=0
		probabilities.append(probability) # appending the summed up probability to the probabilities list
	return(probabilities.index(min(probabilities))) # finally returning the minimum of the probabilities of test sentences.
	"""
	sentences: A list of single sentences. All but one of these consists of entirely random words.
	Return an integer i, which is the (zero-indexed) index of the sentence in sentences which is non-random.
	"""

# defining a preprocess function to using negations to words in a sentences after encountering a word containing "n't" at the end.
def preprocess(review):
    review = review.translate(str.maketrans('', '', string.punctuation))  # removing punctuation
    sentences = review.split('.')  # converting into list of sentences.
    for i in range(len(sentences)):
        words = sentences[i].split()  # split sentence into words
        for j in range(len(words)):
            if "n't" in words[j]:  # check if word contains n't
                # apply NOT_ to all subsequent words
                words[j+1:] = ['NOT_' + word for word in words[j+1:]]
                break  # stop once we find n't and apply to the rest
        sentences[i] = ' '.join(words)  # join words back into a sentence
    return '. '.join(sentences)  # join sentences back into a review

sentiment_vectorizer = CountVectorizer(ngram_range=(1,2))  # defining a CountVectorizer to get unigrams and bigrams from the reviews. Using bigrams as well to improve precisiona and accuracy
sent_model = MultinomialNB() # defining a global MultinomalNB model to use the same instance of it in both the train and test functions.
def calcSentiment_train(trainFile):

	labels = [] # labels is the list of sentiments
	lemmatized_reviews = [] # list of lemmatized reviews
	with open(trainFile, 'r', encoding = 'utf-8') as F:
		# retrieving data
		for line in F:
			line = line.strip()
			if line:
				data = json.loads(line)
				review = data['review'].translate(str.maketrans('', '', string.punctuation)) # removing punctuations
				review = review.split() # splitting into a list of words
				review = " ".join([lemmatizer.lemmatize(token) for token in review]) # lemmatizing each word and converting the review back into a string.
				review = preprocess(review) # using the preprocess() function defined above to add _NOT to words after words containing "n't" in a sentence. 
				lemmatized_reviews.append(review) # appending the preprocessed review to the main list
				if data['sentiment']:
					labels.append(1) # if True append 1
				else:
					labels.append(0) # else append 0
	x = sentiment_vectorizer.fit_transform(lemmatized_reviews) # using fit_transform to get x labels

	sent_model.fit(x, labels) # training the model

	"""
	trainFile: A jsonlist file, where each line is a json object. Each object contains:
		"review": A string which is the review of a movie
		"sentiment": A Boolean value, True if it was a positive review, False if it was a negative review.
	"""
	pass #don't return anything from this function!

def calcSentiment_test(review):
	"""
	review: A string which is a review of a movie
	Return a boolean which is the predicted sentiment of the review.
	Must run in under 120 seconds, and must use Naive Bayes
	"""	
	review = review.translate(str.maketrans('', '', string.punctuation)) # removing punctuations
	review = review.split() # splitting into a list of words
	review = " ".join([lemmatizer.lemmatize(token) for token in review]) # lemmatizing each word and converting the review back into a string.
	review = preprocess(review) # using the preprocess() function defined above to add _NOT to words after words containing "n't" in a sentence. 
	test_review = sentiment_vectorizer.transform([review]) # using transform on features to be used for the model
	result = sent_model.predict(test_review) # using .predict() to get the result
	# final answer
	if result == 1:
		return True
	elif result == 0:
		return False
	