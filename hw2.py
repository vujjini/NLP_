"""This is a sample file for hw2. 
It contains the function that should be submitted,
except all it does is output a random value.
- Dr. Licato"""

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import sklearn.naive_bayes
import random

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def lemmatize(token, tag):
	if tag.lower() in ['n', 'v', 'a', 'r']:
		return lemmatizer.lemmatize(token, tag.lower())
	else:
		return token
	
class vectorizer_:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.ngram_counts = {}

    def transform(self, documents):
        # Get n-gram counts for the input documents
        X = self.vectorizer.fit_transform(documents)  # For new data, use transform
        ngrams = self.vectorizer.get_feature_names_out()
        counts = X.toarray().sum(axis=0)  # Sum the counts across documents
        self.ngram_counts = {i: int(j) for i, j in dict(zip(ngrams, counts)).items()} # TODO: change the dict(zip(ngrams, counts))
		

	
V = vectorizer_(CountVectorizer(ngram_range=(2, 3)))

def calcNGrams_train(trainFile):

	stemmer = PorterStemmer()


	documents = []
	stemmed = []
	with open(trainFile, 'r', encoding='utf-8') as file:
		for line in file:
			documents.append(line.strip())

	tagged = []
	lemmatized_doc = []
	# need to convert each line in documents to sentences using sent_tokenize
	for document in documents:
		sentences = sent_tokenize(document)
		for sentence in sentences:
			tokens = word_tokenize(sentence)
			lemmatized_sentence = [lemmatize(token.lower(), tag) for token, tag in pos_tag(tokens)]
			# lemmatized_sentence = ['<s>'] + lemmatized_sentence + ['</s>']
			lemmatized_doc.append(" ".join(lemmatized_sentence))
			
			# print(tokens)
			# TODO: Remove stemming if there is no use for it
			for token in tokens:
				stemmed.append(stemmer.stem(token))
	# for document in documents:
	# 	sentences = sent_tokenize(document)
	# 	for sentence in sentences:
	# 		tokens = word_tokenize(sentence)
	# 		l = pos_tag(tokens)
	# 		for i in l:
	# 			tagged.append((i[0], get_wordnet_pos(i[1])))
			
	# 		# print(tokens)
	# 		for token in tokens:
	# 			stemmed.append(stemmer.stem(token))
	# tagged = [[pos_tag(word_tokenize(sentence)) for sentence in sent_tokenize(document)] for document in documents]
	# lemmatized = [lemmatize(word, tag) for word, tag in tagged]
				
	# print(tagged[1])
	# print(lemmatized[1])
	# print(stemmed[1])
	# print(tagged[1])
	# print(lemmatizer.lemmatize('commented', pos=get_wordnet_pos('VBD')))

	# TODO: change the way the doc is lemmatized. instead of lemmatizing the doc as a whole, lemmatize each sentence seperately 
	# in the loop before and add em to lemmatized doc. Nvm for TODO. instead maybe remove punctuations.
	# TODO: need to add a sentence start token (<s>) and a sentence end token (</s>) for each sentence.
	# lemmatized_doc = [" ".join(lemmatized)]
	
	# print(lemmatized_doc)
	V.transform(documents)


	"""
	trainFile: a text file, where each line is arbitratry human-generated text
	Outputs n-grams (n=2, or n=3, your choice). Must run in under 120 seconds
	"""
	pass #don't return anything from this function!

def calcNGrams_test(sentences):

	# odds = []

	vocab = V.vectorizer.vocabulary_
	print(V.ngram_counts)
	# for sentence in sentences:
	# 	s = [sentence]
	# 	f = V.transform(s)
	# 	print(f)
	# tested_frequencies = V.transform(sentences)
	# key = next((key for key, val in vocab.items() if val == 302), None)
	# print(key)
	# print(tested_frequencies)

	# TODO: we calulating probability of the sentence. we calculate each part of the product by dividing using the formula
	# 
	"""
	sentences: A list of single sentences. All but one of these consists of entirely random words.
	Return an integer i, which is the (zero-indexed) index of the sentence in sentences which is non-random.
	"""
	return random.randint(0, len(sentences)-1)

def calcSentiment_train(trainFile):
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
	return random.choice([True, False])

calcNGrams_train('problem1_trainingFile.txt')

# dev sentence
calcNGrams_test(["We have heard her clear, bird-like voice mingling with the scarlet symbol, and the most agreeable of his.", 
	"poetry unthrifty ignominy devoting passages ceases strewn wished concerned progenitors arrangement borne sergeants express contains flowers medicine vain mahogany social",
	"I have ever cherished, and would be convulsed with rage of grief and sob out her love for her."])
