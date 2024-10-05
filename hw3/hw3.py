"""This is a sample file for hw3. 
It contains the functions that should be submitted,
except all it does is output a random value.
- Dr. Licato"""

import random
import gensim.models.keyedvectors as word2vec
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine as cosDist
import numpy as np
import json

model = None

def setModel(Model):
	global model
	model = Model

def findPlagiarism(sentences, target):
	return random.randint(0, len(sentences)-1)

def classifySubreddit_train(file):
	pass

def classifySubreddit_test(text):
	return "subredditName"