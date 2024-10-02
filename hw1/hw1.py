"""This is a sample file for hw1. 
It contains the function that should be submitted,
except all it does is output a random value out of the
possible values that are allowed.
- Dr. Licato"""

# name: Sriram Vujjini
# UID: U59519443

import random
import re


def problem1(NPs, s):
	# lowercasing all the noun phrases so that comparision would be easier
	words = [word.lower() for word in NPs]
	# declaring the set to store the hypernymns 
	hypernyms = set()
	# Two regex patterns are created. The first one represents hearst patterns of type 2 and the second one in the list represents the hearst patterns of type 1.
	# In both of the regexes I have used .join(words) function so that I can make sure that the word being grouped exists in the noun phrase list provided.
	patterns = [r'.*?(' + '|'.join(words) + r'),?\s(?:such as|including)\s(' + '|'.join(words) + r')(?:,\s(' + '|'.join(words) + r'))*(?:(?:,)?\s(?:and|or)\s(' + '|'.join(words) + r'))?', r'(' + '|'.join(words) + r')\s(?:is|was|are)\s(?:(?:a\s|an\s)(?:type of|kind of)\s)?(?:a\s|an\s)?(' + '|'.join(words) + r')']
	# checking seperately for both the patterns if the sentence has a match or not.
	for i in range(len(patterns)):
		# using re.findall() to ignore all the characters that appear before and after the matching sentences. Using re.I to ignore cases for the matching groups.
		matches = re.findall(patterns[i], s, re.I)
		# if matched are found
		if matches:
			# since re.findall() returns a set of matched groups...
			for match in matches:
				# if the pattern being matched is the first one (for the type 2 hearst pattern), then the first match is going to be the parent or the first word appearing in the hearst tuple and teh follwing words in the second part of the tuple.
				if i == 0:
					# parent is the first match.
					parent = match[0]
					# for all the matches following parent
					for j in range(1, len(match)):
						# if the nothing is matched for a specific part of the sentence skip.
						if match[j] == '':
							continue
						# the final hypernymn
						hypernyms.add((parent, match[j].lower()))
				elif i == 1:
					# if the match is for teh second pattern in the list (type 1 hearst pattern), do reverse.
					hypernyms.add((match[1].lower(), match[0].lower()))
		else:
			continue
	return hypernyms


def problem2(s1, s2):
	# adding a # to the beginning of each of the string
	s1 = "#" + s1
	s2 = "#" + s2
	# getting the lengths of the strings
	m = len(s1)
	n = len(s2)
	# initially making each cell in the table equal to 0 to build the table.
	D = [[0 for j in range(m)] for i in range(n)]
	# giving an edit distance of position of the character for each character in the first row and column
	for i in range(n):
		D[i][0] = i
	for j in range(m):
		D[0][j] = j
	for i in range(1, n):
		for j in range(1, m):
			# if the corresponding characters are same, set the sub_op (substitution operation) as the same as the edit distance in the previous cell diagonally.
			if s1[j] == s2[i]:
				sub_op = D[i-1][j-1]
			# else add two to the edit distance in the previous diagonal cell and set it to sub_op
			else:
				sub_op = D[i-1][j-1] + 2

			# the final edit distance for the current cell is going to be the minimum of the deletion, addition and substitution operations which are equal to the arguments provided to the minimum function below.
			D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1, sub_op)

	# returning the Levenshtein edit distance from s1 to s2. It is going to be the last cell in the table.
	return D[n-1][m-1]

