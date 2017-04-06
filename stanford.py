from __future__ import division
import json
from subprocess import call
import sys

inputFile = sys.argv[1]
cleanFile = inputFile + '.clean'

with open(cleanFile, 'w') as f:
	pass

with open(inputFile, 'r') as f:
	for blob in f:
		review = json.loads(blob)
		text = review['reviewText']
		with open(cleanFile, 'a') as cf:
			cf.write(text + '\n')

call([
	'java',
	'-cp',
	'stanford-corenlp-full-2016-10-31/*',
	'-mx5g',
	'edu.stanford.nlp.sentiment.SentimentPipeline',
	'-file',
	cleanFile
])

# count = 0
# correct = 0

# scores = {
#     'Very negative': 1,
#     'Negative': 2,
#     'Neutral': 3,
#     'Positive': 4
# }

# with open(cleanFile, 'r') as f:
# 	for blob in f:
# 		review = json.loads(blob)
# 		text = review['reviewText']
# 		expectedScore = review['overall']
# 		print text, expectedScore
# 		actualScoreLabel, err = pipe.communicate(text)
# 		print actualScoreLabel, err
# 		if scores[actualScoreLabel.strip()] == expectedScore:
# 			correct += 1
# 		count += 1

# print correct / count