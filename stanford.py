from __future__ import division
from argparse import ArgumentParser
import json
from subprocess import Popen, PIPE
import sys


def get_args():
	parser = ArgumentParser(description='Stanford sentiment analysis wrapper')
	parser.add_argument(
		'-f', '--inputFile',
		help='Input file',
		required=True
	)
	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()
	inputFile = args.inputFile

	count = 0
	correct = 0

	scores = {
	    'Very negative': 1,
	    'Negative': 2,
	    'Neutral': 3,
	    'Positive': 4,
	    'Very positive': 5
	}

	with open(inputFile, 'r') as f:
		for blob in f:
			if blob:
				review = json.loads(blob)
				text = review['reviewText']
				sentences = text.split('.')
				expectedScore = review['overall']
				print text, expectedScore
				sentenceScores = []
				p = Popen([
					'java',
					'-cp',
					'stanford-corenlp-full-2016-10-31/*',
					'-mx5g',
					'edu.stanford.nlp.sentiment.SentimentPipeline',
					'-stdin'
				], stdin=PIPE, stdout=PIPE, stderr=PIPE)
				sentenceScores, err = p.communicate('\n'.join(sentences) + '\n')
				sentenceScores = sentenceScores.split('\n')
				sentenceScores = [scores[score.strip()] for score in sentenceScores if score]
				averageScore = sum(sentenceScores) / len(sentenceScores)
				print averageScore
				if abs(averageScore - expectedScore) < 2:
					correct += 1
				count += 1
		print correct / count