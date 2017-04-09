from argparse import ArgumentParser
from pycorenlp import StanfordCoreNLP


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
    nlp = StanfordCoreNLP('http://localhost:9000')
    count = 0
    correct = 0
    with open(inputFile, 'r') as f:
        for review in f:
            print(review)
            res = nlp.annotate(
                review,
                properties={
                    'annotators': 'sentiment',
                    'outputFormat': 'json',
                    'timeout': 100000,
                }
            )
            result = res['sentences']
            numSentences = len(result)
            if numSentences:
                totalScore = sum([int(s['sentimentValue']) for s in result])
                averageScore = totalScore / numSentences
                print(averageScore)
                if 'neg' in inputFile:
                    if averageScore < 1.5:
                        correct += 1
                elif 'pos' in inputFile:
                    if averageScore > 2.5:
                        correct += 1
                count += 1
        accuracy = correct / count
        print('Accuracy: {:.2f}'.format(correct / count))
