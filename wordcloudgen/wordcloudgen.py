from argparse import ArgumentParser
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def get_args():
    parser = ArgumentParser(description='Simple wordcloud generator')
    parser.add_argument(
        '-f', '--inputFile',
        help='Input file',
        required=True
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    input_file = args.inputFile
    with open(input_file, 'r') as f:
        text = f.read()
        wordcloud = WordCloud(background_color='white', max_font_size=40).generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
