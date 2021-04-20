from argparse import ArgumentParser

RASA, SKLEARN = 'rasa', 'sklearn'
ENGINES = [RASA, SKLEARN]
SVM, KMEANS = 'svm', 'kmeans'
CLASSIFIERS = [SVM, KMEANS]
CLASS_ENGINES = {
    RASA: {},
    SKLEARN: {c for c in CLASSIFIERS}
}
DEF_TEXT_COLUMN = 'text'
DEF_CLASS_COLUMN = 'cls'


class TrainArgParser(object):
    @property
    def train(self) -> str:
        return self._args.train

    @property
    def test(self) -> str:
        return self._args.test

    @property
    def engine(self) -> str:
        return self._args.engine

    @property
    def classifier(self) -> str:
        return self._args.classifier

    @property
    def tfidf(self) -> bool:
        return self._args.tfidf

    @property
    def lsi(self) -> bool:
        return  self._args.lsi

    @property
    def lda(self) -> bool:
        return  self._args.lsi

    @property
    def embeddings(self) -> bool:
        return self._args.embeddinga

    @property
    def embedding_num(self) -> int:
        return self._args.embedding_num

    @property
    def embedding_threshold(self) -> int:
        return self._args.embedding_threshold

    @property
    def value_column(self) -> str:
        return self._args.value_column

    @property
    def class_column(self) -> str:
        return self._args.class_column

    @property
    def output(self) -> str:
        return self._args.output

    def __init__(self) -> None:
        self._parser = ArgumentParser(description='Train a model.')
        self.set_arguments(self._parser)
        self._args = self._parser.parse_args()

    @staticmethod
    def set_arguments(parser: ArgumentParser) -> None:
        parser.add_argument('-t', '--train', type=str, metavar='FILE', nargs='+',
                            help='CSV or Excel files with the texts and classes of the samples to train')
        parser.add_argument('-l', '--lang', metavar='LANGUAGE', type=str, default='en', help='The corpus language.')
        parser.add_argument('-v', '--value_column', metavar='COL', type=str, default='text',
                            help='The column name for the sample text in the CSV file.')
        parser.add_argument('-c', '--class_column', metavar='COL', type=str, default='cls',
                            help='The column name for the sample class in the CSV file.')
        parser.add_argument('-e', '--engine', type=str, metavar='ENGINE', default=SKLEARN,
                            help=f'The machine learning engine to use to train. Available engines: {ENGINES}. '
                                 f'By default, {SKLEARN}')
        parser.add_argument('--classifier', type=str, metavar='CLASSIFIER', default=SVM,
                            help=f'The suitable classifier to use for the specific engine to use to train. '
                                 f'Available classifiers: {CLASSIFIERS}. By default, {SVM}')
        parser.add_argument('--tfidf', action='store_true', default=False,
                            help=f'Use TF/Idf for feature weighted. Only for {SKLEARN} engine.')
        parser.add_argument('--lsi', action='store_true', default=False,
                            help=f'Use LSI features. Only for {SKLEARN} engine.')
        parser.add_argument('--lda', action='store_true', default=False,
                            help=f'Use LDA features. Only for {SKLEARN} engine.')
        parser.add_argument('--embeddings', action='store_true', default=False,
                            help=f'Use word embeddings for text expansions. Only for {SKLEARN} engine.')
        parser.add_argument('--embedding_num', metavar='NUM', type=int, default=10,
                            help=f'If the embeddings are used, the number of embeddings. By default, 10.')
        parser.add_argument('--embedding_threshold', metavar='VALUE', type=int, default=0.85,
                            help=f'If embeddings are used, the threshold of accepted embeddings. By default, 0.85.')
        parser.add_argument('--output', metavar='FILE', type=str,
                            help=f'The file with the trained model. For sklearn engine it would be a pickle file and '
                                 f'for rasa it would be a tar.gz file.')
