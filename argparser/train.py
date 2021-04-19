from argparse import ArgumentParser

RASA = 'rasa'
SKLEARN = 'sklearn'
ENGINES = [RASA, SKLEARN]
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
    def tfidf(self) -> bool:
        return self._args.tfidf

    @property
    def lsi(self) -> bool:
        return  self._args.lsi

    @property
    def embedding(self) -> bool:
        return self._args.embedding

    @property
    def value_column(self) -> str:
        return self._args.value_column

    @property
    def class_column(self) -> str:
        return self._args.class_column

    def __init__(self) -> None:
        parser = ArgumentParser(description='Train a model.')
        self.set_arguments(parser)
        self._args = parser.parse_args()
        self._args = parser.parse_args()

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
        parser.add_argument('--tfidf', action='store_true', default=False,
                            help=f'Use TF/Idf for feature weighted. Only for {SKLEARN} engine.')
        parser.add_argument('--lsi', action='store_true', default=False,
                            help=f'Use LSI features. Only for {SKLEARN} engine.')
        parser.add_argument('--lda', action='store_true', default=False,
                            help=f'Use LDA features. Only for {SKLEARN} engine.')
        parser.add_argument('--embeddings', action='store_true', default=False,
                            help=f'Use word embeddings for text expansions. Only for {SKLEARN} engine.')
        parser.add_argument('--test', metavar='FILE', type=str,
                            help=f'The corpus used to test the model.')
