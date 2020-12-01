from argparse import ArgumentParser


class LifeArgParser(object):
    @property
    def lang(self) -> str:
        return self.__args.lang

    @property
    def data(self) -> str:
        return self.__args.data

    @property
    def embeddings(self) -> int:
        return self.__args.embeddings

    @property
    def embedding_threshold(self) -> float:
        return self.__args.emb_threshold

    @property
    def evaluate(self) -> bool:
        return self.__args.evaluate

    @property
    def eval_file(self) -> str:
        return self.__args.eval_file

    @property
    def boostrapping(self) -> bool:
        return self.__args.boostrapping

    @property
    def cross_folder(self) -> int:
        return self.__args.cross

    @property
    def no_risk(self) -> float:
        return self.__args.no_risk

    @property
    def risk(self) -> float:
        return self.__args.risk

    @property
    def file(self) -> str:
        return self.__args.file

    def __init__(self):
        parser = ArgumentParser(description='Train, evaluate and increase the corpus using boostraping techniques')
        parser.add_argument('-l', '--lang', metavar='LANG', type=str, default='en', help='The corpus language.')
        parser.add_argument('-d', '--data', metavar='DIR', type=str, default='data',
                            help='The folder where the data is stored.')
        parser.add_argument('-en', '--embeddings', metavar='NUM', type=int, default=1000,
                            help='The maximum number of embeddings neighbors to retrieve for each term.')
        parser.add_argument('-et', '--emb_threshold', metavar='VALUE', type=float, default=0.95,
                            help='The threshold to include embedding neighbors.')
        parser.add_argument('-e', '--evaluate', default=False, action='store_true',
                            help='The threshold to include embedding neighbors.')
        parser.add_argument('-ef', '--eval_file', metavar='FILE', type=str,
                            help='The output file to store the evaluation result.')
        parser.add_argument('-b', '--boostrapping', default=True, action='store_false',
                            help='The threshold to include embedding neighbors.')
        parser.add_argument('-c', '--cross', metavar='NUM', type=int, default=10, help='The cross validation folder.')
        parser.add_argument('-n', '--no_risk', metavar='VALUE', type=float, default=0.2,
                            help='The confidence of no risk samples in the boostrapping.')
        parser.add_argument('-r', '--risk', metavar='VALUE', type=float, default=0.2,
                            help='The confidence of risk samples in the boostrapping.')
        parser.add_argument('-f', '--file', metavar='FILE', type=str, required=True,
                            help='The output file to store the boostrapped corpus.')
        self.__args = parser.parse_args()
