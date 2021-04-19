from argparse import ArgumentParser
from sys import stdout

from filedatasource import CsvWriter

from corptrans.corpus.corpus import ArrayCorpus
from corptrans.corpus.csv import CsvCorpus
from corptrans.transformers import ClassReduction
from corptrans.transformers.csv import CsvTransformer
from sklearn_train import SVM, CLASSIFIERS, train, corpus2matrix


class ShowErrorsArgParser(object):
    @property
    def corpus(self) -> str:
        return self._args.corpus

    @property
    def test(self) -> str:
        return self._args.test[0]

    @property
    def method(self) -> str:
        return self._args.method

    @property
    def output(self) -> str:
        return self._args.output

    def __init__(self) -> None:
        parser = ArgumentParser(description='Create a model from a train corpus an evaluate it with other test corpus.')
        self.set_arguments(parser)
        self._args = parser.parse_args()

    @staticmethod
    def set_arguments(parser: ArgumentParser) -> None:
        parser.add_argument('-c', '--corpus', metavar='TRAIN_FILE', type=str, required=True,
                            help='The path to CSV corpus file.')
        parser.add_argument('test', metavar='TEST_FILE', type=str, nargs=1,
                            help='The path to CSV test file.')
        parser.add_argument('-m', '--method', metavar='METHOD', type=str, default=SVM, choices=CLASSIFIERS,
                            help='The path to CSV test file.')
        parser.add_argument('-o', '--output', metavar='FILE', type=str,
                            help='The output file to store the result. '
                                 'If it is not given then the standard output is used.')


def main() -> None:
    args = ShowErrorsArgParser()
    ml, corpus = train(args.corpus, args.method)
    with CsvCorpus(args.test) as test_corpus:
        test_corpus.set_transformers(corpus.transformers)
        test_corpus = ArrayCorpus(test_corpus)
        test_corpus._metadata = corpus.metadata
        X = corpus2matrix(test_corpus)
        y = test_corpus.classes()
        y_pred = [ml.predict(X[i])[0] for i in range(X.shape[0])]
    with CsvCorpus(args.test) as test_corpus:
        test_corpus.add_transformer(CsvTransformer('text', 'cls'))
        test_corpus.add_transformer(ClassReduction({0: ['No risk'], 1: ['Possible', 'Risk', 'Urgent', 'Immediate']}))
        test_corpus = ArrayCorpus(test_corpus)
    with open(args.output, 'wt') if args.output else stdout as output:
        with CsvWriter(output, ['text', 'prediction', 'real']) as writer:
            for i, y1 in enumerate(y):
                writer.write_row(text=test_corpus[i].features, prediction=y_pred[i], real=test_corpus[i].cls)


if __name__ == '__main__':
    main()
