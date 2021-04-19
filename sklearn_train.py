from argparse import ArgumentParser

from scipy.sparse import lil_matrix

from corptrans import Corpus, Transformers
from corptrans.corpus.corpus import ArrayCorpus
from corptrans.corpus.csv import CsvCorpus
from corptrans.transformers import ClassReduction
from corptrans.transformers.csv import CsvTransformer
from corptrans.transformers.embeddings import EmbeddingExtension
from corptrans.transformers.gensim import Tokens2Freq, Dict2Tuples, BoW, TfIdf
from corptrans.transformers.preprocess import Preprocess
from eval import print_metrics, metrics
from sklearn_train_old_ import create_ml

SVM = 'svm'
KMEANS = 'kmeans'
CLASSIFIERS = [SVM, KMEANS]


class SkLearnSimpleArgParser(object):
    @property
    def corpus(self) -> str:
        return self._args.corpus

    @property
    def test(self) -> str:
        return self._args.test[0]

    @property
    def method(self) -> str:
        return self._args.method

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


def corpus2matrix(corpus: Corpus) -> lil_matrix:
    M = lil_matrix((len(corpus), len(corpus.metadata['dictionary'])))
    for i, sample in enumerate(corpus):
        for token in sample:
            M[i, token[0]] = token[1]
    return M


def train(fname: str, ml: str):
    with CsvCorpus(fname) as corpus:
        corpus.add_transformer(CsvTransformer('text', 'cls'))
        corpus.add_transformer(ClassReduction({0: ['No risk'], 1: ['Possible', 'Risk', 'Urgent', 'Immediate']}))
        train_corpus = corpus_trainer(corpus)
    return create_ml(corpus2matrix(train_corpus), train_corpus.classes(), ml), train_corpus


def corpus_trainer(corpus):
    corpus.add_transformer(Preprocess())
    corpus.add_transformer(Tokens2Freq())
    # corpus.add_transformer(EmbeddingExtension('en', 10, 0.85))
    corpus.add_transformer(Dict2Tuples())
    corpus.add_transformer(BoW())
    # corpus.add_transformer(TfIdf())
    train_corpus = corpus.train()
    return train_corpus


def evaluate(ml, fname: str, corpus: Corpus) -> dict:
    with CsvCorpus(fname) as test_corpus:
        test_corpus.set_transformers(corpus.transformers)
        test_corpus = ArrayCorpus(test_corpus)
        test_corpus._metadata = corpus.metadata
        X = corpus2matrix(test_corpus)
        y = test_corpus.classes()
        y_pred = [ml.predict(X[i])[0] for i in range(X.shape[0])]
        return metrics(y, y_pred)


def main() -> None:
    args = SkLearnSimpleArgParser()
    ml, corpus = train(args.corpus, args.method)
    measures = evaluate(ml, args.test, corpus)
    print_metrics(measures)
    # with CsvCorpus(args.test) as test_corpus:
    #     test_corpus.set_transformers(corpus.transformers)
    #     test_corpus = ArrayCorpus(test_corpus)
    #     test_corpus._metadata = corpus.metadata
    #     X = corpus2matrix(test_corpus)
    #     y = test_corpus.classes()
    #     y_pred = [ml.predict(X[i])[0] for i in range(X.shape[0])]
    #     print_metrics(metrics(y, y_pred))
    # with CsvCorpus(args.test) as test_corpus:
    #     test_corpus.add_transformer(CsvTransformer('text', 'cls'))
    #     test_corpus.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
    #     test_corpus = ArrayCorpus(test_corpus)
    #
    #     y_pred = [ml.predict(X[i])[0] for i in range(X.shape[0])]
    #     print_metrics(metrics(y, y_pred))
    #     for i, y1 in enumerate(y):
    #         print(test_corpus[i], y_pred[i], sep=': ')

if __name__ == '__main__':
    main()
