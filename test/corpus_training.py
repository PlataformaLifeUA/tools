import unittest

from corptrans.corpus.corpus import ArrayCorpus
from corptrans.corpus.csv import CsvCorpus
from corptrans.transformers import ClassReduction
from corptrans.transformers.csv import CsvTransformer
from corptrans.transformers.embeddings import EmbeddingExtension
from corptrans.transformers.gensim import Tokens2Freq, Dict2Tuples, BoW, TfIdf
from corptrans.transformers.preprocess import Preprocess
from eval import metrics, print_metrics
from sklearn_train import corpus2matrix
from sklearn_train_old_ import create_ml


class MyTestCase(unittest.TestCase):
    def test_trainning(self):
        with CsvCorpus('data/gold_reddit_corpus_agree.csv') as corpus:
            corpus.add_transformer(CsvTransformer('text', 'cls'))
            corpus.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            corpus.add_transformer(Preprocess())
            corpus.add_transformer(Tokens2Freq())
            corpus.add_transformer(EmbeddingExtension('en', 10, 0.85))
            corpus.add_transformer(Dict2Tuples())
            corpus.add_transformer(BoW())
            corpus.add_transformer(TfIdf())
            train_corpus = corpus.train()
        ml = create_ml(corpus2matrix(train_corpus), train_corpus.classes(), 'SVM')
        with CsvCorpus('data/reddit_corpus_agree.csv') as test_corpus:
            test_corpus.set_transformers(train_corpus.transformers)
            test_corpus = ArrayCorpus(test_corpus)
            test_corpus._metadata = corpus.metadata
            X = corpus2matrix(test_corpus)
            y = test_corpus.classes()
        with CsvCorpus('data/reddit_corpus_agree.csv') as test_corpus:
            test_corpus.add_transformer(CsvTransformer('text', 'cls'))
            test_corpus.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            test_corpus = ArrayCorpus(test_corpus)

            y_pred = [ml.predict(X[i])[0] for i in range(X.shape[0])]
            print_metrics(metrics(y, y_pred))
            for i, y1 in enumerate(y):
                print(test_corpus[i], y_pred[i], sep=': ')

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
