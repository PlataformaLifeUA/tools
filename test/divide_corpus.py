import unittest
from random import Random

from corptrans.corpus.corpus import ArrayCorpus
from corptrans.corpus.csv import CsvCorpus
from corptrans.transformers import ClassReduction
from corptrans.transformers.csv import CsvTransformer
from eval import metrics, print_metrics
from sklearn_train import corpus_trainer, corpus2matrix
from sklearn_train_old_ import create_ml


class MyTestCase(unittest.TestCase):
    def test_gold_reddit(self):
        print('data/reddit_corpus_agree.csv')
        with CsvCorpus('data/gold_reddit_corpus_agree.csv') as corpus:
            corpus.add_transformer(CsvTransformer('text', 'cls'))
            corpus.add_transformer(ClassReduction({0: ['No risk'], 1: ['Possible', 'Risk', 'Urgent', 'Immediate']}))
            corpus = ArrayCorpus(corpus)
            r = Random()
            for i in range(10):
                train_corpus, test_corpus = corpus.divide(0.9, r.randint(0, 10000000))
                train_corpus.save(f'data/gold_reddit_train-{i}.csv.gz', 'text')
                test_corpus.save(f'data/gold_reddit_test-{i}.csv.gz', 'text')
            # train_corpus = corpus_trainer(train_corpus)
            # ml = create_ml(corpus2matrix(train_corpus), train_corpus.classes(), 'SVM')
            # test_corpus._metadata = train_corpus.metadata
            # test_corpus.set_transformers(corpus.transformers)
            # X = corpus2matrix(test_corpus)
            # y = test_corpus.classes()
            # y_pred = [ml.predict(X[i])[0] for i in range(X.shape[0])]
            # print_metrics(metrics(y, y_pred))
        self.assertEqual(True, True)

    # def test_reddit(self):
    #     print('data/reddit_corpus_agree.csv')
    #     with CsvCorpus('data/reddit_corpus_agree.csv') as corpus:
    #         corpus.add_transformer(CsvTransformer('text', 'cls'))
    #         corpus.add_transformer(ClassReduction({0: ['No risk'], 1: ['Possible', 'Risk', 'Urgent', 'Immediate']}))
    #         corpus = ArrayCorpus(corpus)
    #         train_corpus, test_corpus = corpus.divide(0.8, 0)
    #         train_corpus = corpus_trainer(train_corpus)
    #         ml = create_ml(corpus2matrix(train_corpus), train_corpus.classes(), 'SVM')
    #         test_corpus._metadata = train_corpus.metadata
    #         test_corpus.set_transformers(corpus.transformers)
    #         X = corpus2matrix(test_corpus)
    #         y = test_corpus.classes()
    #         y_pred = [ml.predict(X[i])[0] for i in range(X.shape[0])]
    #         print_metrics(metrics(y, y_pred))
    #     self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
