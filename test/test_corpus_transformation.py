import unittest

from scipy.sparse import lil_matrix
from tqdm import tqdm

from corpustrans.corpus import CsvCorpus, Corpus, ArrayCorpus
from corpustrans.transformers.embeddings import EmbeddingExtension
from corpustrans.transformers.gensim import Tokens2Freq, Dict2Tuples, BoW, TfIdf, Lsi, Lda, Gensim2Matrix
from corpustrans.transformers.preprocess import Preprocess
from corpustrans.transformers.base import CsvTransformer, ClassReduction
from sklearn_train_old_ import create_ml


def corpus2matrix(corpus: Corpus) -> lil_matrix:
    M = lil_matrix((len(corpus), len(corpus.metadata['dictionary'])))
    for i, sample in enumerate(corpus):
        for token in sample:
            M[i, token[0]] = token[1]
    return M


class Series(object):
    def __init__(self, low, high):
        self.current = low
        self.high = high

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __next__(self):
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return self.current - 1

class MyTestCase(unittest.TestCase):

    def test_00_simple(self):

        for n in  Series(1, 10):
            print(n)

    def test_01_simple(self):
        # with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            for sample in corpus_train:
                print(sample)

    def test_lda(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            corpus_train.add_transformer(CsvTransformer('text', 'cls'))
            corpus_train.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            corpus_train.add_transformer(Preprocess())
            corpus_train.add_transformer(Tokens2Freq())
            corpus_train.add_transformer(EmbeddingExtension('en', 10, 0.85))
            corpus_train.add_transformer(Dict2Tuples())
            corpus_train.add_transformer(BoW())
            corpus_train.add_transformer(TfIdf())
            corpus_train.add_transformer(Gensim2Matrix())
            # corpus_train = ArrayCorpus(corpus_train)
            corpus_train.train()
            ml = create_ml(corpus2matrix(corpus_train), corpus_train.classes(), 'SVM')
            with CsvCorpus('data/gold_reddit_corpus_agree.csv') as corpus_test:
                # corpus_test.add_transformer(corpus_train.transformers)
                corpus_test.set_transformers(corpus_train.transformers)
                for sample in corpus_test:
                    print(sample)  # corpus_train.transformers.transform(sample))
                self.assertListEqual(sample.features,
                             [(99, 0.7509992)])
        self.assertEqual(sample.cls, 1)

# [(228, 1), (44, 1), (117, 1), (383, 1), (412, 0.8552255034446716), (388, 0.8684839606285095), (368, 0.8723466992378235), (415, 0.8729560971260071), (21, 0.879010796546936), (274, 0.8831303715705872), (101, 0.8891324996948242), (377, 0.8930044174194336), (397, 0.8959280848503113), (4386, 1), (90, 1), (1154, 1)]
if __name__ == '__main__':
    unittest.main()
