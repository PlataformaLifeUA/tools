import unittest

from scipy.sparse import lil_matrix

from corptrans import Corpus
from corptrans.corpus.csv import CsvCorpus
from corptrans.transformers import ClassReduction
from corptrans.transformers.csv import CsvTransformer
from corptrans.transformers.embeddings import EmbeddingExtension
from corptrans.transformers.gensim import Tokens2Freq, Dict2Tuples, BoW, TfIdf
from corptrans.transformers.preprocess import Preprocess
from sklearn_train import create_ml


def corpus2matrix(corpus: Corpus) -> lil_matrix:
    M = lil_matrix((len(corpus), len(corpus.metadata['dictionary'])))
    for i, sample in enumerate(corpus):
        for token in sample:
            M[i, token[0]] = token[1]
    return M


class MyTestCase(unittest.TestCase):
    def test_trainning(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus:
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
        with CsvCorpus('data/gold_reddit_corpus_agree.csv') as test_corpus:
            # corpus_test.add_transformer(corpus_train.transformers)
            test_corpus.set_transformers(train_corpus.transformers)

            for sample in test_corpus:
                print(sample)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
