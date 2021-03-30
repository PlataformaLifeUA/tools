import unittest

from corptrans.corpus.csv import CsvCorpus
from corptrans.transformers import ClassReduction
from corptrans.transformers.csv import CsvTransformer
from corptrans.transformers.embeddings import EmbeddingExtension
from corptrans.transformers.gensim import Tokens2Freq, Dict2Tuples, BoW, TfIdf, Lsi, Lda
from corptrans.transformers.preprocess import Preprocess


class MyTestCase(unittest.TestCase):
    def test_01_bow(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            corpus_train.add_transformer(CsvTransformer('text', 'cls'))
            corpus_train.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            corpus_train.add_transformer(Preprocess())
            corpus_train.add_transformer(Tokens2Freq())
            corpus_train.add_transformer(EmbeddingExtension('en', 10, 0.85))
            corpus_train.add_transformer(Dict2Tuples())
            corpus_train.add_transformer(BoW())
            corpus_train.train()
        with CsvCorpus('data/reddit_corpus_agree.csv') as test_train:
            test_train.add_transformer(corpus_train.transformers)
            for i, sample in enumerate(test_train):
                pass
            self.assertEqual(i + 1, len(test_train))
            self.assertListEqual(sample.features, [
                (238, 1), (290, 1), (159, 1), (536, 1), (592, 0.8552255034446716), (541, 0.8684839606285095),
                (549, 0.8723466992378235), (596, 0.8729560971260071), (82, 0.879010796546936),
                (346, 0.8831303715705872), (164, 0.8891324996948242), (502, 0.8930044174194336),
                (216, 0.8959280848503113), (3701, 1), (211, 1), (622, 1)
            ])
            self.assertEqual(sample.cls, 1)

    def test_02_tfidf(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            corpus_train.add_transformer(CsvTransformer('text', 'cls'))
            corpus_train.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            corpus_train.add_transformer(Preprocess())
            corpus_train.add_transformer(Tokens2Freq())
            corpus_train.add_transformer(EmbeddingExtension('en', 10, 0.85))
            corpus_train.add_transformer(Dict2Tuples())
            corpus_train.add_transformer(BoW())
            corpus_train.add_transformer(TfIdf())
            corpus = corpus_train.train()
        size = len(corpus)
        self.assertListEqual(corpus[-1].features, [
            (238, 0.2961783835830041), (290, 0.09302013280227447), (159, 0.08952428359479193),
            (536, 0.17693053596782407), (592, 0.15131550669781793), (541, 0.1464988436074247),
            (549, 0.15434476904591035), (596, 0.1544525901408843), (82, 0.13042479229572818),
            (346, 0.1353333429200973), (164, 0.050158230579069464), (502, 0.13684646904041656),
            (216, 0.1417751894550695), (3701, 0.8008689714194923), (211, 0.11756486144597839),
            (622, 0.2009702696735927)
        ])
        self.assertEqual(corpus[-1].cls, 1)
        with CsvCorpus('data/reddit_corpus_agree.csv') as test_corpus:
            test_corpus.add_transformer(corpus_train.transformers)
            for i, sample in enumerate(test_corpus):
                pass
            self.assertEqual(size, len(test_corpus))
            self.assertListEqual(sample.features, [
                (238, 0.2961783835830041), (290, 0.09302013280227447), (159, 0.08952428359479193),
                (536, 0.17693053596782407), (592, 0.15131550669781793), (541, 0.1464988436074247),
                (549, 0.15434476904591035), (596, 0.1544525901408843), (82, 0.13042479229572818),
                (346, 0.1353333429200973), (164, 0.050158230579069464), (502, 0.13684646904041656),
                (216, 0.1417751894550695), (3701, 0.8008689714194923), (211, 0.11756486144597839),
                (622, 0.2009702696735927)
            ])
            self.assertEqual(sample.cls, 1)

    def test_03_lsi(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            corpus_train.add_transformer(CsvTransformer('text', 'cls'))
            corpus_train.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            corpus_train.add_transformer(Preprocess())
            corpus_train.add_transformer(Tokens2Freq())
            corpus_train.add_transformer(EmbeddingExtension('en', 10, 0.85))
            corpus_train.add_transformer(Dict2Tuples())
            corpus_train.add_transformer(BoW())
            corpus_train.add_transformer(TfIdf())
            corpus_train.add_transformer(Lsi())
            corpus = corpus_train.train()
            for sample in corpus:
                pass
            self.assertEqual(sample.cls, 1)
            self.assertEqual(len(sample), 171)
            self.assertEqual(len(sample.features), 171)

    def test_04_lsi(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            corpus_train.add_transformer(CsvTransformer('text', 'cls'))
            corpus_train.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            corpus_train.add_transformer(Preprocess())
            corpus_train.add_transformer(Tokens2Freq())
            corpus_train.add_transformer(EmbeddingExtension('en', 10, 0.85))
            corpus_train.add_transformer(Dict2Tuples())
            corpus_train.add_transformer(BoW())
            corpus_train.add_transformer(TfIdf())
            corpus_train.add_transformer(Lda())
            corpus = corpus_train.train()
            for sample in corpus:
                pass
            print(sample)

            self.assertEqual(sample.cls, 1)
            self.assertEqual(corpus[-2].cls, 0)


if __name__ == '__main__':
    unittest.main()
