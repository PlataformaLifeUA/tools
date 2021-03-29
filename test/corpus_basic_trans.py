import unittest

from corptrans.corpus.csv import CsvCorpus
from corptrans.transformers import ClassReduction
from corptrans.transformers.csv import CsvTransformer
from corptrans.transformers.embeddings import EmbeddingExtension
from corptrans.transformers.gensim import Tokens2Freq, Dict2Tuples, BoW
from corptrans.transformers.preprocess import Preprocess


class MyTestCase(unittest.TestCase):
    def test_01_simple(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            for i, sample in enumerate(corpus_train):
                pass
            self.assertEqual(i + 1, len(corpus_train))
            self.assertDictEqual(sample.features,
                                 {'text': 'Why do so many people get to die by accident and for me to die I have to '
                                          'kill myself.\nFuck', 'cls': 'Risk'},)

    def test_02_csv_transformer(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            corpus_train.add_transformer(CsvTransformer('text', 'cls'))
            for i, sample in enumerate(corpus_train):
                pass
            self.assertEqual(i + 1, len(corpus_train))
            self.assertEqual(sample.features,
                             'Why do so many people get to die by accident and for me to die I have to kill myself.'
                             '\nFuck')
            self.assertEqual(sample.cls, 'Risk')

    def test_03_class_reduction(self):
        # with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            corpus_train.add_transformer(CsvTransformer('text', 'cls'))
            corpus_train.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            for i, sample in enumerate(corpus_train):
                pass
            self.assertEqual(i + 1, len(corpus_train))
            self.assertEqual(sample.features,
                             'Why do so many people get to die by accident and for me to die I have to kill myself.'
                             '\nFuck')
            self.assertEqual(sample.cls, 1)

    def test_04_preprocess(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            corpus_train.add_transformer(CsvTransformer('text', 'cls'))
            corpus_train.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            corpus_train.add_transformer(Preprocess())
            for i, sample in enumerate(corpus_train):
                pass
            self.assertEqual(i + 1, len(corpus_train))
            self.assertListEqual(sample.features, ['many', 'people', 'get', 'die', 'accident', 'die', 'kill', 'fuck'])
            self.assertEqual(sample.cls, 1)

    def test_05_token2freq(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            corpus_train.add_transformer(CsvTransformer('text', 'cls'))
            corpus_train.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            corpus_train.add_transformer(Preprocess())
            corpus_train.add_transformer(Tokens2Freq())
            for i, sample in enumerate(corpus_train):
                pass
            self.assertEqual(i + 1, len(corpus_train))
            self.assertDictEqual(sample.features, {'die': 2, 'many': 1, 'people': 1, 'get': 1,
                                                   'accident': 1, 'kill': 1, 'fuck': 1})
            self.assertEqual(sample.cls, 1)

    def test_06_embeddings(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            corpus_train.add_transformer(CsvTransformer('text', 'cls'))
            corpus_train.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            corpus_train.add_transformer(Preprocess())
            corpus_train.add_transformer(Tokens2Freq())
            corpus_train.add_transformer(EmbeddingExtension('en', 10, 0.85))
            for i, sample in enumerate(corpus_train):
                pass
            self.assertEqual(i + 1, len(corpus_train))
            self.assertDictEqual(sample.features, {
                'many': 1, 'people': 1, 'get': 1, 'die': 1, 'shape': 0.8552255034446716, 'face': 0.8684839606285095,
                'fly': 0.8723466992378235, 'stem': 0.8729560971260071, 'count': 0.879010796546936,
                'sit': 0.8831303715705872, 'go': 0.8891324996948242, 'act': 0.8930044174194336,
                'lead': 0.8959280848503113, 'accident': 1, 'kill': 1, 'fuck': 1})
            self.assertEqual(sample.cls, 1)

    def test_07_embeddings(self):
        with CsvCorpus('data/reddit_corpus_agree.csv') as corpus_train:
            corpus_train.add_transformer(CsvTransformer('text', 'cls'))
            corpus_train.add_transformer(ClassReduction({0: ['No risk'], 1: ['Risk']}))
            corpus_train.add_transformer(Preprocess())
            corpus_train.add_transformer(Tokens2Freq())
            corpus_train.add_transformer(EmbeddingExtension('en', 10, 0.85))
            corpus_train.add_transformer(Dict2Tuples())
            for i, sample in enumerate(corpus_train):
                pass
            self.assertEqual(i + 1, len(corpus_train))
            self.assertListEqual(sample.features, [
                ('many', 1), ('people', 1), ('get', 1), ('die', 1), ('shape', 0.8552255034446716),
                ('face', 0.8684839606285095), ('fly', 0.8723466992378235), ('stem', 0.8729560971260071),
                ('count', 0.879010796546936), ('sit', 0.8831303715705872), ('go', 0.8891324996948242),
                ('act', 0.8930044174194336), ('lead', 0.8959280848503113), ('accident', 1), ('kill', 1), ('fuck', 1)])
            self.assertEqual(sample.cls, 1)


if __name__ == '__main__':
    unittest.main()
