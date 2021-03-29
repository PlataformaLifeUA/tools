from collections import Counter
from typing import Any, Union

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel, LdaModel

from corptrans import Sample, FuturibleTransformer, NoTrainingTransformer, Transformer


class Tokens2Freq(NoTrainingTransformer):
    def transform(self, sample: Sample) -> Sample:
        return sample.gen(Counter(sample))


class Dict2Tuples(NoTrainingTransformer):
    def transform(self, sample: Sample) -> Sample:
        return sample.gen([(term, weight) for term, weight in sample.features.items()])


class BoW(Transformer):
    def __init__(self):
        super(BoW, self).__init__()
        self.__dictionary = Dictionary()
        self.attributes['dictionary'] = self.__dictionary

    def train(self, sample: Sample) -> Sample:
        self.__dictionary.add_documents([[feature[0] for feature in sample.features]])
        return self.transform(sample)

    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        token2id = self.__dictionary.token2id
        return sample.gen([(token2id[term], freq) for term, freq in sample if term in token2id])


class TfIdf(FuturibleTransformer):
    def __init__(self):
        super(TfIdf, self).__init__()
        self.tfidf = None

    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        return sample.gen(self.tfidf[sample])

    def finish_training(self):
        self.tfidf = TfidfModel(self._samples)

        for sample in self._samples:
            sample.set_features(self.tfidf[sample])


class Lsi(FuturibleTransformer):
    def __init__(self):
        super(Lsi, self).__init__()
        self.lsi = None

    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        return sample.gen(self.lsi[sample])

    def finish_training(self):
        self.lsi = LsiModel(self._samples)

        for sample in self._samples:
            sample.set_features(self.lsi[sample])


class Lda(FuturibleTransformer):
    def __init__(self):
        super(Lda, self).__init__()
        self.lda = None

    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        return sample.gen(self.lda[sample])

    def finish_training(self):
        self.lda = LdaModel(self._samples)

        for sample in self._samples:
            sample.set_features(self.lda[sample])


class Gensim2Matrix(FuturibleTransformer):
    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        row = [0] * len(self.attributes['dictionary'])
        for token in sample:
            row[token[0]] = token[1]
        return sample.gen(row)

    def finish_training(self):
        for i, sample in enumerate(self._samples):
            row = {token[0]: token[1] for token in sample}
            sample.set_features(row)
