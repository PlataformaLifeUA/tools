from typing import Any, Union

from corptrans import NoTrainingTransformer, Sample
from wemb import WordEmbeddings


def add_term(sample: dict, term: str, weight: float = 1):
    if term not in sample:
        sample[term] = 0
    sample[term] += 1 * weight


class EmbeddingExtension(NoTrainingTransformer):
    def __init__(self, lang: str, neighbors: int, threshold: float) -> None:
        super(EmbeddingExtension, self).__init__()
        self.embeddings = WordEmbeddings(lang, neighbors=neighbors, threshold=threshold)
        self.lang = lang

    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        features = {}
        for term in sample:
            add_term(features, term)
            for synonym, weight in self.embeddings.synonyms(term, self.lang):
                add_term(features, synonym, weight)
        return sample.gen(features)
