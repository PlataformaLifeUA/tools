from abc import abstractmethod, ABCMeta, ABC
from typing import Any

from argparser.train import TrainArgParser, SKLEARN, RASA, ENGINES
from corptrans import Transformers
from corptrans.corpus.csv import CsvCorpus
from corptrans.transformers import ClassReduction
from corptrans.transformers.csv import CsvTransformer
from corptrans.transformers.embeddings import EmbeddingExtension
from corptrans.transformers.gensim import Tokens2Freq, Dict2Tuples, BoW, TfIdf, Lsi, Lda
from corptrans.transformers.preprocess import Preprocess


class Trainer(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, corpus: str):
        pass

    @abstractmethod
    def save(self, file: str):
        pass


class StandardNlpTransformer(Transformers):
    def __init__(self, lang: str, tfidf: bool, lsi: bool, lda: bool,
                 embeddings: bool, embedding_num: int, embedding_threshold: float) -> None:
        super(StandardNlpTransformer, self).__init__()
        self.add(Preprocess())
        self.add(Tokens2Freq())
        self.add(EmbeddingExtension(lang, embedding_num, embedding_threshold) if embeddings else None)
        self.add(Dict2Tuples())
        self.add(BoW())
        self.add(TfIdf() if tfidf else None)
        self.add(Lsi() if lsi else None)
        self.add(Lda() if lda else None)


class SkLearnTrainer(Trainer):
    def __init__(self, classifier: str, lang: str, tfidf: bool, lsi: bool, lda: bool,
                 embeddings: bool, embedding_num: int, embedding_threshold: float):
        self.__classifier = classifier
        self.__lang = lang
        self.__tfidf = tfidf
        self.__lsi = lsi
        self.__lda = lda
        self.__embeddings = embeddings
        self.__embedding_num = embedding_num
        self.__embedding_threshold = embedding_threshold

    def train(self, fname: str) -> Any:
        with CsvCorpus(fname) as corpus:
            corpus.add_transformer(CsvTransformer('text', 'cls'))
            corpus.add_transformer(ClassReduction({0: ['No risk'], 1: ['Possible', 'Risk', 'Urgent', 'Immediate']}))
            corpus.add_transformer(StandardNlpTransformer(self.__lang, self.__tfidf, self.__lsi, self.__lda,
                                                          self.__embeddings, self.__embedding_num,
                                                          self.__embedding_threshold))
            train_corpus = corpus.train()


    def save(self, fname: str):
        pass


class RasaTrainer(Trainer):
    pass


def build_trainer(engine: str, classifier: str, **kwargs) -> Trainer:
    if engine == SKLEARN:
        return SkLearnTrainer(classifier, kwargs['lang'], kwargs['tfidf'], kwargs['lsi'], kwargs['lda'],
                              kwargs['embeddings'], kwargs['embedding_num'], kwargs['embedding_threshold'])
    if engine == RASA:
        return RasaTrainer()
    raise ValueError(f'The {engine} is not a valid engine. The available engines are: {ENGINES}')


def main() -> None:
    args = TrainArgParser()
    trainer = build_trainer(args.engine, args.classifier, tfidf=args.tfidf, lsi=args.lsi, lda=args.lda,
                            embeddings=args.embeddings, embedding_num=args.embedding_num,
                            embedding_threshold=args.embedding_threshold)
    trainer.train(args.train)
    trainer.save(args.output)


if __name__ == '__main__':
    main()