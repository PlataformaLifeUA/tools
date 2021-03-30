from abc import ABC, ABCMeta, abstractmethod
from random import Random
from typing import Tuple, List, Any, Iterable, Union

from corptrans import Sample
from corptrans.transformers import Transformers, Transformer


class Corpus(ABC, Iterable):
    __metaclass__ = ABCMeta

    @property
    def transformers(self) -> Transformers:
        return self._transformers

    @property
    def metadata(self) -> dict:
        return self._metadata

    def __init__(self, transformers: Transformers = None, metadata: dict = None):
        self._transformers = transformers if transformers else Transformers()
        self._metadata = metadata if metadata else {}
        self._transformed = False

    def add_transformer(self, transformer: Transformer) -> None:
        self._transformers.add(transformer)

    def set_transformers(self, transformers: Transformers) -> None:
        self._transformers = transformers

    def train(self) -> 'ArrayCorpus':
        corpus = ArrayCorpus(transformers=self.transformers, metadata=self._metadata)
        self.transformers.start_training()
        try:
            while True:
                corpus.add(self.transformers.train(self.next()))
        except StopIteration:
            self.transformers.finish_training()
        self._metadata.update(self.transformers.attributes)
        corpus._metadata = self._metadata
        return corpus

    @abstractmethod
    def next(self) -> Sample:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return self._transformers.transform(self.next())

    @abstractmethod
    def divide(self, factor: float, seed: int = None) -> Tuple['Corpus', 'Corpus']:
        pass

    @abstractmethod
    def cross_validation(self, folders: int, seed: int = None):
        pass

    def get_sample(self, i: int) -> Sample:
        pass

    def classes(self) -> List[Any]:
        return [sample.cls for sample in self]

    def samples(self) -> List[Sample]:
        return [sample for sample in self]


class ArrayCorpus(Corpus):
    def __init__(self, samples: Union[Corpus, List[Sample]] = None, transformers: Transformers = None,
                 metadata: dict = None):
        super(ArrayCorpus, self).__init__(transformers, metadata)
        self.__samples = [sample for sample in samples] if samples else []
        self.__i = 0

    def cross_validation(self, folders: int, seed: int = None):
        for folder in range(folders):
            seed = Random(seed).randint(0, 2 ** 32)
            yield self.divide(1 - 1 / folders, seed)
        raise StopIteration

    def divide(self, factor: float, seed: int = None) -> Tuple['Corpus', 'Corpus']:
        length = len(self)
        train = Random(seed).sample(self.__samples, int(length * factor))
        test = [sample for sample in self if sample not in train]
        return ArrayCorpus(train), ArrayCorpus(test)

    def add(self, sample: Sample) -> None:
        self.__samples.append(sample)

    def next(self):
        if self.__i < len(self.__samples):
            return self.__samples[self.__i]
        else:
            raise StopIteration

    def __iter__(self):
        yield from self.__samples

    def __len__(self) -> int:
        return len(self.__samples)

    def __enter__(self):
        self.__i = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __str__(self) -> str:
        return f'ArrayCorpus{str(self.__samples)}'

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, item: int) -> Sample:
        return self.__samples[item]


