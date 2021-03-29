from abc import ABC, ABCMeta, abstractmethod
from collections import Iterable, Sequence
from tempfile import NamedTemporaryFile
from types import GeneratorType
from typing import Tuple, List, Any, TextIO, BinaryIO, Union, Iterator

from filedatasource import CsvReader, ReadMode, CsvWriter

from corpustrans.sample import Sample
from corpustrans.transformers.base import Transformer, Transformers
from random import Random


class Corpus(ABC):
    __metaclass__ = ABCMeta

    @property
    def transformers(self) -> Transformers:
        return self._transformers

    @property
    def transformed(self) -> bool:
        return self._transformed

    @property
    def metadata(self) -> dict:
        # if not self.transformed:
        #     self.train()
        return self._metadata

    def __init__(self, transformers: Transformers = None, metadata: dict = None):
        self._transformers = transformers if transformers else Transformers()
        self._metadata = metadata if metadata else {}
        self._samples = []
        self._transformed = False

    def add_transformer(self, transformer: Transformer) -> None:
        self._transformers.add(transformer)

    def set_transformers(self, transformers: Transformers) -> None:
        self._transformers = transformers

    def train(self) -> None:
        if not self.transformed:
            self.transformers.start_training()
            try:
                while True:
                    n = self.next()
                    if isinstance(n, GeneratorType):
                        for sample in n:
                            self._samples.append(self.transformers.train(sample))
                    else:
                        self._samples.append(self.transformers.train(n))
            except StopIteration:
                self.transformers.finish_training()
            self._metadata.update(self.transformers.attributes)
            self._transformed = True

    @abstractmethod
    def next(self) -> Sample:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self):
        if self._samples:
            yield from self._samples
        else:
            return self

    def __next__(self):
        # if not self.transformed and not self.transformers.trained:
        #     self.train()
        return self._transformers.transform(self.next())

    def __getitem__(self, item: Union[int, Tuple[int, int]]) -> Union[Sample, Any]:
        if not isinstance(item, int) and isinstance(item, Tuple) and len(item) > 1 and isinstance(item[0], int):
            ValueError('The index must be an integer or a tuple.')
        # If the corpus need to have trained transformers, then train it
        # if not self.transformers.trained:
        #     self.train()
        if isinstance(item, int):
            # If the index is an integer then return the sample.
            return self.get_sample(item)
        # If the index is a tuple, then return the specific attribute of that sample
        sample = self.get_sample(item[0])
        return sample[item[1]] if len(item) == 2 else sample[tuple([e for i, e in item if i > 0])]

    @abstractmethod
    def divide(self, factor: float, seed: int = None) -> Tuple['Corpus', 'Corpus']:
        pass

    @abstractmethod
    def cross_validation(self, folders: int, seed: int = None): # -> Iterable[Tuple['Corpus', 'Corpus']]:
        pass

    def get_sample(self, i: int) -> Sample:
        pass

    def classes(self) -> List[Any]:
        # if not self.transformed:
        #     self.train()
        return [sample.cls for sample in self]

    def samples(self) -> List[Sample]:
        # if not self.transformers.trained:
        #     self.train()
        # if self._transformed:
        #     return [sample for sample in self._samples]
        return [sample for sample in self]


class DownloadableCorpus(Corpus):
    @abstractmethod
    def download(self, dest: str = None) -> None:
        pass


class ArrayCorpus(Corpus):
    def __init__(self, samples: Tuple[Corpus, List[Sample]] = None, transformers: Transformers = None, attributes: dict = None):
        super(ArrayCorpus, self).__init__(transformers, attributes)
        self.__samples = [sample for sample in samples] if samples else []
        self.__i = 0

    def cross_validation(self, folders: int, seed: int = None):
        for folder in range(folders):
            seed = Random(seed).randint(0, 2 ** 32)
            yield self.divide(1 - 1 / folders, seed)
        raise StopIteration

    def divide(self, factor: float, seed: int = None) -> Tuple['Corpus', 'Corpus']:
        length = len(self)
        train = Random(seed).sample(self, int(length * factor))
        test = [sample for sample in self if sample not in train]
        return ArrayCorpus(train), ArrayCorpus(test)

    def next(self):
        if self.__i < len(self.__samples):
            return self.__samples[self.__i]
        else:
            raise StopIteration

    def __len__(self) -> int:
        return len(self.__samples)

    def __enter__(self):
        self.__i = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __str__(self) -> str:
        return str(self.__samples)

    def __repr__(self) -> str:
        return repr(self.__samples)


class CsvCorpus(Corpus):
    @property
    def file(self) -> str:
        return self.__fname

    @property
    def encoding(self) -> str:
        return self.__encoding

    def __init__(self, fname: Union[str, TextIO, BinaryIO], encoding: str = 'utf-8',
                 transformers: Transformers = None, attributes: dict = None):
        super(CsvCorpus, self).__init__(transformers, attributes)
        self.__fname = fname
        self.__encoding = encoding
        self._reader = CsvReader(fname, ReadMode.DICT, encoding)

    def cross_validation(self, folders: int, seed: int = None):  # -> Iterable[Tuple['Corpus', 'Corpus']]:
        for folder in range(folders):
            seed = Random(seed).randint(0, 2 ** 32)
            yield self.divide(1 - 1 / folders, seed)
        raise StopIteration

    def divide(self, factor: float, seed: int = None) -> Tuple['Corpus', 'Corpus']:
        train_tmp = NamedTemporaryFile(suffix='.csv.gz').name
        test_tmp = NamedTemporaryFile(suffix='.csv.gz').name
        length = len(self)
        train = {i for i in Random(seed).sample(list(range(length)), int(length * factor))}
        fieldnames = self._reader.fieldnames + (['cls'] if self.cls else [])
        with CsvWriter(train_tmp, fieldnames) as train_file:
            with CsvWriter(test_tmp, fieldnames) as test_file:
                for i, sample in enumerate(self):
                    row = sample.features + ([sample.cls] if sample.cls else [])
                    if i in train:
                        train_file.write_list(row)
                    else:
                        test_file.write_list(row)
        return CsvCorpus(train_file), ArrayCorpus(test_file)

    def next(self):
        return Sample(next(self._reader))

    def close(self) -> None:
        self._reader.close()

    # def __iter__(self):
    #     return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self) -> int:
        return len(self._reader)
