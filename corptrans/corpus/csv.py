from random import Random
from tempfile import NamedTemporaryFile
from typing import Union, TextIO, BinaryIO, Tuple

from filedatasource import CsvReader, ReadMode, CsvWriter

from corptrans import Corpus, Transformers, Sample


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
        return CsvCorpus(train_file), CsvCorpus(test_file)

    def next(self):
        return Sample(next(self._reader))

    def close(self) -> None:
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self) -> int:
        return len(self._reader)
