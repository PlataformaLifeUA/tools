from abc import ABC, ABCMeta, abstractmethod
from typing import Union, List, Any, Dict

from corpustrans.sample import Sample


class Transformer(ABC):
    __metaclass__ = ABCMeta

    @property
    def trained(self) -> bool:
        return self._trained

    @property
    def attributes(self) -> dict:
        return self.__attributes

    def __init__(self, trained: bool = False):
        self.__attributes = {}
        self._trained = trained

    @abstractmethod
    def train(self, sample: Sample) -> Sample:
        pass

    @abstractmethod
    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        pass

    def _set_attributes(self, attributes: dict) -> None:
        self.__attributes = attributes

    def _update_attributes(self, attributes: dict) -> None:
        self.__attributes.update(attributes)

    def start_training(self):
        pass

    def finish_training(self):
        pass


class Transformers(Transformer):
    def __init__(self):
        super(Transformers, self).__init__(False)
        self._transformers = []

    def add(self, transformer: Union['Transformer', List['Transformer']]):
        if transformer:
            transformers = transformer if isinstance(transformer, List) else [transformer]
            for transformer in transformers:
                if transformer:
                    self._update_attributes(transformer.attributes)
                    transformer._set_attributes(self.attributes)
                    self._transformers.append(transformer)

    def transform(self, sample: Sample) -> Sample:
        for transformer in self._transformers:
            sample = transformer.transform(sample)
        return sample

    def train(self, sample: Sample) -> Sample:
        for transformer in self._transformers:
            sample = transformer.train(sample)
            # transformer._trained = True
        # self._trained = True
        return sample

    def start_training(self):
        for transformer in self._transformers:
            transformer.start_training()

    def finish_training(self):
        for transformer in self._transformers:
            transformer.finish_training()
            transformer._trained = True
        self._trained = True


class NoTrainingTransformer(Transformer, ABC):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(NoTrainingTransformer, self).__init__(True)

    def train(self, sample: Sample) -> Sample:
        return self.transform(sample)


class CsvTransformer(NoTrainingTransformer):
    def __init__(self, columns: Union[str, List[str]], cls: str) -> None:
        super(CsvTransformer, self).__init__()
        self.__columns = [columns] if isinstance(columns, str) else columns
        self.__cls = cls

    def transform(self, sample: Sample) -> Sample:
        if not isinstance(sample, Sample):
            return sample
        features = []
        for feature in self.__columns:
            if feature in sample.features:
                features.append(sample.features[feature])
            else:
                raise ValueError(f'The column "{feature}" are not in the CSV columns: {self.__columns}.')
        if self.__cls in sample:
            return Sample('\n'.join(features), sample.features[self.__cls])
        raise ValueError(f'The column {self.__cls} is not in the CSV.')


class ClassReduction(NoTrainingTransformer):
    def __init__(self, rules: Dict[Any, Union[List[Any], Any]]) -> None:
        super(ClassReduction, self).__init__()
        self.__rules = {}
        for to, fr in rules.items():
            fr = fr if isinstance(fr, List) else [fr]
            for value in fr:
                if value in self.__rules:
                    raise ValueError('It is not allowed to give the same class to different rules.')
                self.__rules[value] = to

    def transform(self, sample: Sample) -> Sample:
        if not isinstance(sample, Sample):
            return sample
        return Sample(sample.features, self.__rules[sample.cls])
