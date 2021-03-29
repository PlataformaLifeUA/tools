from typing import Any, Callable, Tuple, Union


class Sample(object):
    @property
    def cls(self):
        return self._cls

    @property
    def features(self) -> Any:
        return self._features

    @property
    def attributes(self) -> dict:
        return self._attributes

    def __init__(self, features: Any, cls: Any = None, attributes: dict = None) -> None:
        self._features = features
        self._cls = cls
        self._attributes = attributes if attributes else {}

    def apply(self, func: Callable) -> 'Sample':
        return Sample(func(self.features), self.cls, self.attributes)

    def gen(self, features: Any) -> 'Sample':
        return Sample(features, self.cls, self.attributes)

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self):
        return iter(self.features)

    def __getitem__(self, item: Union[int, Tuple]) -> Any:
        return self.features[item]

    def __str__(self) -> str:
        return f'Sample({self.features},)' if self.cls is None else f'({str(self.features)}, {str(self.cls)})'

    def __repr__(self) -> str:
        return str(self)


class FutureSample(Sample):
    def __init__(self, sample: Sample):
        super(FutureSample, self).__init__(sample.features, sample.cls, sample.attributes)

    def set_features(self, features: Any) -> None:
        self._features = features

    def set_cls(self, cls: Any) -> None:
        self._cls = cls

    def _set_attributes(self, attributes: dict) -> None:
        self._attributes = attributes
