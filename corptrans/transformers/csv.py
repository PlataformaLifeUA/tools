from typing import Union, List, Any

from corptrans import NoTrainingTransformer, Sample


class CsvTransformer(NoTrainingTransformer):
    def __init__(self, columns: Union[str, List[str]], cls: str) -> None:
        super(CsvTransformer, self).__init__()
        self.__columns = [columns] if isinstance(columns, str) else columns
        self.__cls = cls

    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
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
