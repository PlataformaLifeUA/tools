from collections import OrderedDict
from typing import Dict, List, Union

import typing
from googletrans import Translator
from pandas import DataFrame
from tqdm import tqdm
import yaml
from yaml import Loader, Dumper
from yaml.representer import SafeRepresenter
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))

Dumper.add_representer(OrderedDict, dict_representer)
Loader.add_constructor(_mapping_tag, dict_constructor)
Dumper.add_representer(str, SafeRepresenter.represent_str)
# Dumper.add_representer(unicode, SafeRepresenter.represent_unicode)


def translate(corpus: Dict[str, List[str]], src: str, dest: str) -> Dict[str, List[str]]:
    translator = Translator()
    languages = corpus['Language']
    texts = corpus['Text']
    for i in tqdm(range(len(texts)), desc='Translating the corpus.'):
        if languages[i].lower() == src.lower():
            languages[i] = dest
            texts[i] = translator.translate(texts[i], dest=dest, src=src).text
    return corpus


def format_value(value: float) -> str:
    """
    Format the metrics.
    :param value: The value to format.
    :return: The formated value.
    """
    return '{0:.2f}%'.format(value * 100)


def save(fname: str, content: Union[str, bytes], encoding: str = 'utf-8'):
    with open(fname, 'wt', encoding=encoding) if isinstance(content, str) else open(fname, 'wb') as file:
        file.write(content)


def save_csv(corpus, fname: str, encoding: str = 'utf-8') -> None:
    df = DataFrame(corpus)
    df.to_csv(fname, encoding=encoding, index=False)


def save_yaml(d: OrderedDict, fname: str) -> None:
    with open(fname, 'wt') as file:
        yaml.dump(d, file, Dumper=Dumper, default_flow_style=False)