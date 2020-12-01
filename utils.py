from typing import Dict, List, Union
from googletrans import Translator
from pandas import DataFrame
from tqdm import tqdm


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
