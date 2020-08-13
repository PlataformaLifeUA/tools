from typing import Dict, List
from googletrans import Translator
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
