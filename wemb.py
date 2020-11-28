from os.path import sep, exists
from typing import List, Union, Dict

from polyglot.downloader import downloader
from polyglot.mapping import Embedding
from polyglot.text import Text
import nltk
import logging

EMBEDDINGS_FILE = 'embeddings_pkl.tar.bz2'
log = logging.getLogger()
# List of the supported languages and their full names
lang_name = {
    'ar': 'arabic',
    'az': 'azerbaijani',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'fi': 'finnish',
    'fr': 'french',
    'de': 'German',
    'el': 'greek',
    'hu': 'hungarian',
    'id': 'indonesian',
    'kk': 'kazakh',
    'ne': 'nepali',
    'no': 'norwegian',
    'pt': 'portuguese',
    'ro': 'romanian',
    'ru': 'russian',
    'sl': 'slovene',
    'es': 'spanish',
    'sv': 'swedish',
    'tg': 'tajik',
    'tr': 'turkish'
}


class InvalidLanguage(Exception):
    """
    Error if the selected language is not available.
    """
    pass


class WordEmbeddings(object):
    """
    Class to get synonyms from word previosly generated embeddings.
    """
    def __init__(self, languages: Union[str, List[str]], folder: str, neighbors: int = 10):
        """
        Constructor.
        :param languages: The languages to support.
        :param folder: The folder where the embeddings models and stopword lists will be stored.
        """
        self.embeddings = self.download_embeddings(languages, folder)
        self.stopwords = self.download_stopwords(languages, folder)
        self.neighbors = neighbors
        self.caches = {lang: {} for lang in self.embeddings}

    def synonyms(self, term: str, lang: str = 'en') -> List[str]:
        """
        Using the word embedding and given a text, get a list of synonyms from the text terms.
        :param text: The text to obtain the synonyms.
        :param lang: The language of that text.
        :return: the top relevant synonyms of the text terms.
        """
        if lang not in self.embeddings:
            raise InvalidLanguage(f'The language code "{lang}" is not supported in the current version.')
        embeddings = self.embeddings[lang].normalize_words()
        cache = self.caches[lang]
        if term in cache:
            return cache[lang]
        synonyms = []
        if len(term) > 1 and term.lower() not in self.stopwords and term in embeddings:
            neighbors = embeddings.nearest_neighbors(term, self.neighbors)
            synonyms = list(zip(neighbors, embeddings.distances(term, neighbors)))
        cache[term] = [t for t in synonyms if t[1] > 0.95]

        return cache[term]

    @staticmethod
    def suggestions(synonyms: Dict[str, List[str]], top: int = 5):
        """
        Give the top most relevant terms in the query synonyms.
        :param synonyms: The synonyms of the query obte
        :param top: The number of synonyms to retrieve.
        """
        synonyms = [synonym[0] for synonym in [value for value in synonyms.values()]]
        return sorted(synonyms, key=lambda x: -x[1])[:top if top > 0 else len(synonyms)]

    @staticmethod
    def download_stopwords(languages: Union[str, List[str]], folder: str) -> dict:
        """
        Download the lists of stopwords from nltk library.
        :param languages: The list of languages to download their stopwords.
        :param folder: Where the list will be stored.
        :return: A dictionary where the key is the language code and the value is the set of stopwords.
        """
        stopwords = {}
        for lang in (languages if isinstance(languages, List) else [languages]):
            file = sep.join([folder, 'corpora', 'stopwords', lang_name[lang]])
            if not exists(file):
                log.info(f'Downloading the stopwords for the {lang_name[lang]} language.')
                nltk.download('stopwords', folder)
                log.info(f'Language downloaded successfully.')
            with open(file) as stream:
                stopwords[lang] = {term[:-1] for term in stream}

        return stopwords

    @staticmethod
    def download_embeddings(languages: Union[str, List[str]], folder: str) -> dict:
        """
        Download the embeddings models from Polyglot library.
        :param languages: The list of languages to download their embeddings' models.
        :param folder: Where the models will be stored.
        :return: A dictionary where the key is the language code and the value is the word embedding for that language.
        """
        embeddings = {}
        for lang in (languages if isinstance(languages, List) else [languages]):
            file = sep.join([folder, 'embeddings2', lang, EMBEDDINGS_FILE])
            if not exists(file):
                log.info(f'Downloading the word embeddings for {lang_name[lang].title()} language...')
                downloader.download_life(f'embeddings2.{lang}', folder)
                log.info(f'Model downloaded successfully.')
            log.info(f'Loading the word embeddings for {lang_name[lang].title()} language.')
            embeddings[lang] = Embedding.load(file)
            log.info(f'Word embedding for {lang_name[lang].title()} language loaded successfully.')
        return embeddings
