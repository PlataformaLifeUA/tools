import csv
from abc import ABC, ABCMeta, abstractmethod
from random import randint
from typing import List, Tuple, Dict, Any

from gensim.corpora import Dictionary
from gensim import models
from googletrans import Translator
from scipy.sparse import lil_matrix
from tqdm import tqdm
from os import path, mkdir

from downloader import Downloader
from preproc import preprocess_corpus
from utils import save_csv
from wemb import WordEmbeddings

GOLD_STANDARD_CORPUS = 'life_corpus.csv'
REDDIT_CORPUS = 'reddit_messages.csv'


class Corpus(ABC):
    __metadata__ = ABCMeta

    @property
    @abstractmethod
    def corpus(self):
        pass

    @property
    def folder(self) -> str:
        return self.__folder

    def __init__(self, folder: str = 'data'):
        if not path.exists(folder):
            mkdir(folder)
        self.__folder = folder

    def __iter__(self):
        return self.corpus.__iter__()

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus[item]

    def __delitem__(self, key):
        del self.corpus[key]


class AnnotatedCorpus(Corpus, ABC):
    __metadata__ = ABCMeta


class LifeCorpus(AnnotatedCorpus):
    @property
    def corpus(self):
        return self.__corpus

    @property
    def fname(self) -> str:
        return self.__gold_file.replace('.csv', f'_{self.lang}.csv')

    def __init__(self, folder: str = 'data', lang: str = 'en'):
        super().__init__(folder)
        self.lang = lang
        self.__gold_file = path.join(self.folder, GOLD_STANDARD_CORPUS)
        if not path.exists(self.__gold_file):
            self.download()
        if not path.exists(self.__gold_file):
            self.__corpus = self.__load(self.__gold_file)
            self.translate('en')
            self.save()
        self.__corpus = self.__load(self.fname)

    def save(self, fname: str = None, encoding: str = 'utf-8') -> None:
        save_csv(self.__corpus, fname if fname else self.fname, encoding)

    def load(self, fname: str = None, encoding: str = 'utf-8'):
        return self.__load(fname if fname else self.fname, encoding)

    @staticmethod
    def __load(fname: str, encoding: str = 'utf-8') -> Dict[str, List[str]]:
        corpus = {
            'Language': [],
            'Text': [],
            'Alert level': [],
            'Message types': []
        }
        with open(fname, "r", encoding=encoding) as File:
            reader = csv.DictReader(File)
            for row in tqdm(reader, desc='Loading the corpus'):
                corpus['Language'].append(row['Language'])
                corpus['Text'].append(row['Text'])
                corpus['Alert level'].append(row['Alert level'])
                corpus['Message types'].append(row['Message types'])

        return corpus

    def download(self):
        Downloader().download_gold_standard(path.join(self.folder, GOLD_STANDARD_CORPUS))

    def translate(self, dest: str) -> None:
        translator = Translator()
        languages = self.__corpus['Language']
        texts = self.__corpus['Text']
        for i in tqdm(range(len(texts)), desc='Translating the corpus.'):
            src = languages[i].lower()
            if src != dest.lower():
                texts[i] = translator.translate(texts[i], dest=dest, src=src).text
                languages[i] = dest

    @property
    def samples(self):
        return self.__corpus['Text']

    @property
    def alert_level(self):
        return self.__corpus['Alert level']


class RedditCorpus(Corpus):

    @property
    def corpus(self):
        return self.texts

    def __init__(self, folder: str = 'data'):
        super().__init__(folder)
        if not path.exists('data/reddit_messages.csv'):
            self.download()
        self.texts = self.load('data/reddit_messages.csv')

    def download(self):
        Downloader.download_reddit_corpus(path.join(self.folder, REDDIT_CORPUS))

    @staticmethod
    def load(fname: str, encoding: str = 'utf-8') -> List[str]:
        texts = []
        with open(fname, "r", encoding=encoding) as file:
            reader = csv.DictReader(file)
            for row in tqdm(reader, desc='Loading the corpus'):
                title = row['title']
                body = row['body']
                texts.append('\n'.join([title, body]))

        return texts


def expand_wordembedding(sample: List[Tuple[int, int]], dictionary: Dictionary, embedings: WordEmbeddings = None,
                         lang: str = 'en', fixed: bool = False) -> List[Tuple[int, float]]:
    if not embedings:
        return sample
    sample_dict = {}
    for id, freq in sample:
        sample_dict[id] = freq
        term = dictionary[id]
        synonyms = embedings.synonyms(term, lang)
        if not fixed:
            dictionary.add_documents([[t[0] for t in synonyms]])
        for synonym, w in synonyms:
            if synonym in dictionary.token2id:
                synonym_id = dictionary.token2id[synonym]
                sample_dict[synonym_id] = sample_dict[synonym_id] + w * freq if synonym_id in sample_dict else w * freq

    return [(term, freq) for term, freq in sample_dict.items()]

def corpus2bow(corpus: List[List[str]], dictionary: Dictionary, embedings: WordEmbeddings = None,
               lang: str = 'en', fixed: bool = False) -> List[List[Tuple[int, float]]]:
    bow_corpus = []
    for sample in tqdm(corpus, desc='Obtaining the corpus BoW'):
        bow_corpus.append(expand_wordembedding(dictionary.doc2bow(sample), dictionary, embedings, lang, fixed))
    return bow_corpus


def features2matrix(samples: List[List[Tuple[int, Any]]], dictionary: Dictionary) -> lil_matrix:
    M = lil_matrix((len(samples), len(dictionary)))
    for i, sample in enumerate(samples):
        for token in sample:
            M[i, token[0]] = token[1]
    return M


def bow2tfidf(bow_corpus):
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_corpus = []
    for doc in bow_corpus:
        tfidf_corpus.append(tfidf[doc])
    return tfidf_corpus


def corpus2matrix(corpus: List[List[str]], dictionary: Dictionary, method: str = 'TF/IDF',
                  embedings: WordEmbeddings = None, lang: str = 'en', fixed: bool = False) -> lil_matrix:
    """
    Convert a corpus to a sparse matrix.
    :param corpus: The corpus which contains FeaturedSample objects.
    :return: A sparse matrix which each row represents a sample and each column the each feature of that sample.
    """
    bow_corpus = corpus2bow(corpus, dictionary, embedings, lang, fixed)
    if method == 'BoW':
        return features2matrix(bow_corpus, dictionary), bow_corpus, None
    tfidf_corpus = bow2tfidf(bow_corpus)
    if method == 'TF/IDF':
        return features2matrix(tfidf_corpus, dictionary), bow_corpus, tfidf_corpus
    # lsi_corpus = ...
    # lda_corpus = ...
    return features2matrix(bow_corpus, dictionary), bow_corpus, None


def divide_corpus(corpus:  LifeCorpus, ratio: float = 0.9):
    train_corpus = preprocess_corpus(corpus.samples)
    y_train = [0 if cls.lower() == 'no risk' else 1 for cls in corpus.alert_level]
    train_size = int(len(train_corpus) * ratio)
    test_size = len(train_corpus) - train_size
    # Generate a random index from 0 to len(train_corpus)
    test_corpus = []
    y_test = []
    while len(train_corpus) > train_size:
        i = randint(0, len(train_corpus) - 1)
        test_corpus.append(train_corpus[i])
        y_test.append(y_train[i])
        del train_corpus[i]
        del y_train[i]

    return train_corpus, test_corpus, y_train, y_test
