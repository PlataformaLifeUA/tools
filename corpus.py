import csv
from typing import List, Tuple, Dict, Any

from gensim.corpora import Dictionary
from pandas import DataFrame
import requests
from scipy.sparse import csc_matrix
from tqdm import tqdm
from lxml import html, etree

URL = 'https://github.com/PlataformaLifeUA/corpus'


def corpus2bow(corpus: List[List[str]], dictionary: Dictionary) -> List[List[Tuple[int, int]]]:
    bow_corpus = []
    for sample in tqdm(corpus, desc='Obtaining the corpus BoW'):
        bow_corpus.append(dictionary.doc2bow(sample))
    return bow_corpus


def features2matrix(samples: List[List[Tuple[int, Any]]], dictionary: Dictionary) -> csc_matrix:
    M = csc_matrix((len(samples), len(dictionary)))
    for i, sample in enumerate(samples):
        for token in sample:
            M[i, token[0]] = token[1]
    return M


def corpus2matrix(corpus: List[List[str]], dictionary: Dictionary) -> csc_matrix:
    """
    Convert a corpus to a sparse matrix.
    :param corpus: The corpus which contains FeaturedSample objects.
    :return: A sparse matrix which each row represents a sample and each column the each feature of that sample.
    """
    return features2matrix(corpus2bow(corpus, dictionary), dictionary)


def get_feature_text(doc, feature_name: str) -> str:
    values = []
    for feature in doc.xpath('//AnnotationSet[@Name="consensus"]/Annotation/Feature'):
        if feature.xpath('Name/text()')[0] == feature_name:
            values.append(feature.xpath('Value/text()')[0])
    return ','.join(values)


def get_doc_text(doc) -> str:
    return ''.join(doc.xpath('//TextWithNodes/text()'))


def download() -> Dict[str, List[str]]:
    corpus = {
        'Language': [],
        'Text': [],
        'Alert level': [],
        'Message types': []
    }
    page = requests.get(URL)
    doc = html.fromstring(page.content)
    for url in tqdm(doc.xpath('//div[@role="row"]/div[2]//a/@href'), desc='Downloading Life corpus'):
        if 'LICENSE' not in url and 'README.md' not in url:
            url = 'https://raw.githubusercontent.com' + url.replace('blob/', '')
            page = requests.get(url)
            doc = etree.fromstring(page.content)
            text = get_doc_text(doc)
            language = get_feature_text(doc, 'Language')
            alert_level = get_feature_text(doc, 'AlertLevel')
            msg_types = ', '.join([get_feature_text(doc, 'PrimaryMessageType'),
                                      get_feature_text(doc, 'SecondaryMessageType')])
            corpus['Language'].append(language)
            corpus['Text'].append(text)
            corpus['Alert level'].append(alert_level)
            corpus['Message types'].append(msg_types)

    return corpus


def save_corpus(corpus, fname: str, encoding: str = 'utf-8') -> None:
    df = DataFrame(corpus)
    df.to_csv(fname, encoding=encoding, index=False)


def load_life_corpus(fname: str, encoding: str = 'utf-8') -> Dict[str, List[str]]:
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
