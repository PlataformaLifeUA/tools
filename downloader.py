from typing import Union, Dict, List
from os import path
from lxml import html, etree
import requests
from tqdm import tqdm

from utils import save, save_csv

GOLD_STANDARD_URL = 'https://github.com/PlataformaLifeUA/corpus'
REDDIT_CORPUS_URL = 'https://raw.githubusercontent.com/PlataformaLifeUA/corpus/master/reddit/reddit_messages.csv'


def download(url: str, fname: str = None, encoding: str = 'utf-8') -> Union[str, bytes]:
    page = requests.get(url)
    if fname:
        save(fname, page.content.decode(encoding) if encoding else page.content, encoding)

    return page.content.decode(encoding) if encoding else page.content


class Downloader(object):
    @staticmethod
    def download_gold_standard(fname: str) -> None:
        corpus = {
            'Language': [],
            'Text': [],
            'Alert level': [],
            'Message types': []
        }

        doc = html.fromstring(requests.get(GOLD_STANDARD_URL).content)
        for url in tqdm(doc.xpath('//div[@role="row"]/div[2]//a/@href'), desc='Downloading Life corpus'):
            if url.endswith('.xml'):
                url = 'https://raw.githubusercontent.com' + url.replace('blob/', '')
                doc = etree.fromstring(requests.get(url).content)
                text = Downloader.__get_doc_text(doc)
                language = Downloader.__get_feature_text(doc, 'Language')
                alert_level = Downloader.__get_feature_text(doc, 'AlertLevel')
                msg_types = ', '.join([Downloader.__get_feature_text(doc, 'PrimaryMessageType'),
                                       Downloader.__get_feature_text(doc, 'SecondaryMessageType')])
                corpus['Language'].append(language)
                corpus['Text'].append(text)
                corpus['Alert level'].append(alert_level)
                corpus['Message types'].append(msg_types)

            save_csv(corpus, fname)

    @staticmethod
    def __get_feature_text(doc, feature_name: str) -> str:
        values = []
        for feature in doc.xpath('//AnnotationSet[@Name="consensus"]/Annotation/Feature'):
            if feature.xpath('Name/text()')[0] == feature_name:
                values.append(feature.xpath('Value/text()')[0])
        return ','.join(values)

    @staticmethod
    def __get_doc_text(doc) -> str:
        return ''.join(doc.xpath('//TextWithNodes/text()'))

    @staticmethod
    def download_reddit_corpus(fname: str):
        download(REDDIT_CORPUS_URL, fname)
