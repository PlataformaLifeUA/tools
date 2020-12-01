from typing import List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import stem
import re
import csv
from bs4 import BeautifulSoup
from unidecode import unidecode
from pprint import pprint  # pretty-printer
from collections import defaultdict
import spacy
from word2number import w2n
import logging
from tqdm import tqdm
from spacy import data

nlp = spacy.load("en_core_web_md")
stop_words = set(stopwords.words('english'))


def expand_contractions(text: str) -> str:
    """
    Expand contractions.

    :param text: the text with the contractions.
    :return: the text with the expanded contractions.
    """
    # specific
    text = re.sub(r"\ '", "'", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\’", "'", text)
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"a\'t", "can not", text)
    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text


def strip_html_tags(text: str) -> str:
    """
    Removing strip html tags.
    :param text: A text with HTML tags.
    :return: The same text without those HTML togs.
    """
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_accented_chars(text: str) -> str:
    """
    Convert accented characters into ASCII characters. It works mainly for latin languages.
    :param text: The text with accented characters.
    :return: The same text with with unicode representation of the accented characters.
    """
    return unidecode(text)


def remove_special_characters(text: str) -> str:
    """
    Remove spcial characters.
    :param text: The text with special characters.
    :return: the cleaned text.
    """
    nstr = re.sub(r'[?|$|.|!]', r'', text)
    nestr = re.sub(r'[^a-zA-Z0-9 ]', r'', nstr)
    return nestr


def removing_words_number(text):
    """
    Eliminar palabras de numeros
    """
    stopwords = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'ten']
    querywords = text.split()

    resultwords = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result


def remove_number(text):
    """
    Eliminar numeros
    """
    output = ''.join(c for c in text if not c.isdigit())
    return output


def removing_stop_words_tokenize(text):
    """
    Removing stop words Tokenización
    """
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def stemmer(text):
    """
    Stemize verbs in list of tokenized words
    """
    lemmatizer = stem.WordNetLemmatizer()
    lemmas = []
    for word in text:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def preprocess(text: str) -> str:
    # Eliminar etiquetas HTML
    text = strip_html_tags(text.lower()).strip()
    # Eliminar acentos
    text = remove_accented_chars(text)
    # Eliminar contracciones
    text = expand_contractions(text)
    # Eliminar caracteres especiales
    text = remove_special_characters(text)
    # Eliminar palabras de numeros
    text = removing_words_number(text)
    # Eliminar numeros
    text = remove_number(text)
    # Eliminar stop_words y tokenizar
    text = removing_stop_words_tokenize(text)
    # Lematización
    text = stemmer(text)

    return text


def load_corpus(fname: str, encoding: str = 'utf-8') -> List[str]:
    """
    Load the corpus from a CSV file.
    :param fname: the path to the CSV file.
    :param encoding: the text encoding.
    :return: the corpus
    """
    corpus = []
    with open(fname, "r", encoding=encoding) as File:
        reader = csv.DictReader(File)
        for row in tqdm(reader, desc='Loading the corpus'):
            title = preprocess(row['title'])
            body = preprocess(row['body'])
            # Join the title and body
            corpus.append(title + body)
    return corpus


def preprocess_corpus(corpus: List[str]) -> List[List[str]]:
    result = []
    for sample in tqdm(corpus, desc='Preprocesing the corpus'):
        result.append(preprocess(sample))
    return result
