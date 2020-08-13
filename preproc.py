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

def expand_contractions(phrase):
    """
    Expandir contracciones.
    """
    # specific
    phrase = re.sub(r"\ '", "'", phrase)
    phrase = re.sub(r"\/", " ", phrase)
    phrase = re.sub(r"\’", "'", phrase)
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"a\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def strip_html_tags(text):
    """
    Removing strip html tgs.
    """
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_accented_chars(text):
    """
    Convierte caracteres acentuados en caracteres ASCII.
    Funciona principalmente para idioma español.
    """
    return unidecode(text)


def remove_special_characters(text):
    """
    Eliminar caracteres especiales
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


def load_corpus(fname: str, encoding: str = 'utf-8') -> List[List[str]]:
    corpus = []
    with open(fname, "r", encoding=encoding) as File:
        reader = csv.DictReader(File)
        for row in tqdm(reader, desc='Loading the corpus'):
            title = preprocess(row['title'])
            body = preprocess(row['body'])
            # Agrega valor al corpus
            corpus.append(title + body)
    return corpus


def preprocess_corpus(corpus: List[str]):
    result = []
    for sample in tqdm(corpus, desc='Preprocesing the corpus.'):
        result.append(preprocess(sample))
    return result