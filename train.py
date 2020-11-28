import csv
from os import mkdir
from os.path import exists
from typing import List, Dict

from gensim import models
from tqdm import tqdm

from corpus import download_life, load_life_corpus, save_corpus, corpus2matrix, divide_corpus, corpus2bow, \
    features2matrix, \
    expand_wordembedding, download
from eval import evaluate, metrics, print_metrics
from preproc import preprocess_corpus, preprocess
from gensim.corpora import Dictionary
import logging
from sklearn.svm import SVC

from utils import translate
from wemb import WordEmbeddings

log = logging.getLogger(__name__)
REDDIT_CORPUS = 'https://raw.githubusercontent.com/PlataformaLifeUA/corpus/master/reddit/reddit_messages.csv'


def increase_measures(total_measures: dict, measures: dict):
    for key, value in measures.items():
        if key in total_measures:
            total_measures[key] += value
        else:
            total_measures[key] = value


def div_measures(total_measures: dict, divisor: int) -> dict:
    measures = {}
    for key, value in total_measures.items():
        measures[key] = value / divisor
    return measures


def cross_validation(corpus:  Dict[str, List[str]], folders: int = 10, embedings: WordEmbeddings = None,
                     lang: str = 'en'):
    sum_measures = {}
    for i in range(folders):
        train_corpus, test_corpus, y_train, y_test = divide_corpus(corpus, 1 - 100 / (folders * 100))
        dictionary = Dictionary(train_corpus)
        ml = create_ml(dictionary, train_corpus, y_train, 'TF/IDF', embedings, lang)

        X_test = corpus2matrix(test_corpus, dictionary, 'TF/IDF', embedings, lang, True)

        y_pred = evaluate(X_test, ml)
        measures = metrics(y_test, y_pred)
        increase_measures(sum_measures, measures)

    return div_measures(sum_measures, folders)


def create_ml(dictionary, corpus, y_train, text_model = 'TF/IDF', embedings: WordEmbeddings = None, lang: str = 'en'):
    X_train = corpus2matrix(corpus, dictionary, text_model, embedings, lang)
    ml = SVC(probability=True)
    ml.fit(X_train, y_train)
    return ml


def load_reddit_corpus(fname: str, encoding: str = 'utf-8') -> List[str]:
    texts = []
    with open(fname, "r", encoding=encoding) as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader, desc='Loading the corpus'):
            title = row['title']
            body = row['body']
            texts.append('\n'.join([title, body]))

    return texts


def main():
    if not exists('data/'):
        mkdir('data')
    if not exists('data/life_corpus.csv'):
        corpus = download_life()
        save_corpus(corpus, 'data/life_corpus.csv')
    if not exists('data/life_corpus_en.csv'):
        corpus = load_life_corpus('data/life_corpus.csv')
        corpus = translate(corpus, 'es', 'en')
        save_corpus(corpus, 'data/life_corpus_en.csv')
    else:
        corpus = load_life_corpus('data/life_corpus_en.csv')

    embedings = WordEmbeddings('en', 'data', 1000)

    measures = cross_validation(corpus, 10, embedings)
    print_metrics(measures)
    if not exists('data/reddit_messages.csv'):
        download(REDDIT_CORPUS, 'data/reedit_messages.csv')
    texts = load_reddit_corpus('data/reedit_messages.csv')
    results = []
    finish = False
    it = 1
    while not finish:
        detected = detect_suicide_messages(corpus, texts, 0.2 ** it, embedings)
        if detected:
            for i, confidence in reversed(detected):
                text = texts[i]
                del texts[i]
                corpus['Language'].append('en')
                corpus['Text'].append(text)
                corpus['Alert level'].append('Urgent')
                corpus['Message types'].append('Undefined')
                results.append((it, confidence, text))
        else:
            finish = True
        it += 1

    with open('data/result.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Confidence', 'Text'])
        for result in results:
            writer.writerow(result)


def detect_suicide_messages(corpus: Dict[str, List[str]], texts: List[str], confidence: float,
                            embedings: WordEmbeddings, lang: str = 'en') -> List[int]:
    train_corpus, _, y_train, _ = divide_corpus(corpus, 1)
    dictionary = Dictionary(train_corpus)
    ml = create_ml(dictionary, train_corpus, y_train, 'TF/IDF', embedings, lang)
    bow_corpus = corpus2bow(train_corpus, dictionary, embedings, lang)
    tfidf = models.TfidfModel(bow_corpus)
    detected = []
    for i, text in enumerate(texts):
        sample = preprocess(text)
        bow_sample = expand_wordembedding(dictionary.doc2bow(sample), dictionary, embedings, lang, True)
        tfidf_sample = tfidf[bow_sample]
        X = features2matrix([tfidf_sample], dictionary)
        y = ml.predict_proba(X[0])
        if y[0][0] <= confidence:
            detected.append((i, y[0][1]))
    return detected


if __name__ == '__main__':
    main()
