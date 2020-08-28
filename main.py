from os import mkdir
from os.path import exists
from typing import List, Dict

from corpus import download, load_life_corpus, save_corpus, corpus2matrix, divide_corpus
from eval import evaluate, metrics, print_metrics
from preproc import preprocess_corpus, preprocess
from gensim.corpora import Dictionary
import logging
from sklearn.svm import SVC

from utils import translate

log = logging.getLogger(__name__)


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


def cross_validation(corpus:  Dict[str, List[str]], folders: int = 10):
    sum_measures = {}
    for i in range(folders):
        train_corpus, test_corpus, y_train, y_test = divide_corpus(corpus, 1 - 100 / (folders * 100))
        dictionary = Dictionary(train_corpus)
        X_train = corpus2matrix(train_corpus, dictionary)
        X_test = corpus2matrix(test_corpus, dictionary)

        ml = SVC()
        ml.fit(X_train, y_train)

        y_pred = evaluate(X_test, ml)
        measures = metrics(y_test, y_pred)
        increase_measures(sum_measures, measures)

    return div_measures(sum_measures, folders)


def main():
    if not exists('data/'):
        mkdir('data')
    if not exists('data/life_corpus.csv'):
        corpus = download()
        save_corpus(corpus, 'data/life_corpus.csv')
    if not exists('data/life_corpus_en.csv'):
        corpus = load_life_corpus('data/life_corpus.csv')
        corpus = translate(corpus, 'es', 'en')
        save_corpus(corpus, 'data/life_corpus_en.csv')
    else:
        corpus = load_life_corpus('data/life_corpus_en.csv')

    measures = cross_validation(corpus, 10)

    print_metrics(measures)


if __name__ == '__main__':
    main()