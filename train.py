import csv
from os import mkdir, path
from typing import List, Dict, Tuple

from gensim import models
from tqdm import tqdm

from corpus import corpus2matrix, divide_corpus, features2matrix, expand_wordembedding, LifeCorpus, RedditCorpus
from eval import evaluate, metrics, print_metrics
from lifeargparsers import LifeArgParser
from preproc import preprocess
from gensim.corpora import Dictionary
import logging
from sklearn.svm import SVC

from wemb import WordEmbeddings

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


def cross_validation(corpus:  LifeCorpus, folders: int = 10, embedings: WordEmbeddings = None,
                     lang: str = 'en'):
    sum_measures = {}
    for i in range(folders):
        train_corpus, test_corpus, y_train, y_test = divide_corpus(corpus, 1 - 100 / (folders * 100))
        dictionary = Dictionary(train_corpus)
        X_train, _, _ = corpus2matrix(train_corpus, dictionary, 'TF/IDF', embedings, lang)
        ml = create_ml(X_train, y_train)

        X_test, _, _ = corpus2matrix(test_corpus, dictionary, 'TF/IDF', embedings, lang, True)

        y_pred = evaluate(X_test, ml)
        measures = metrics(y_test, y_pred)
        increase_measures(sum_measures, measures)

    return div_measures(sum_measures, folders)


def create_ml(X_train, y_train):
    ml = SVC(probability=True)
    ml.fit(X_train, y_train)
    return ml


def main():
    args = LifeArgParser()
    corpus = LifeCorpus(args.data, args.lang)
    
    embeddings = WordEmbeddings(args.lang, args.data, args.embeddings, args.embedding_threshold)
    if args.evaluate:
        measures = cross_validation(corpus, args.cross_folder, embeddings)
        print_metrics(measures)
    reddit_corpus = RedditCorpus(args.data)
    results = []
    it = 1
    detected = detect_suicide_messages(corpus, reddit_corpus, args.no_risk ** it, args.risk ** it, embeddings)
    while detected:
        for i, no_risk_confidence, risk_confidence in reversed(detected):
            text = reddit_corpus[i]
            del reddit_corpus[i]
            corpus['Language'].append('en')
            corpus['Text'].append(text)
            if risk_confidence > no_risk_confidence:
                corpus['Alert level'].append('Urgent')
            else:
                corpus['Alert level'].append('No risk')
            corpus['Message types'].append('Undefined')
            results.append((it, no_risk_confidence, risk_confidence, text))

        detected = detect_suicide_messages(corpus, reddit_corpus, 0.1 ** it, 0.2 ** it, embeddings)
        it += 1

    with open(args.file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'No risk confidence', 'Risk confidence', 'Text'])
        for result in results:
            writer.writerow(result)


def detect_suicide_messages(corpus: LifeCorpus, texts: RedditCorpus, nr_confidence: float, r_confidence: float,
                            embedings: WordEmbeddings, lang: str = 'en') -> List[Tuple[int, float, float]]:
    train_corpus, _, y_train, _ = divide_corpus(corpus, 1)
    dictionary = Dictionary(train_corpus)
    X_train, bow_corpus, tfidf_corpus = corpus2matrix(train_corpus, dictionary, 'TF/IDF', embedings, lang)
    ml = create_ml(X_train, y_train)
    tfidf = models.TfidfModel(bow_corpus)
    detected = []
    for i, text in tqdm(enumerate(texts), desc='Classifying Reddit corpus', total=len(texts)):
        sample = preprocess(text)
        bow_sample = expand_wordembedding(dictionary.doc2bow(sample), dictionary, embedings, lang, True)
        tfidf_sample = tfidf[bow_sample]
        X = features2matrix([tfidf_sample], dictionary)
        y = ml.predict_proba(X[0])
        if y[0][0] <= r_confidence or y[0][1] <= nr_confidence:
            detected.append((i, y[0][0], y[0][1]))
    return detected


if __name__ == '__main__':
    main()
