import json
from random import randint
from typing import List, Any, Dict

from corpus import LifeCorpus
from eval import metrics, print_metrics
from rasa_wrapper import RasaWrapper
from train import increase_measures, div_measures


def invert_dict(dictionary: Dict[Any, Any]) -> Dict[Any, Any]:
    d = {}
    for rule in dictionary:
        for cls in dictionary[rule]:
            d[cls] = rule
    return d


def reduce_classes(y_train: List[str], rules: Dict[str, List[str]]) -> List[str]:
    d = invert_dict(rules)
    return [d[cls] for cls in y_train]


def divide_corpus(corpus:  LifeCorpus, ratio: float = 0.9):
    train_size = int(len(corpus.samples) * ratio)
    y_train = corpus.alert_level.copy()
    train_corpus = [sample for sample in corpus.samples]
    test_corpus = []
    y_test = []
    while len(train_corpus) > train_size:
        i = randint(0, len(train_corpus) - 1)
        test_corpus.append(train_corpus[i])
        y_test.append(y_train[i])
        del train_corpus[i]
        del y_train[i]

    return train_corpus, test_corpus, y_train, y_test


def cross_validation(corpus:  LifeCorpus, folders: int = 10):
    sum_measures = {}
    for i in range(folders):
        train_corpus, test_corpus, y_train, y_test = divide_corpus(corpus, 1 - 100 / (folders * 100))
        y_train = reduce_classes(y_train, {'No risk': ['No risk'], 'Risk': ['Possible', 'Risk', 'Urgent', 'Immediate']})
        y_test = reduce_classes(y_test, {'No risk': ['No risk'], 'Risk': ['Possible', 'Risk', 'Urgent', 'Immediate']})
        ml = RasaWrapper('data/rasa')
        ml.train(train_corpus, y_train)
        y_pred = [x[0] for x in ml.evaluate(test_corpus)]
        y_test = [0 if x == 'No risk' else 1 for x in y_test]
        y_pred = [0 if x == 'no_risk' else 1 for x in y_pred]
        measures = metrics(y_test, y_pred)
        increase_measures(sum_measures, measures)

    return div_measures(sum_measures, folders)


def main() -> None:
    corpus = LifeCorpus()
    measures = []
    for i in range(30):
        measures.append(cross_validation(corpus))
        print_metrics(measures[-1])
    with open('results/rasa_evaluation.json', 'wt') as file:
        json.dump(measures, file)


if __name__ == '__main__':
    main()