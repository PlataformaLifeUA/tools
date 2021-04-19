import json
from argparse import ArgumentParser
from random import randint
from typing import List, Any, Dict

from corpus import LifeCorpus
from eval import metrics, print_metrics
from rasa_wrapper import RasaWrapper
from sklearn_train_old_ import increase_measures, div_measures


class ArgParser(object):
    @property
    def file(self) -> str:
        return self._args.file

    @property
    def output(self) -> str:
        return self._args.output

    @property
    def repetitions(self) -> int:
        return self._args.repetitions

    def __init__(self):
        parser = ArgumentParser(description='Train with cross validation.')
        parser.add_argument('-f', '--file', metavar='FILE', type=str, required=True,
                            help='The file path to the corpus.')
        parser.add_argument('-o', '--output', metavar='FILE', type=str, required=True,
                            help='The file with the results.')
        parser.add_argument('-r', '--repetitions', metavar='NUM', type=int, default=30,
                            help='How many time the experiments are repeated.')
        self._args = parser.parse_args()

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
    args = ArgParser()
    corpus = LifeCorpus(args.file)
    measures = []
    for i in range(args.repetitions):
        measures.append(cross_validation(corpus))
        print_metrics(measures[-1])
    with open(args.output, 'wt') as file:
        json.dump(measures, file)


if __name__ == '__main__':
    main()
