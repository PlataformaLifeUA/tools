from argparse import ArgumentParser
from enum import Enum
from typing import Tuple, Dict, List, Any, Union

from pandas import read_excel, read_csv

AGREEMENT = 'agreement'
MAX = 'max'
ANY = 'any'
EVAL_TYPES = [AGREEMENT, MAX, ANY]


class Classes(Enum):
    NO_RISK = 0
    RISK = 1


class EvalType(Enum):
    AGREEMENT = 0
    MAX = 1
    ANY = 2


class ArgParser(object):
    @property
    def evaluations(self) -> str:
        return self.__args.evaluations

    @property
    def results(self) -> str:
        return self.__args.results

    @property
    def type(self) -> EvalType:
        if self.__args.type == AGREEMENT:
            return EvalType.AGREEMENT
        if self.__args.type == MAX:
            return EvalType.MAX
        if self.__args.type == ANY:
            return EvalType.ANY

    @property
    def encoding(self) -> str:
        return self.__args.encoding

    def __init__(self):
        parser = ArgumentParser(description='Feedback evaluator.')
        parser.add_argument('-e', '--evaluations', metavar='FILE', type=str, required=True,
                            help='The file path with the evaluators\' annotations')
        parser.add_argument('-r', '--results', metavar='FILE', type=str, required=True,
                            help='The classification results.')
        parser.add_argument('-t', '--type', metavar='TYPE', type=str, default=AGREEMENT, choices=EVAL_TYPES,
                            help=f'The type of the evaluation.\n'
                                 f'  * {AGREEMENT}: all evaluators must be agree.\n'
                                 f'  * {MAX}: in case of conflict, those class with more confidence.\n'
                                 f'  * {ANY}: in case of conflict, all classes are accepted.')
        parser.add_argument('--encoding', metavar='FILE', type=str, default='utf-8',
                            help='The encoding to read the CSV file with the results. By default utf-8.')
        self.__args = parser.parse_args()


class Evaluators(object):
    def __init__(self, fname: str, type: EvalType = EvalType.AGREEMENT):
        self.type = type
        self.evaluations = {}
        self.texts = {}
        self.iterations = []
        df = read_excel(fname)
        for row in range(len(df)):
            eval_id, it, conf, text = df['Evaluator'][row], df['Iteration'][row], df['Confidence'][row], df['Text'][row]
            cl = Classes.RISK if df['Classification'][row] == 'Risk' else Classes.NO_RISK
            if eval_id not in self.evaluations:
                self.evaluations[eval_id] = {}
            self.evaluations[eval_id][it] = (conf, text, cl)
            if text not in self.texts:
                self.texts[text] = []
            self.texts[text].append((eval_id, it, conf, cl))
            while it >= len(self.iterations):
                self.iterations.append([])
            self.iterations[it - 1].append((eval_id, it, conf, text, cl))

    def evaluate(self, results: Dict[str, Tuple[int, float, float]]) -> Dict[Any, List[List[int]]]:
        measures = {'total': [[0, 0], [0, 0]]}
        for text, (step, no_risk, risk) in results.items():
            if text in self:
                cl = Classes.NO_RISK.value if no_risk > risk else Classes.RISK.value
                if step not in measures:
                    measures[step] = [[0, 0], [0, 0]]
                eval_classes = self[text]
                for eval_cl in eval_classes:
                    measures[step][cl][eval_cl] += 1
                    measures['total'][cl][eval_cl] += 1

        return measures

    def __getitem__(self, item) -> List[int]:
        evaluations = self.texts[item]
        if self.type == EvalType.AGREEMENT:
            return self.__agreement(evaluations)
        if self.type == EvalType.MAX:
            return self.__max(evaluations)
        if self.type == EvalType.ANY:
            return self.__any(evaluations)

    def __contains__(self, item) -> bool:
        try:
            return bool(self[item])
        except KeyError:
            return False

    @staticmethod
    def __any(evaluations: List[Tuple[str, float, int]]) -> List[int]:
        result = []
        for eval in evaluations:
            _, _, cl = eval
            if cl not in result:
                result.append(cl)
        return result

    @staticmethod
    def __max(evaluations: List[Tuple[str, float, int]]):
        result = max(evaluations, key=lambda x: x[1])

        return [result[2]]

    @staticmethod
    def __agreement(evaluations):
        result = []
        for eval in evaluations:
            _, _, cl = eval
            if result:
                if cl not in result:
                    return []
            else:
                result.append(cl)
        return result


def read_results(fname: str, encoding: str = 'utf-8') -> Dict[str, Tuple[int, float, float]]:
    results = {}
    df = read_csv(fname, encoding=encoding)
    for row in range(len(df)):
        results[df['Text'][row]] = (df['Iteration'][row], df['No risk confidence'][row], df['Risk confidence'][row])

    return results


def main(args: ArgParser):
    eval = Evaluators(args.evaluations)
    results = read_results(args.results)
    measures = eval.evaluate(results)
    print(measures)


if __name__ == '__main__':
    main(ArgParser())