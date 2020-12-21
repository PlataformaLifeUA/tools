import pickle as pk
from logging import getLogger
from argparse import ArgumentParser
from enum import Enum
from os.path import exists
from typing import Tuple, Dict, List, Any
from strsimpy.weighted_levenshtein import WeightedLevenshtein
from pandas import read_excel, read_csv
from tqdm import tqdm

AGREEMENT = 'agreement'
MAX = 'max'
ANY = 'any'
EVAL_TYPES = [AGREEMENT, MAX, ANY]
DEF_BUFFER = '.buffer'
logger = getLogger(__name__)


class Classes(Enum):
    NO_RISK = 0
    RISK = 1

    def __int__(self) -> int:
        return self.value


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
    def __init__(self, fname: str, eval_type: EvalType = EvalType.AGREEMENT, lev_dist: float = 0.1,
                 buffer: bool = True):
        self.type = eval_type
        self.evaluations = {}
        self.texts = {}
        self.iterations = []
        self._lev = WeightedLevenshtein()
        self._lev_dist = lev_dist
        self.__buffer = self.__load_buffer() if buffer else {}
        df = read_excel(fname, engine='openpyxl')
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
        for text, (step, no_risk, risk) in tqdm(results.items(), desc='Evaluating boostrapping'):
            if text in self:
                cl = Classes.NO_RISK.value if no_risk > risk else Classes.RISK.value
                if step not in measures:
                    measures[step] = [[0, 0], [0, 0]]
                eval_classes = self[text]
                for eval_cl in eval_classes:
                    measures[step][cl][int(eval_cl)] += 1
                    measures['total'][cl][int(eval_cl)] += 1

        return measures

    def __getitem__(self, item: str) -> List[int]:
        dist = len(item) * self._lev_dist
        if item in self.__buffer:
            evaluations = self.__buffer[item]
        else:
            evaluations = [self.texts[text] for text in self.texts if self._lev.distance(item, text) < dist]
            if len(evaluations) == 0:
                logger.warning(f'The following text:\n\n{item}\n\nDoes not have any valid answer.')
            elif len(evaluations) > 1:
                raise ValueError(f'The text:\n\n{item}\n\nHas more than one match.')
            else:
                evaluations = evaluations[0]
            self.__buffer[item] = evaluations
            self.__save_buffer(self.__buffer)
        # evaluations = self.texts[item]
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
        for evaluation in evaluations:
            _, _, cl = evaluation
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
        for evaluation in evaluations:
            _, _, _, cl = evaluation
            if result:
                if cl not in result:
                    return []
            else:
                result.append(cl)
        return result

    @staticmethod
    def __load_buffer(fname: str = DEF_BUFFER) -> Dict[str, Tuple]:
        if not exists(fname):
            return {}
        with open(fname, 'rb') as file:
            return pk.load(file)

    @staticmethod
    def __save_buffer(buffer: Dict[str, Tuple], fname: str = DEF_BUFFER):
        with open(fname, 'wb') as file:
            return pk.dump(buffer, file)


def read_results(fname: str, encoding: str = 'utf-8') -> Dict[str, Tuple[int, float, float]]:
    results = {}
    df = read_csv(fname, encoding=encoding)
    for row in range(len(df)):
        results[df['Text'][row]] = (df['Iteration'][row], df['No risk confidence'][row], df['Risk confidence'][row])

    return results


def main(args: ArgParser):
    evaluator = Evaluators(args.evaluations)
    results = read_results(args.results)
    measures = evaluator.evaluate(results)
    print(measures)


if __name__ == '__main__':
    main(ArgParser())
