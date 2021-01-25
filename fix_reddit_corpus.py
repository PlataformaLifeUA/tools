import csv
import pickle
import numpy as np
from os.path import exists

import pandas as pd
from logging import getLogger
from argparse import ArgumentParser
from typing import List, Dict, Union
from strsimpy import WeightedLevenshtein
from tqdm import tqdm

DEF_BUFFER = '.cache'
logger = getLogger(__name__)
lev = WeightedLevenshtein()
if exists(DEF_BUFFER):
    with open(DEF_BUFFER, 'rb') as file:
        cache = pickle.load(file)
else:
    cache = {}


class ArgParser(object):
    @property
    def csv(self) -> str:
        return self._args.csv

    @property
    def output(self) -> str:
        return self._args.output

    # @property
    # def join_method(self) -> str:
    #     return self._args.join

    @property
    def files(self) -> List[str]:
        return self._args.files

    def __init__(self):
        parser = ArgumentParser(description='Join and correct the encoding of the reddit corpus')
        parser.add_argument('-c', '--csv', metavar='FILE', type=str, default='data/reddit_messages.csv',
                            help='The CSV file with the correct reddit messages')
        parser.add_argument('-o', '--output', metavar='FILE', type=str, required=True,
                            help='The output CSV file with the corrected reddit messages')
        # parser.add_argument('-j', '--join', metavar='METHOD', type=str, choices=JOIN_METHODS, default=AGREEMENT,
        #                     help='The method to use to join the evaluation from different evaluators.')
        parser.add_argument('files', metavar='FILE', type=str, nargs='+',
                            help='The Excel files with the human evaluations')
        self._args = parser.parse_args()


def load_reddit(fname: str, encoding: str = 'utf-8') -> List[Dict[str, str]]:
    with open(fname, "rt", encoding=encoding) as file:
        reader = csv.DictReader(file)
        return [
            {'title': row['title'],
             'body': row['body'],
             'text': '\n'.join([row['title'], row['body']])
             } for row in tqdm(reader, desc='Loading the corpus')]


class FormatError(Exception):
    pass


def load_evaluations(files: List[str]) -> List[Dict[str, str]]:
    corpus = []
    for file in tqdm(files, desc='Loading evaluations'):
        df = pd.read_excel(file).replace(np.nan, '', regex=True)
        for i in range(len(df)):
            try:
                if df['Text'][i]:
                    corpus.append({
                        'evaluator': df['Evaluator'][i],
                        'text': df['Text'][i],
                        'cls': df['Classification'][i]
                    })
            except KeyError as e:
                raise FormatError(f'In file "{file}: {e}"')
    return corpus


def is_correct(text: str, reddit: List[Dict[str, str]]):
    for sample in reddit:
        if text == sample['text']:
            return True
    return False


def lev_comparison(text: str, reddit: List[Dict[str, str]], dist: float = 0.05) -> Union[str, None]:
    if text in cache:
        return cache[text]
    max_dist = round(len(text) * dist)
    for sample in reddit:
        sample_text = sample['text']
        if len(sample_text) - len(text) < max_dist:
            # if lev.distance(text[:50], sample_text[:50]) < 5:
            lev_dist = lev.distance(text, sample_text)
            if lev_dist < max_dist:
                cache[text] = sample['text']
                with open(DEF_BUFFER, 'wb') as file:
                    pickle.dump(cache, file)
                return sample_text
    return None


def correct_texts(reddit: List[Dict[str, str]], evaluations: List[Dict[str, str]], dist: float = 0.05) -> None:
    for evaluation in tqdm(evaluations, desc='Correcting the evaluation texts.'):
        text = evaluation['text']
        if not is_correct(text, reddit):
            text = lev_comparison(text, reddit, dist)
            if text:
                evaluation['text'] = text
            else:
                logger.warning(f'The text "{evaluation["text"]}" does not match.')


def save_reddit(evaluations: List[Dict[str, str]], fname: str) -> None:
    if evaluations:
        with open(fname, 'wt') as file:
            writer = csv.DictWriter(file, fieldnames=evaluations[0].keys())
            writer.writeheader()
            for evaluation in tqdm(evaluations, desc='Writing the corrected evaluations'):
                writer.writerow(evaluation)
    else:
        logger.warning('The evaluation results are empty. Nothing to save.')


def main(args: ArgParser) -> None:
    reddit = load_reddit(args.csv)
    evaluations = load_evaluations(args.files)
    correct_texts(reddit, evaluations, 0.1)
    save_reddit(evaluations, args.output)


if __name__ == '__main__':
    main(ArgParser())
