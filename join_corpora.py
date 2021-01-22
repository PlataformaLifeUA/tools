import csv
from argparse import ArgumentParser
from typing import List, Dict

from tqdm import tqdm

DEF_GOLD_CORPUS = 'data/life_corpus_en.csv'
DEF_REDDIT_CORPUS = 'data/reddit_evaluation.csv'
AGREE, ALL = 'AGREE', 'ALL'
JOIN_TYPES = [AGREE, ALL]


class ArgParser(object):
    @property
    def gold_standard(self) -> str:
        return self._args.gold

    @property
    def reddit(self) -> str:
        return self._args.reddit

    @property
    def mode(self) -> str:
        return self._args.mode

    @property
    def output(self) -> str:
        return self._args.output

    def __init__(self):
        parser = ArgumentParser(description='Generate a corpus from goldstandard and reddit corpora.')
        parser.add_argument('-g', '--gold', metavar='FILE', type=str,
                            help=f'The path to the gold-standard corpus.'
                                 f'If it is not given, then the gold standard corpus is not used.')
        parser.add_argument('-r', '--reddit', metavar='FILE', type=str, default=DEF_REDDIT_CORPUS,
                            help=f'The path to the annotated reddit corpus. By default {DEF_REDDIT_CORPUS}')
        parser.add_argument('-m', '--mode', metavar='MODE', type=str, choices=JOIN_TYPES, default=AGREE,
                            help=f'Mode of corpus join. By default {AGREE}. This is all available values:'
                                 f'\n  - {AGREE} for only take into account the agreement classes for all evalutaros. '
                                 f'\n  - {ALL} for only take into account the agreement classes for all evalutaros. ')
        parser.add_argument('-o', '--output', metavar='FILE', type=str, required=True,
                            help='The output file with the corpus joined.')

        self._args = parser.parse_args()


def load_reddit_corpus(fname: str) -> List[Dict[str, List[str]]]:
    corpus = {}
    with open(fname, 'rt') as file:
        reader = csv.DictReader(file)
        for row in reader:
            text, cls = row['text'], row['cls']
            if text in corpus:
                corpus[text].append(cls)
            else:
                corpus[text] = [cls]
    return [{'text': text, 'cls': cls} for text, cls in corpus.items()]


def join_corpus(reddit: List[Dict[str, List[str]]], gold: List[Dict[str, List[str]]], mode: str) -> List[Dict[str, List[str]]]:
    corpus = gold.copy()
    for sample in reddit:
        if mode.upper() == ALL:
            corpus.append(sample)
        elif mode.upper() == AGREE:
            classes = sample['cls']
            if all(cls == classes[0] for cls in classes):
                corpus.append({'text': sample, 'cls': [classes[0]]})
        else:
            raise ValueError(f'The mode "{mode}" is not allowed, only {ALL} or {AGREE}.')

    return corpus


def save_corpus(corpus: List[Dict[str, List[str]]], fname: str) -> None:
    with open(fname, 'wt') as file:
        writer = csv.DictWriter(file, fieldnames=corpus[0].keys())
        writer.writeheader()
        for sample in tqdm(corpus, desc='Writing the corrected evaluations'):
            writer.writerow({
                'text': sample['text'],
                'cls': ';'.join(sample['cls'])
            })


def main(args: ArgParser):
    gold = load_gold_standard(args.gold_standard) if args.gold_standard else []
    reddit = load_reddit_corpus(args.reddit)
    corpus = join_corpus(reddit, gold, args.mode)
    save_corpus(corpus, args.output)


def load_gold_standard(fname: str) -> List[Dict[str, List[str]]]:
    # corpus = []
    with open(fname, 'rt') as file:
        reader = csv.DictReader(file)
        return [{
            'text': row['Text'],
            'cls': ['No risk' if row['Alert level'] == 'No risk' else 'Risk']} for row in reader]
    #     for row in reader:
    #         corpus.append({
    #             'text': row['Text'],
    #             'cls': row['Alert level']
    #         })
    # return corpus


if __name__ == '__main__':
    main(ArgParser())
