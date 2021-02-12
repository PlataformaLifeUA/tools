from argparse import ArgumentParser
from typing import List, Any

from filedatasource import open_writer, CsvReader
from gensim.corpora import Dictionary
from tqdm import tqdm

from corpus import divide_corpus, LifeCorpus, corpus2matrix
from sklearn_train import create_ml
from wemb import WordEmbeddings

DEF_CORPUS = 'data/gold_reddit_corpus_agree.csv'


class EvaluateArgParser(object):
    @property
    def corpus(self) -> str:
        return self._args.corpus

    @property
    def files(self) -> List[str]:
        return self._args.files

    @property
    def output(self) -> str:
        return self._args.output

    def __init__(self) -> None:
        parser = ArgumentParser(description='Evaluate a model')
        parser.add_argument('-c', '--corpus', metavar='FILE', type=str, default=DEF_CORPUS,
                            help='The corpus to train with.')
        parser.add_argument('files', metavar='CORPUS', type=str, nargs='+', help='The corpora to evaluate.')
        parser.add_argument('-o', '--output', metavar='FILE', type=str, required=True,
                            help='The output file. It could be csv, csv.gz, xlsx or xls.')
        self._args = parser.parse_args()


class Evaluator(object):

    def __init__(self, fname: str, embeddings: int = 10, threshold: float = 0.8, train_size: int = 0.9,
                 cl: str = 'SVM', lang: str = 'en') -> None:
        corpus = LifeCorpus(fname)
        self.embeddings = WordEmbeddings(lang, neighbors=embeddings, threshold=threshold)
        self.__ml = self.__train(corpus, embeddings, train_size, lang, cl)
        self.lang = lang

    def __train(self, corpus: LifeCorpus, embeddings: WordEmbeddings, test_size: int, lang: str, cl: str) -> Any:
        train_corpus, test_corpus, y_train, y_test = divide_corpus(corpus, test_size)
        self.dictionary = Dictionary(train_corpus)
        X_train, _, _ = corpus2matrix(train_corpus, self.dictionary, 'TF/IDF', self.embeddings, lang)
        return create_ml(X_train, y_train, cl)

    def predict(self, text: str) -> Any:
        X_train = corpus2matrix([[text]], self.dictionary, 'TF/IDF', self.embeddings, self.lang)
        return self.__ml.predict(X_train[0])[0]


def main() -> None:
    args = EvaluateArgParser()
    evaluate(args.corpus, args.files, args.output)


def evaluate(corpus: str, files: List[str], output: str):
    eval = Evaluator(corpus)
    with open_writer(output, ['Source', 'Sample', 'Class', 'Prediction']) as writer:
        for corpus in files:
            with CsvReader(corpus) as reader:
                for row in tqdm(reader, desc=f'Predicting for {corpus}'):
                    text = row.text if 'text' in reader.fieldnames else row.Text
                    prediction = eval.predict(text)
                    cls = row.cls if 'cls' in reader.fieldnames else row.Alert_level
                    writer.write_row(Source=corpus, Sample=text, Class=cls, Prediction=prediction)


if __name__ == '__main__':
    main()
