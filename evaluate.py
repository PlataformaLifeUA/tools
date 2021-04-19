from argparse import ArgumentParser
from typing import List, Any, Tuple

from filedatasource import open_writer, CsvReader
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from scipy.sparse import lil_matrix
from tqdm import tqdm

from corpus import divide_corpus, LifeCorpus
from eval import metrics
from preproc import preprocess_corpus
from sklearn_train_old_ import create_ml
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


def add_term(sample: dict, term: str, weight: float = 1):
    if term not in sample:
        sample[term] = 0
    sample[term] += 1 * weight


class Evaluator(object):

    def __init__(self, fname: str, embeddings: int = 10, threshold: float = 0.8, train_size: int = 0.9,
                 cl: str = 'SVM', lang: str = 'en') -> None:
        corpus = LifeCorpus(fname)
        self.embeddings = WordEmbeddings(lang, neighbors=embeddings, threshold=threshold)
        self.lang = lang
        self.__ml = self.__train(corpus, train_size, cl)

    def __train(self, corpus: LifeCorpus, test_size: int, classifier: str) -> Any:
        train_corpus, test_corpus, y_train, y_test = divide_corpus(corpus, test_size)
        self.dictionary = Dictionary(train_corpus)
        X_train = self.__corpus2matrix(train_corpus)

        return create_ml(X_train, y_train, classifier)

    def __corpus2matrix(self, corpus: List[List[str]]) -> lil_matrix:
        corpus = self.terms2freq(corpus)
        corpus = self.embedding_expansion(corpus)
        bow_corpus = self.corpus2bow(corpus)
        self.tfidf = TfidfModel(bow_corpus)
        tfidf_corpus = self.bow2tfidf(bow_corpus)
        return self.features2matrix(tfidf_corpus)

    def corpus2bow(self, corpus: List[List[Tuple[str, float]]]) -> List[List[Tuple[int, float]]]:
        token2id = self.dictionary.token2id
        bow_corpus = []
        for sample in corpus:
            bow_corpus.append([(token2id[term], weight) for term, weight in sample if term in token2id])
        return bow_corpus

    def predict(self, text: str) -> Any:
        samples = preprocess_corpus([text])
        X_test = self.__corpus2matrix(samples)
        return self.__ml.predict(X_test[0])[0]

    def terms2freq(self, corpus: List[List[str]]) -> List[List[Tuple[str, int]]]:
        result = []
        for sample in corpus:
            freq = {}
            for term in sample:
                add_term(freq, term)

            result.append([(term, freq) for term, freq in freq.items()])

        return result

    def embedding_expansion(self, corpus: List[List[Tuple[str, int]]]) -> List[List[Tuple[str, float]]]:
        token2id = self.dictionary.token2id
        expanded_corpus = []
        for sample in corpus:
            sample_freq = {}
            for term, freq in sample:
                add_term(sample_freq, term, freq)
                for synonym, weight in self.embeddings.synonyms(term, self.lang):
                    add_term(sample_freq, synonym, weight)

            expanded_corpus.append([(term, freq) for term, freq in sample_freq.items() if term in token2id])

        return expanded_corpus

    def bow2tfidf(self, corpus: List[List[Tuple[int, float]]]) -> List[List[Tuple[int, float]]]:
        tfidf_corpus = []
        for sample in corpus:
            tfidf_corpus.append(self.tfidf[sample])
        return tfidf_corpus

    def features2matrix(self, corpus: List[List[Tuple[int, float]]]) -> lil_matrix:
        M = lil_matrix((len(corpus), len(self.dictionary)))
        for i, sample in enumerate(corpus):
            for token in sample:
                M[i, token[0]] = token[1]
        return M


def main() -> None:
    args = EvaluateArgParser()
    evaluate(args.corpus, args.files, args.output)


def evaluate(corpus: str, files: List[str], output: str):
    eval = Evaluator(corpus)
    with open_writer(output, ['Source', 'Sample', 'Class', 'Prediction']) as writer:
        for corpus in files:
            with CsvReader(corpus) as reader:
                y_test, y_pred = [], []
                for row in tqdm(reader, desc=f'Predicting for {corpus}'):
                    text = row.text if 'text' in reader.fieldnames else row.Text
                    prediction = eval.predict(text)
                    cls = row.cls if 'cls' in reader.fieldnames else row.Alert_level
                    writer.write_row(Source=corpus, Sample=text, Class=cls, Prediction=prediction)
                    y_test.append(0 if cls.lower() not in ['no risk', 'no_risk'] else 1)
                    y_pred.append(prediction)
                print(metrics(y_test, y_pred))


if __name__ == '__main__':
    main()
