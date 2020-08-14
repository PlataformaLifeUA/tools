from os.path import exists

from corpus import download, load_life_corpus, save_corpus, corpus2matrix
from eval import evaluate, metrics, print_metrics
from preproc import preprocess_corpus, preprocess
from gensim.corpora import Dictionary
import logging
from sklearn.svm import SVC

from utils import translate

log = logging.getLogger(__name__)

if not exists('data/life_corpus.csv'):
    corpus = download()
    save_corpus(corpus, 'data/life_corpus.csv')
if not exists('data/life_corpus_en.csv'):
    corpus = load_life_corpus('data/life_corpus.csv')
    corpus = translate(corpus, 'es', 'en')
    save_corpus(corpus, 'data/life_corpus_en.csv')
else:
    corpus = load_life_corpus('data/life_corpus_en.csv')

train_corpus = preprocess_corpus(corpus['Text'])
dictionary = Dictionary(train_corpus)
X_train = corpus2matrix(train_corpus, dictionary)
y_train = [0 if cls.lower() == 'no risk' else 1 for cls in corpus['Alert level']]

ml = SVC()
ml.fit(X_train, y_train)

y_pred = evaluate(X_train, ml)
measures = metrics(y_train, y_pred)
print_metrics(measures)
