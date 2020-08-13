from os.path import exists

from scipy.sparse import csc_matrix

from corpus import corpus2bow, download, load_life_corpus, save_corpus
from preproc import load_corpus, preprocess_corpus, preprocess
from gensim.corpora import Dictionary
from tqdm import tqdm
import logging
from sklearn.svm import SVC
from numpy import matrix

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
classes = [0 if cls.lower() == 'no risk' else 1 for cls in corpus['Alert level']]
dictionary = Dictionary(train_corpus)
bow_corpus = corpus2bow(train_corpus, dictionary)
ml = SVC()


def corpus2matrix(corpus, dictionary):
    m = csc_matrix((len(corpus), len(dictionary)))
    for i, sample in enumerate(corpus):
        for token in sample:
            m[i, token[0]] = token[1]
    return m


m = corpus2matrix(bow_corpus, dictionary)

ml.fit(m, classes)
X = corpus2matrix(corpus2bow([preprocess("I hate feeling good for a while, and from one moment to the next having a horrible anguish, wanting to cry, just thinking about bullshit.")], dictionary), dictionary)
print('Result:', ml.predict(X[0]))
