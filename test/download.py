import unittest
from tempfile import mkdtemp
from shutil import rmtree
from os import path
from corpus import LifeCorpus, RedditCorpus, REDDIT_CORPUS, GOLD_STANDARD_CORPUS
from wemb import WordEmbeddings


class DownloadTest(unittest.TestCase):

    def test_download_gold_standard(self) -> None:
        folder = mkdtemp()
        corpus = LifeCorpus(folder)
        self.assertTrue(path.exists(path.join(folder, GOLD_STANDARD_CORPUS)))
        self.assertTrue(path.exists(path.join(folder, corpus.fname)))
        rmtree(folder)

    def test_download_embeddings(self) -> None:
        folder = mkdtemp()
        WordEmbeddings('en', folder, 10000, 0.95)
        self.assertTrue(path.exists(path.join(folder, 'embeddings2')))
        rmtree(folder)

    def test_download_reddit(self) -> None:
        folder = mkdtemp()
        RedditCorpus(folder)
        self.assertTrue(path.exists(path.join(folder, REDDIT_CORPUS)))
        rmtree(folder)


if __name__ == '__main__':
    unittest.main()