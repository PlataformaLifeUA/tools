from typing import List, Union, Any
import re
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from unidecode import unidecode

from corptrans import NoTrainingTransformer, Sample, Transformers


def expand_contractions(text: str) -> str:
    """
    Expand contractions.

    :param text: the text with the contractions.
    :return: the text with the expanded contractions.
    """
    # specific
    text = re.sub(r"\'", "'", text)
    text = re.sub(r"/", " ", text)
    text = re.sub(r"’", "'", text)
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"a\'t", "can not", text)
    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text


def strip_html_tags(text: str) -> str:
    """
    Removing strip html tags.
    :param text: A text with HTML tags.
    :return: The same text without those HTML togs.
    """
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=' ')
    return stripped_text


def remove_accented_chars(text: str) -> str:
    """
    Convert accented characters into ASCII characters. It works mainly for latin languages.
    :param text: The text with accented characters.
    :return: The same text with with unicode representation of the accented characters.
    """
    return unidecode(text)


def remove_special_characters(text: str) -> str:
    """
    Remove spcial characters.
    :param text: The text with special characters.
    :return: the cleaned text.
    """
    nstr = re.sub(r'[?|$|.|!]', ' ', text)
    nestr = re.sub(r'[^a-zA-Z0-9 ]', ' ', nstr)
    return nestr


def remove_words_number(text):
    """
    Eliminar palabras de numeros
    """
    stopwords = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'ten']
    querywords = text.split()

    resultwords = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result


def remove_numbers(text):
    """
    Eliminar numeros
    """
    output = ''.join(c for c in text if not c.isdigit())
    return output


def stemmer(tokens: List[str]) -> List[str]:
    """
    Stemize verbs in list of tokenized words
    """
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in tokens:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


class StripHtmlTags(NoTrainingTransformer):
    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        if isinstance(sample, Sample):
            return sample.apply(strip_html_tags)
        return strip_html_tags(sample)


class RemoveAccentedChars(NoTrainingTransformer):
    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        if isinstance(sample, Sample):
            return sample.apply(unidecode)
        return unidecode(sample)


class ExpandContractions(NoTrainingTransformer):
    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        if isinstance(sample, Sample):
            return sample.apply(expand_contractions)
        return expand_contractions(sample)


class RemoveSpecialChracters(NoTrainingTransformer):
    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        if isinstance(sample, Sample):
            return sample.apply(remove_special_characters)
        return remove_special_characters(sample)


class RemoveWordNumbers(NoTrainingTransformer):
    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        if isinstance(sample, Sample):
            return sample.apply(remove_words_number)
        return remove_words_number(sample)


class RemoveNumbers(NoTrainingTransformer):
    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        if isinstance(sample, Sample):
            return sample.apply(remove_numbers)
        return remove_numbers(sample)


class Tokenize(NoTrainingTransformer):
    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        if isinstance(sample, Sample):
            return sample.apply(word_tokenize)
        return word_tokenize(sample)


class StopWords(NoTrainingTransformer):
    def __init__(self):
        super(StopWords, self).__init__()
        self.__stopwords = set(stopwords.words('english'))

    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        if isinstance(sample, Sample):
            return sample.gen([w for w in sample if not w in self.__stopwords])
        return [w for w in sample if not w in self.__stopwords]


class Lemmatize(NoTrainingTransformer):
    def __init__(self):
        super(Lemmatize, self).__init__()
        self.__lemmatizer = WordNetLemmatizer()

    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        if isinstance(sample, Sample):
            return  sample.gen([self.__lemmatizer.lemmatize(word, pos='v') for word in sample])
        return [self.__lemmatizer.lemmatize(word, pos='v') for word in sample]


class Preprocess(Transformers):
    def __init__(self):
        super(Preprocess, self).__init__()
        self.add([StripHtmlTags(), Lowercase(), RemoveAccentedChars(), ExpandContractions(), RemoveSpecialChracters(),
                  RemoveWordNumbers(), RemoveNumbers(), Tokenize(), StopWords(), Lemmatize()])


class Lowercase(NoTrainingTransformer):
    def transform(self, sample: Union[Sample, Any]) -> Union[Sample, Any]:
        if isinstance(sample, str):
            return sample.lower()
        if isinstance(sample, list):
            return [term.lower() for term in sample]
        if isinstance(sample.features, str):
            return sample.gen(sample.features.lower())
        return sample.gen([term.lower() for term in sample.features])
