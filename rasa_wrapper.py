from collections import OrderedDict
from sys import stdout
from typing import List, Dict, Any, Tuple
import os

from rasa.model import unpack_model
from rasa.nlu.model import Interpreter
from rasa.train import train
from ruamel.yaml import YAML
from yaml import dump

from utils import save_yaml

DOMAIN_FILE = 'domain.yml'
CONFIG_FILE = 'config.yml'
NLU_FILE = 'nlu.yml'
MODEL_FILE = 'models/test.tar.gz'


class RasaWrapper(object):
    def __init__(self, folder: str = '.'):
        self.folder = folder

    def train(self, corpus: List[str], y_train: List[str]) -> None:
        domain_file = os.path.join(self.folder, DOMAIN_FILE)
        config_file = os.path.join(self.folder, CONFIG_FILE)
        nlu_file = os.path.join(self.folder, NLU_FILE)
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        no_risk = self.extract_samples(corpus, y_train, "No risk")
        risk = self.extract_samples(corpus, y_train, "Risk")
        # domain = self.create_model_domain()
        # with open(domain_file, 'wt') as file:
        #     dump(domain, file)
        nlu = self.generate_nlu(no_risk, risk)
        self.save_nlu(nlu, nlu_file)
        # with open(nlu_file, 'wt') as file:
        #     dump(nlu, file)
        train(domain_file, config_file, nlu_file, MODEL_FILE, True)

    @staticmethod
    def create_model_domain() -> dict:
        return {
            'version': "2.0",
            'intents': ['no_risk', 'risk'],
            'stories': [
                {
                    'story': 'no_risk',
                    'steps': [
                        {'intent': 'no_risk'},
                        {'action': 'action_no_risk'}
                    ]
                },
                {
                    'story': 'risk',
                    'steps': [
                        {'intent': 'risk'},
                        {'action': 'action_risk'}
                    ]
                }
            ]
        }

    @staticmethod
    def extract_samples(corpus: List[str], y_train: List[str], cls: str) -> List[str]:
        return [corpus[i] for i in range(0, len(corpus)) if y_train[i] == cls]

    def generate_nlu(self, no_risk_examples: List[str], risk_examples: List[str]) -> dict:
        nlu =OrderedDict()
        nlu['version'] = '2.0'
        no_risk_nlu = OrderedDict()
        no_risk_nlu['intent'] = 'no_risk'
        no_risk_nlu['examples'] = no_risk_examples
        risk_nlu = OrderedDict()
        risk_nlu['intent'] = 'risk'
        risk_nlu['examples'] = risk_examples
        nlu['nlu'] = [no_risk_nlu, risk_nlu]
        return nlu

    def save_nlu(self, nlu: dict, nlu_file: str):
        yml = YAML()
        yml.indent(mapping=4, sequence=4, offset=4)
        with open(nlu_file, 'wt') as file:
            print(f'version: \'{nlu["version"]}\'', file=file)
            print(file=file)
            print(f'nlu:', file=file)
            for intent in nlu['nlu']:
                print(f'- intent: {intent["intent"]}', file=file)
                print(f'  examples: | ', file=file)
                yml.dump(intent['examples'], file)

    def evaluate(self, corpus: List[str]) -> List[Tuple[str, float]]:
        folder = unpack_model(MODEL_FILE)
        interpreter = Interpreter.load(os.path.join(folder, 'nlu'))
        # return [interpreter.parse(sample)[1] for sample in corpus]
        y_pred = []
        for sample in corpus:
            # y_pred.append(interpreter.parse(sample)[0]['intent'])
            intent = interpreter.parse(sample)['intent']
            # intents = sorted(intents, key=lambda x: -x['confidence'])
            # intent, confidence = intents[0]['intent'], intents[0]['confidence']
            y_pred.append((intent['name'], intent['confidence']))
        return y_pred
