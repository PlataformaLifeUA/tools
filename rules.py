import json
import re
from typing import List, Dict, Union

DEF_REPL_RULES = 'data/replace.json'
GENERIC_LANG = 'generic'


class ReplaceRules(object):
    """ A class to replace text based on regular expressions. """
    @property
    def language_list(self) -> List[str]:
        """
        :return: The list of available languages in the rule file.
        """
        return self._language_list

    @property
    def __dict__(self):
        """
        :return: The original representation of this replace rules.
        """
        return self._rules

    def __init__(self, fname_or_dict: Union[str, dict], lang: str = 'en'):
        """
        Constructor. Load the rule file.
        :param fname_or_dict: The file path to the rule files or the dictionary with the replace rules.
        :param lang: The language to use.
        """
        # Load the replace rules for all languages
        if isinstance(fname_or_dict, str):
            with open(fname_or_dict, 'rt') as file:
                self._rules = json.load(file)
        else:
            self._rules = fname_or_dict
        # Defines what language(s) this component can handle.
        self._language_list = [lang for lang in self._rules if lang != GENERIC_LANG]
        # If the language is not defined, then this process will use the generic rules
        self._replace_rules = self._create_rules(self._rules, lang)

    @staticmethod
    def _create_rules(rules: Dict[str, Dict[str, Union[str, List[str]]]], language: str) -> Dict[str, str]:
        """
        Create a dictionary with the rules to replace for a specific language.
        :param rules: The loaded rules for all languages.
        :param language: The language to create the rules.
        :return: A dictionary where the key is the pattern to replace and the value is the replace value.
        """
        result = {}
        for lang in rules:
            if lang == GENERIC_LANG or lang == language:
                for term, substitutes in rules[lang].items():
                    # If the substitutes are only a string element
                    if isinstance(substitutes, str):
                        result[substitutes] = term
                    else:
                        # If the substitutes are a list of strings
                        for substitute in substitutes:
                            result[substitute] = term

        return result

    def replace(self, text: str) -> str:
        """
        Replace parts of a text using the replace rules.
        :param text: The original text.
        :return: The replaced text.
        """
        for pattern in self._replace_rules:
            for match in reversed(list(re.finditer(pattern, text, flags=re.IGNORECASE))):
                pos = match.regs[0]  # Obtain the position of the match (initial and end)
                text = text[:pos[0]] + self._replace_rules[pattern] + text[pos[1]:]
        return text
