import os
from typing import List, Type, Optional, Dict, Text, Any

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
import logging

from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

from rules import ReplaceRules, DEF_REPL_RULES

log = logging.getLogger(__name__)
ORIGINAL_MSG_ATT = 'original_msg_text'
RULES_ATT = 'replace_rules'
RULES_FILE_ATT = 'rules'
OUTPUT_PROPS_ATT = 'output_properties'


class Replace(Component):
    """
    Replace texts using regular expressions for different languages.
    """
    name = "Replace"
    provides = []
    requires = []
    defaults = {'force': False, RULES_FILE_ATT: os.path.join('data', 'replace.json')}
    # Defines what language(s) this component can NOT handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    not_supported_language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None, lang: str = 'en') -> None:
        """
        Constructor.
        :param component_config: The configuration of this Rasa Component.
        :param lang: The language to deal with.
        """
        super().__init__(component_config)
        # Create the replace rules
        fname_or_dict = component_config[RULES_FILE_ATT] if RULES_FILE_ATT in component_config else DEF_REPL_RULES
        if 'force' not in component_config or not component_config['force']:
            fname_or_dict = component_config[RULES_ATT] if RULES_ATT in component_config else fname_or_dict
        if isinstance(fname_or_dict, str):
            log.info(f'Loading the replace rules from {fname_or_dict}')
        else:
            log.info('Using the previously loaded replace rules.')
        self.replace_rules = ReplaceRules(fname_or_dict, lang)
        # Defines what language(s) this component can handle.
        self.supported_language_list = self.replace_rules.language_list

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline.
        :return: A empty list.
        """
        return []

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one.
        :param training_data: The training data.
        :param config: The NLU configuration.
        :param kwargs: Extra parameters if they are necessary.
        """
        pass

    def process(self, message: Message, **kwargs: Any) -> None:
        """Replace following some regex rules the message text.

        :param message: The user message to answer.
        :param kwargs: Extra arguments if they are necessary.
        """
        text = message.get('text')
        if text:
            message.set(ORIGINAL_MSG_ATT, text, add_to_output=True)
            message.set('text', self.replace_rules.replace(text))

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """
        This component doesn't need any persist process.
        :param file_name: The file name.
        :param model_dir: The folder where the model is stored.
        :return: Dictionary with parameters for this component. They will be used to load the component.
        """
        return {RULES_ATT: self.replace_rules.__dict__}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """
        Load this component from file.

        :param meta: The metadata returned when this object was persisted.
        :param model_dir: The folder where the model is stored.
        :param model_metadata: NLU configuration.
        :param cached_component: If this object was already loaded.
        :param kwargs: Extra argument if they are necessary.
        :return: An instance of this Rasa Component.
        """
        if cached_component:
            return cached_component
        else:
            return cls(meta, model_metadata.language)


class PublishSpacyTokens(Component):
    """
    Publish the generated Spacy Tokens to be used in actions.
    """
    name = "PublishSpacyTokens"
    provides = []
    requires = []
    defaults = {}
    # Defines what language(s) this component can NOT handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    not_supported_language_list = None
    supported_language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        """
        Constructor.
        :param component_config: The configuration of this Rasa Component.
        """
        super().__init__(component_config)

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline.
        :return: A empty list.
        """
        return []

    def process(self, message: Message, **kwargs: Any) -> None:
        """Select the Spacy tokens for the message and convert to a standard dictionary.

        :param message: The user message to answer.
        :param kwargs: Extra arguments if they are necessary.
        """
        if 'text_tokens' in message.data:
            tokens = message.data['text_tokens']
            message.set('tokens', [{
                'start': token.start,
                'end': token.end,
                'lemma': token.lemma,
                'pos': token.data['pos'] if 'pos' in token.data else None,
                'text': token.text} for token in tokens], add_to_output=True)
