from transformers import PreTrainedModel, PreTrainedTokenizerFast

from shared.decoder_utils import load_model_and_tokenizer


class Cache:
    """
    A simple in-memory cache that stores key-value pairs.
    Supports setting, getting, and evicting the least recently used item.
    Maximum number of items in the cache can be set during initialization.
    If the cache is full, the least recently used item will be evicted.
    """

    __instance = None
    _data = {}
    _max_items = 100

    def __new__(cls, max_items=None):
        if not Cache.__instance:
            Cache.__instance = super(Cache, cls).__new__(cls)
            if max_items:
                cls._max_items = max_items
        return Cache.__instance

    def set(self, key, value):
        if len(self._data) >= self._max_items:
            self.evict_lru()
        self._data[key] = value

    def get(self, key):
        value = self._data.get(key)
        if value:
            self._data[key] = value
        return value

    def evict_lru(self):
        if self._data:
            del self._data[list(self._data.keys())[0]]

    def clear(self):
        self._data = {}


class ModelSingleton:
    """
    A singleton class that loads the model and tokenizer from the transformers library.
    """

    __instance = None
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast

    def __new__(cls, max_items=None):
        if not ModelSingleton.__instance:
            ModelSingleton.__instance = super(ModelSingleton, cls).__new__(cls)
            tokenizer, model = load_model_and_tokenizer()
            cls.model = model
            cls.tokenizer = tokenizer

        return ModelSingleton.__instance
