from functools import lru_cache
from typing import List


class WordLevelQETokenizer:

    def __init__(self, lang: str):
        if lang in {"de", "en"}:
            from mosestokenizer import MosesTokenizer, MosesDetokenizer
            self.tokenizer = MosesTokenizer(lang)
            self.detokenizer = MosesDetokenizer(lang)
        elif lang in {"zh"}:
            self.tokenizer = lambda sentence: list(sentence)
            self.detokenizer = lambda tokens: "".join(tokens)
        else:
            raise NotImplementedError

    def __call__(self, sentence: str) -> List[str]:
        return self.tokenizer(sentence)

    def detokenize(self, tokens: List[str]) -> str:
        return self.detokenizer(tokens)


@lru_cache(None)
def load_tokenizer(language: str):
    return WordLevelQETokenizer(language)
