from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, List

from stanza.models.common.doc import Sentence

from coverage.utils import is_ignored_upos

try:
    from dict_to_dataclass import DataclassFromDict, field_from_dict
except:  # Python 3.7 compatibility
    DataclassFromDict = object
    def field_from_dict(default=None):
        return default


class ParseTree:

    def __init__(self, sentence: Sentence):
        self.sentence = sentence
        for word in self.sentence.words:
            word.direct_children = set()
            word.indirect_children = set()
        for word in self.sentence.words:
            if word.head == 0:
                continue
            self.sentence.words[word.head - 1].direct_children.add(word)
        for word in self.sentence.words:
            todo = set(list(word.direct_children))  # deepcopy did not work for some reason
            while todo:
                node = todo.pop()
                word.indirect_children.add(node)
                todo.update(node.direct_children)
        self.indirect_children_counter = Counter({
            word: len(word.indirect_children) for word in self.sentence.words
        })

    def iter_constituents(self) -> Iterable['Constituent']:
        for subtree, _ in self.indirect_children_counter.most_common():
            # Check whether subtree is contiguous
            word_ids = sorted({child.id for child in subtree.indirect_children} | {subtree.id})
            if max(word_ids) - min(word_ids) + 1 != len(word_ids):
                continue
            if all(is_ignored_upos(word) for word in subtree.indirect_children | {subtree}):
                continue
            start_char = self.sentence.words[min(word_ids) - 1].parent.start_char
            end_char = self.sentence.words[max(word_ids) - 1].parent.end_char
            if start_char is None or end_char is None:
                continue
            constituent = Constituent(
                original_sequence=self.sentence.text,
                start_char=start_char,
                end_char=end_char,
                removed_upos=[word.upos for word in sorted(subtree.indirect_children | {subtree}, key=lambda word: word.id)],
            )
            if not constituent.remainder or not constituent.removed:
                continue
            yield constituent


@dataclass
class Constituent(DataclassFromDict):
    original_sequence: str = field_from_dict()
    start_char: Optional[int] = field_from_dict(default=None)
    end_char: Optional[int] = field_from_dict(default=None)
    remainder: Optional[str] = field_from_dict(default=None)
    removed: Optional[str] = field_from_dict(default=None)
    removed_upos: Optional[List[str]] = field_from_dict(default=None)
    constituent_score: Optional[float] = field_from_dict(default=None)
    base_score: Optional[float] = field_from_dict(default=None)

    def __post_init__(self):
        if self.remainder is None:
            self.remainder = self._get_remainder()
        if self.removed is None:
            self.removed = self._get_removed()

    def _get_remainder(self) -> str:
        remainder = self.original_sequence[:self.start_char] + self.original_sequence[self.end_char:]
        remainder = remainder.strip()
        return remainder

    def _get_removed(self) -> str:
        removed = self.original_sequence[self.start_char:self.end_char]
        removed = removed.strip()
        return removed
