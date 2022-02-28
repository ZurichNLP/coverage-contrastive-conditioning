import difflib
import random

from stanza import Pipeline

from coverage.parser import ParseTree
from translation_models import TranslationModel


class SyntheticExample:

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 full_source: str,
                 parser: Pipeline,
                 forward_model: TranslationModel,
                 deletion_probability: float = 0.15,
                 ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.full_source = full_source
        self.parser = parser
        self.forward_model = forward_model
        self.deletion_probability = deletion_probability

        self.partial_source = None
        self.removed_source_spans = None
        self.full_machine_translation = None
        self.partial_machine_translation = None
        self.full_machine_source = None
        self.partial_machine_source = None

    def parse_source(self) -> None:
        self.source_doc = self.parser(self.full_source)
        self.source_tree = None
        if self.is_single_sentence():
            self.source_tree = ParseTree(self.source_doc.sentences[0])
            self.source_constituents = list(self.source_tree.iter_constituents())
        else:
            self.source_tree = None
            self.source_constituents = None

    def is_single_sentence(self) -> bool:
        return len(self.source_doc.sentences) == 1

    def create_partial_source(self) -> None:
        removed_spans = []
        for constituent in self.source_constituents:
            if random.random() < self.deletion_probability:
                removed_spans.append((constituent.start_char, constituent.end_char))
        if not removed_spans:
            partial_source = self.full_source
        else:
            partial_source = self.full_source[0:removed_spans[0][0]]
            for i, removed_span in enumerate(removed_spans):
                if i == 0:
                    continue
                partial_source += self.full_source[removed_spans[i-1][1]:removed_span[0]]
            partial_source += self.full_source[removed_spans[-1][1]:]
        self.partial_source = partial_source
        self.removed_source_spans = removed_spans

    def partial_source_is_strict(self) -> bool:
        return self.partial_source != self.full_source

    def translate_full_source(self) -> None:
        self.full_machine_translation = self.forward_model.translate(self.src_lang, self.tgt_lang, [self.full_source])[0]

    def translate_partial_source(self) -> None:
        if self.partial_source == self.full_source and self.full_machine_translation is not None:
            self.partial_machine_translation = self.full_machine_translation
        else:
            self.partial_machine_translation = self.forward_model.translate(self.src_lang, self.tgt_lang, [self.partial_source])[0]

    def partial_translation_is_strict(self) -> bool:
        tokens_original = self.full_machine_translation.split()
        tokens_post_edited = self.partial_machine_translation.split()
        matcher = difflib.SequenceMatcher(a=tokens_original, b=tokens_post_edited)
        opcodes = matcher.get_opcodes()
        operations = [opcode[0] for opcode in opcodes]
        return operations.count("delete") and not operations.count("replace") and not operations.count("insert")

    def to_dict(self) -> dict:
        data = {
            "full_source": self.full_source,
            "partial_source": self.partial_source,
            "full_machine_translation": self.full_machine_translation,
            "partial_machine_translation": self.partial_machine_translation,
            "full_machine_source": self.full_machine_source,
            "partial_machine_source": self.partial_machine_source,
            "removed_source_spans": self.removed_source_spans,
        }
        if not self.partial_source_is_strict():
            data["type"] = "no_error"
        else:
            data["type"] = "error"
        return data
