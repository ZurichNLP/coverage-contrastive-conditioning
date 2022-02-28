import logging
from dataclasses import dataclass
from typing import List, Optional

from stanza.models.common.doc import Document

from coverage.parser import Constituent, ParseTree
from coverage.utils import load_stanza_parser
from translation_models import TranslationModel

try:
    from dict_to_dataclass import DataclassFromDict, field_from_dict
except:  # Python 3.7 compatibility
    DataclassFromDict = object
    def field_from_dict(default=None):
        return default


@dataclass
class CoverageError(DataclassFromDict):
    constituent: Constituent = field_from_dict()

    def __str__(self):
        return self.constituent.removed


@dataclass
class AdditionError(CoverageError):
    pass


@dataclass
class OmissionError(CoverageError):
    pass


@dataclass
class CoverageResult(DataclassFromDict):
    addition_errors: Optional[List[AdditionError]] = field_from_dict()
    omission_errors: Optional[List[OmissionError]] = field_from_dict()
    src_lang: Optional[str] = field_from_dict()
    tgt_lang: Optional[str] = field_from_dict()
    is_multi_sentence_input: Optional[bool] = field_from_dict(default=None)

    @property
    def contains_addition_error(self) -> bool:
        return len(self.addition_errors) >= 1

    @property
    def contains_omission_error(self) -> bool:
        return len(self.omission_errors) >= 1

    def __str__(self):
        return "".join([
            f"Addition errors: {' | '.join(map(str, self.addition_errors))}" if self.addition_errors else "",
            "\n" if self.addition_errors and self.omission_errors else "",
            f"Omission errors: {' | '.join(map(str, self.omission_errors))}" if self.omission_errors else "",
        ])


class CoverageEvaluator:

    def __init__(self,
                 src_lang: str = None,
                 tgt_lang: str = None,
                 forward_evaluator: TranslationModel = None,
                 backward_evaluator: TranslationModel = None,
                 batch_size: int = 16,
                 ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_parser = load_stanza_parser(src_lang) if src_lang is not None else None
        self.tgt_parser = load_stanza_parser(tgt_lang) if tgt_lang is not None else None
        self.forward_evaluator = forward_evaluator
        self.backward_evaluator = backward_evaluator
        self.batch_size = batch_size

    def _get_error_constituents(self,
                                src_lang: str,
                                tgt_lang: str,
                                src_sequence: str,
                                tgt_sequence: str,
                                evaluator,
                                parser=None,
                                src_doc: Document = None,
                                ) -> Optional[List[Constituent]]:
        if src_doc is None:
            src_doc = parser(src_sequence)
        if len(src_doc.sentences) > 1:
            logging.warning("Coverage detection currently does not handle multi-sentence inputs; skipping ...")
            return None
        tree = ParseTree(src_doc.sentences[0])
        constituents = list(tree.iter_constituents())
        if not constituents:
            return []

        scores = evaluator.score(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            source_sentences=[src_sequence] + [constituent.remainder for constituent in constituents],
            hypothesis_sentences=(1 + len(constituents)) * [tgt_sequence],
            batch_size=self.batch_size,
        )
        base_score = scores[0]
        returned_constituents = []
        for score, constituent in zip(scores[1:], constituents):
            if score > base_score:
                constituent.constituent_score = score
                constituent.base_score = base_score
                returned_constituents.append(constituent)
        return returned_constituents

    def detect_errors(self, src: str, translation: str, src_doc: Document = None, translation_doc: Document = None) -> CoverageResult:
        is_multi_sentence_input = False
        addition_errors = None
        if self.backward_evaluator is not None:
            tgt_constituents = self._get_error_constituents(
                src_lang=self.tgt_lang,
                tgt_lang=self.src_lang,
                src_sequence=translation,
                tgt_sequence=src,
                evaluator=self.backward_evaluator,
                parser=self.tgt_parser,
                src_doc=translation_doc,
            )
            if tgt_constituents is None:
                is_multi_sentence_input = True
            else:
                addition_errors = [AdditionError(constituent=constituent) for constituent in tgt_constituents]

        omission_errors = None
        if self.forward_evaluator is not None:
            src_constituents = self._get_error_constituents(
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                src_sequence=src,
                tgt_sequence=translation,
                evaluator=self.forward_evaluator,
                parser=self.src_parser,
                src_doc=src_doc,
            )
            if src_constituents is None:
                is_multi_sentence_input = True
            else:
                omission_errors = [OmissionError(constituent=constituent) for constituent in src_constituents]

        return CoverageResult(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            addition_errors=addition_errors,
            omission_errors=omission_errors,
            is_multi_sentence_input=is_multi_sentence_input,
        )
