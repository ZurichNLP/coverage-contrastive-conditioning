import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Tuple

import jsonlines

from coverage.evaluator import CoverageResult
from data_generation.tokenizer import WordLevelQETokenizer, load_tokenizer


@dataclass
class QualityEstimationSample:
    src_tokens: List[str]
    tgt_tokens: List[str]
    src_tags: List[str]
    tgt_tags: List[str]


class SyntheticData:

    def __init__(self,
                 jsonl_path: Path,
                 src_lang: str,
                 tgt_lang: str,
                 ):
        self.jsonl_path = jsonl_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = WordLevelQETokenizer(src_lang)
        self.tgt_tokenizer = WordLevelQETokenizer(tgt_lang)

        self.qe_src_path = self.jsonl_path.with_suffix(".src")
        self.qe_mt_path = self.jsonl_path.with_suffix(".mt")
        self.qe_source_tags_path = self.jsonl_path.with_suffix(".source_tags")
        self.qe_tags_path = self.jsonl_path.with_suffix(".tags")

    def get_qe_samples(self) -> Iterable[QualityEstimationSample]:
        with jsonlines.open(self.jsonl_path) as f:
            for data in f:
                if data["type"] == "no_error":
                    src = data["full_source"]
                    tgt = data["full_machine_translation"]
                    src_tokens = self.src_tokenizer(src)
                    tgt_tokens = self.tgt_tokenizer(tgt)
                    src_tags = len(src_tokens) * ["OK"]
                    tgt_tags = len(tgt_tokens) * ["OK"]
                    yield QualityEstimationSample(
                        src_tokens=src_tokens,
                        tgt_tokens=tgt_tokens,
                        src_tags=src_tags,
                        tgt_tags=tgt_tags,
                    )
                else:
                    partial_src_tokens = self.src_tokenizer(data["partial_source"])
                    full_src_tokens = self.src_tokenizer(data["full_source"])
                    partial_tgt_tokens = self.tgt_tokenizer(data["partial_machine_translation"])
                    full_tgt_tokens = self.tgt_tokenizer(data["full_machine_translation"])
                    if not partial_src_tokens or not partial_tgt_tokens:
                        continue
                    # Overtranslation: partial_src -> full_tgt
                    i = 0; j = 0
                    addition_tgt_tags = []
                    while j < len(full_tgt_tokens):
                        if i < len(partial_tgt_tokens) and partial_tgt_tokens[i] == full_tgt_tokens[j]:
                            addition_tgt_tags.append("OK")
                            i += 1; j += 1
                        else:
                            addition_tgt_tags.append("BAD")
                            j += 1
                    yield QualityEstimationSample(
                        src_tokens=partial_src_tokens,
                        tgt_tokens=full_tgt_tokens,
                        src_tags=len(partial_src_tokens) * ["OK"],
                        tgt_tags=addition_tgt_tags,
                    )
                    # Undertranslation: full_src -> partial_tgt
                    i = 0; j = 0
                    omission_src_tags = []
                    while j < len(full_src_tokens):
                        if i < len(partial_src_tokens) and partial_src_tokens[i] == full_src_tokens[j]:
                            omission_src_tags.append("OK")
                            i += 1; j += 1
                        else:
                            omission_src_tags.append("BAD")
                            j += 1
                    yield QualityEstimationSample(
                        src_tokens=full_src_tokens,
                        tgt_tokens=partial_tgt_tokens,
                        src_tags=omission_src_tags,
                        tgt_tags=len(partial_tgt_tokens) * ["OK"],
                    )
                    # Correct translations
                    full_src_tags = len(full_src_tokens) * ["OK"]
                    full_tgt_tags = len(full_tgt_tokens) * ["OK"]
                    yield QualityEstimationSample(
                        src_tokens=full_src_tokens,
                        tgt_tokens=full_tgt_tokens,
                        src_tags=full_src_tags,
                        tgt_tags=full_tgt_tags,
                    )
                    partial_src_tags = len(partial_src_tokens) * ["OK"]
                    partial_tgt_tags = len(partial_tgt_tokens) * ["OK"]
                    yield QualityEstimationSample(
                        src_tokens=partial_src_tokens,
                        tgt_tokens=partial_tgt_tokens,
                        src_tags=partial_src_tags,
                        tgt_tags=partial_tgt_tags,
                    )

    def save_as_qe_data(self):
        with open(self.qe_src_path, "w") as f_src, \
                open(self.qe_mt_path, "w") as f_mt, \
                open(self.qe_source_tags_path, "w") as f_source_tags, \
                open(self.qe_tags_path, "w") as f_tags:
            for qe_sample in self.get_qe_samples():
                f_src.write(" ".join(qe_sample.src_tokens) + "\n")
                f_mt.write(" ".join(qe_sample.tgt_tokens) + "\n")
                f_source_tags.write(" ".join(qe_sample.src_tags) + "\n")
                f_tags.write(" ".join(qe_sample.tgt_tags) + "\n")


def tokenize_constituent(constituent: str, language: str) -> List[str]:
    if not constituent.strip():
        return []
    if language in {"en", "de"}:
        # Wrap in latin characters to ensure that punctuation is tokenized as in the full sentence
        wrapped_constituent = "a" + constituent.strip()
        if not constituent.endswith("."):
            wrapped_constituent += "a"
    else:
        wrapped_constituent = constituent
    tokenizer = load_tokenizer(language)
    tokens = tokenizer(wrapped_constituent)
    if language in {"en", "de"}:
        tokens[0] = tokens[0][1:]
        if not constituent.endswith("."):
            tokens[-1] = tokens[0][:-1]
        tokens = [token for token in tokens if token]
    return tokens


def convert_prediction_to_labels(
        src_len: int,
        tgt_len: int,
        prediction: CoverageResult,
        src_lang: str,
        tgt_lang: str,
) -> Tuple[List[str], List[str]]:
    source_labels = src_len * ["OK"]
    target_labels = tgt_len * ["OK"]

    for addition_error in (prediction.addition_errors or []):
        tokenized_prefix = tokenize_constituent(addition_error.constituent.original_sequence[:addition_error.constituent.start_char], tgt_lang)
        tokenized_constituent = tokenize_constituent(addition_error.constituent.removed, tgt_lang)
        tokenized_suffix = tokenize_constituent(addition_error.constituent.original_sequence[addition_error.constituent.end_char:], tgt_lang)
        try:
            # Tolerate off-by-one labels
            assert len(target_labels) - (len(tokenized_prefix) + len(tokenized_constituent) + len(tokenized_suffix)) in {-1, 0, 1}
            for i in range(len(target_labels)):
                if len(tokenized_prefix) <= i < len(tokenized_prefix) + len(tokenized_constituent):
                    target_labels[i] = "BAD"
        except AssertionError:
            logging.warning(f"Was not able to convert addition prediction to labels for {addition_error.constituent.removed}")
    for omission_error in (prediction.omission_errors or []):
        tokenized_prefix = tokenize_constituent(omission_error.constituent.original_sequence[:omission_error.constituent.start_char], src_lang)
        tokenized_constituent = tokenize_constituent(omission_error.constituent.removed, src_lang)
        tokenized_suffix = tokenize_constituent(omission_error.constituent.original_sequence[omission_error.constituent.end_char:], src_lang)
        try:
            assert len(source_labels) - (len(tokenized_prefix) + len(tokenized_constituent) + len(tokenized_suffix)) in {-1, 0, 1}
            for i in range(len(source_labels)):
                if len(tokenized_prefix) <= i < len(tokenized_prefix) + len(tokenized_constituent):
                    source_labels[i] = "BAD"
        except AssertionError:
            logging.warning(f"Was not able to convert omission prediction to labels for {omission_error.constituent.removed}")
    return source_labels, target_labels
