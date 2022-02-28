import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Set, Tuple

try:
    from dict_to_dataclass import DataclassFromDict, field_from_dict
except:  # Python 3.7 compatibility
    DataclassFromDict = object
    def field_from_dict(default=None):
        return default


@dataclass(frozen=True)
class SampleId(DataclassFromDict):
    system: str = field_from_dict()
    seg_id: int = field_from_dict()


@dataclass
class MqmSample(DataclassFromDict):
    id: SampleId = field_from_dict()
    mqm_source: str = field_from_dict()
    mqm_target: str = field_from_dict()
    original_source: Optional[str] = field_from_dict(default=None)
    original_target: Optional[str] = field_from_dict(default=None)

    def __str__(self):
        return f"{self.original_source} -> {self.original_target}"

    @property
    def has_superfluous_quotes(self) -> bool:
        assert self.original_target is not None
        return any([
            '""' in self.clean_mqm_target and '""' not in self.original_target,
            "''" in self.clean_mqm_target and "''" not in self.original_target,
        ])

    @property
    def clean_mqm_target(self) -> str:
        return self.mqm_target.replace("<v>", "").replace("</v>", "")


@dataclass
class MqmRating(DataclassFromDict):
    rater: str = field_from_dict()
    category: str = field_from_dict()
    severity: str = field_from_dict()
    mqm_source: str = field_from_dict(default=None)
    mqm_target: str = field_from_dict(default=None)


@dataclass
class MqmAnnotatedSample(MqmSample):
    ratings: List[MqmRating] = field_from_dict(default=None)

    @property
    def categories_per_rater(self) -> Dict[str, List[str]]:
        categories = defaultdict(list)
        for rating in self.ratings:
            categories[rating.rater].append(rating.category)
        return categories

    @property
    def has_any_nontranslation_rating(self) -> bool:
        return any(rating.category == "Non-translation!" for rating in self.ratings)

    @property
    def might_have_unmarked_coverage_error(self) -> bool:
        """
        :return: True if a rater has stopped at 5 errors without marking a coverage error
        """
        for rater, categories in self.categories_per_rater.items():
            if len(categories) <= 4:
                continue
            if "Accuracy/Addition" not in categories or "Accuracy/Omission" not in categories:
                return True
        return False

    @property
    def has_addition_error_by_majority(self) -> bool:
        return self._has_category_by_majority("Accuracy/Addition")

    @property
    def has_omission_error_by_majority(self) -> bool:
        return self._has_category_by_majority("Accuracy/Omission")

    @property
    def has_addition_error_by_any_rater(self) -> bool:
        return self._has_category_by_any_rater("Accuracy/Addition")

    @property
    def has_omission_error_by_any_rater(self) -> bool:
        return self._has_category_by_any_rater("Accuracy/Omission")

    def _has_category_by_majority(self, category: str) -> bool:
        votes = 0
        for rater, categories in self.categories_per_rater.items():
            votes += category in categories
        return votes > len(self.categories_per_rater) / 2

    def _has_category_by_any_rater(self, category: str) -> bool:
        for rater, categories in self.categories_per_rater.items():
            if category in categories:
                return True
        return False


class MqmDataset:

    def __init__(self, language_pair: str, tsv_path: Path = None):
        self.language_pair = language_pair
        if tsv_path is None:
            self.tsv_path = Path(__file__).parent.parent / "data" / "mqm" / f"mqm_newstest2020_{language_pair.replace('-', '')}.tsv"
        else:
            self.tsv_path = Path(tsv_path)
        assert self.tsv_path.exists()

    def __str__(self):
        return self.tsv_path.name

    def load_original_sequences(self) -> Dict[SampleId, Tuple[str, str]]:
        original_sequences: Dict[SampleId, Tuple[str, str]] = dict()
        src_path = Path(__file__).parent.parent / "data" / "mqm" / "original_src" / \
                   f"newstest2020-{self.language_pair.replace('-', '')}-src.{self.language_pair.split('-')[0]}.txt"
        with open(src_path) as f:
            src_lines = f.read().splitlines()
        tgt_dir = Path(__file__).parent.parent / "data" / "mqm" / "original_tgt"
        for system_path in tgt_dir.glob(f"newstest2020.{self.language_pair}.*.txt"):
            system = system_path.name.replace(f"newstest2020.{self.language_pair}.", "").replace(".txt", "")
            with open(system_path) as f:
                tgt_lines = f.read().splitlines()
            assert len(src_lines) == len(tgt_lines)
            for i in range(len(src_lines)):
                sample_id = SampleId(
                    seg_id=(i + 1),
                    system=system,
                )
                assert sample_id not in original_sequences
                original_sequences[sample_id] = (src_lines[i], tgt_lines[i])
        return original_sequences

    def load_samples(self, load_original_sequences=False) -> Iterable[MqmSample]:
        if load_original_sequences:
            original_sequences = self.load_original_sequences()

        seen_sample_ids: Set[SampleId] = set()
        with open(self.tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for line in reader:
                sample_id = SampleId(
                    system=line["system"],
                    seg_id=int(line["seg_id"]),
                )
                if sample_id in seen_sample_ids:
                    continue
                sample = MqmSample(
                    id=sample_id,
                    mqm_source=line["source"],
                    mqm_target=line["target"],
                )
                if load_original_sequences:
                    sample.original_source, sample.original_target = original_sequences[sample_id]
                seen_sample_ids.add(sample_id)
                yield sample

    def load_annotations(self) -> Dict[SampleId, MqmAnnotatedSample]:
        annotations: Dict[SampleId, MqmAnnotatedSample] = dict()
        with open(self.tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for line in reader:
                rating = MqmRating(
                    rater=line["rater"],
                    category=line["category"],
                    severity=line["severity"],
                    mqm_source=line["source"],
                    mqm_target=line["target"],
                )
                sample_id = SampleId(
                    system=line["system"],
                    seg_id=int(line["seg_id"]),
                )
                annotated_sample = MqmAnnotatedSample(
                    id=sample_id,
                    mqm_source=line["source"],
                    mqm_target=line["target"],
                    ratings=[rating],
                )
                if sample_id in annotations:
                    annotations[sample_id].ratings.append(rating)
                else:
                    annotations[sample_id] = annotated_sample
        return annotations
