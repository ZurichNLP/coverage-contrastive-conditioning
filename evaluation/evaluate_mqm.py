import argparse
from pathlib import Path
from typing import Union

import jsonlines
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from coverage.evaluator import CoverageResult
from evaluation.utils import MqmDataset, MqmSample


def main(language_pair: str, split: str, predictions_path: Union[Path, str]):
    predictions_path = Path(predictions_path)
    assert predictions_path.exists()

    if language_pair == "en-de":
        DEV_SYSTEM = "Online-B.1590"
    elif language_pair == "zh-en":
        DEV_SYSTEM = "Online-B.1605"
    else:
        raise ValueError

    dataset = MqmDataset(language_pair)
    annotations = dataset.load_annotations()

    annotated_additions = []
    predicted_additions = []
    annotated_omissions = []
    predicted_omissions = []

    nontranslations_count = 0
    five_count = 0
    quotes_count = 0
    overlong_count = 0
    multi_sentence_count = 0
    both_count = 0
    with jsonlines.open(predictions_path) as f:
        for line in tqdm(f):
            sample = MqmSample.from_dict(line["sample"])
            prediction = CoverageResult.from_dict(line["prediction"])
            annotated_sample = annotations[sample.id]

            # Evaluate either on dev or test set
            if split == "dev":
                if sample.id.system != DEV_SYSTEM:
                    continue
            elif split == "test":
                if sample.id.system == DEV_SYSTEM:
                    continue
            else:
                raise ValueError

            # Exclude human "systems"
            if "human" in sample.id.system.lower():
                continue

            # Exclude samples with incomplete annotations
            if annotated_sample.has_any_nontranslation_rating:
                nontranslations_count += 1
                continue
            if annotated_sample.might_have_unmarked_coverage_error:
                five_count += 1
                continue

            # Exclude samples with a presumed data processing error regarding quotes
            if sample.has_superfluous_quotes:
                quotes_count += 1
                continue

            # Exclude multi-sentence samples
            if prediction.is_multi_sentence_input:
                multi_sentence_count += 1
                continue

            # Exclude very long samples
            if len(sample.original_source.split()) > 150 or len(sample.original_target.split()) > 150:
                overlong_count += 1
                continue

            if annotated_sample.has_addition_error_by_any_rater and annotated_sample.has_omission_error_by_any_rater:
                both_count += 1
                continue

            annotated_additions.append(annotated_sample.has_addition_error_by_any_rater)
            predicted_additions.append(prediction.contains_addition_error)
            annotated_omissions.append(annotated_sample.has_omission_error_by_any_rater)
            predicted_omissions.append(prediction.contains_omission_error)

    print("Addition errors")
    assert len(annotated_additions) == len(predicted_additions)
    result = precision_recall_fscore_support(np.array(annotated_additions), np.array(predicted_additions), average='binary')
    print("Precision\tRecall\tF1")
    print(f"{result[0]:.3f}\t{result[1]:.3f}\t{result[2]:.3f}")

    print("Omission errors")
    assert len(annotated_omissions) == len(predicted_omissions)
    result = precision_recall_fscore_support(np.array(annotated_omissions), np.array(predicted_omissions), average='binary')
    print("Precision\tRecall\tF1")
    print(f"{result[0]:.3f}\t{result[1]:.3f}\t{result[2]:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-pair")
    parser.add_argument("--split")
    parser.add_argument("--predictions-path")
    args = parser.parse_args()
    main(args.language_pair, args.split, args.predictions_path)
