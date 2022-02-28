import argparse
from pathlib import Path
from typing import Union

import jsonlines
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from baseline.qe_model import SupervisedQualityEstimationModel
from data_generation.tokenizer import WordLevelQETokenizer
from evaluation.utils import MqmDataset, MqmSample, SampleId


def main(language_pair: str, split: str, checkpoint_path: Union[Path, str], nmt_predictions_path: Union[Path, str]):
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists()
    nmt_predictions_path = Path(nmt_predictions_path)
    assert nmt_predictions_path.exists()

    src_lang = language_pair.split("-")[0]

    if language_pair == "en-de":
        DEV_SYSTEM = "Online-B.1590"
    elif language_pair == "zh-en":
        DEV_SYSTEM = "Online-B.1605"
    else:
        raise ValueError

    MAX_SEQ_LEN = 120

    dataset = MqmDataset(language_pair)
    annotations = dataset.load_annotations()

    model = SupervisedQualityEstimationModel(checkpoint_path=checkpoint_path, gpu_id=0)

    annotated_additions = []
    predicted_additions = []
    annotated_omissions = []
    predicted_omissions = []

    nontranslations_count = 0
    five_count = 0
    quotes_count = 0
    overlong_count = 0
    multi_sentence_count = 0
    samples = []
    with jsonlines.open(nmt_predictions_path) as f:
        for line in f:
            sample = MqmSample(
                id=SampleId(
                    system=line["sample"]["id"]["system"],
                    seg_id=line["sample"]["id"]["seg_id"],
                ),
                mqm_source=line["sample"]["mqm_source"],
                mqm_target=line["sample"]["mqm_target"],
                original_source=line["sample"]["original_source"],
                original_target=line["sample"]["original_target"],
            )

            # Exclude multi-sentence samples
            if line["prediction"]["is_multi_sentence_input"]:
                multi_sentence_count += 1
                continue

            # Exclude very long samples
            if len(sample.original_source.split()) > 150 or len(sample.original_target.split()) > 150:
                overlong_count += 1
                continue
            samples.append(sample)

    if language_pair == "zh-en":
        src_tokenizer = WordLevelQETokenizer(src_lang)
        predictions = list(model.predict(
            src_sentences=[" ".join(src_tokenizer(sample.original_source)[:MAX_SEQ_LEN]) for sample in samples],
            tgt_sentences=[sample.original_target for sample in samples],
        ))
    else:
        predictions = list(model.predict(
            src_sentences=[sample.original_source for sample in samples],
            tgt_sentences=[sample.original_target for sample in samples],
        ))

    for i, sample in enumerate(tqdm(samples)):
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

        prediction = predictions[i]

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
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--nmt-predictions-path")
    args = parser.parse_args()
    main(args.language_pair, args.split, args.checkpoint_path, args.nmt_predictions_path)
