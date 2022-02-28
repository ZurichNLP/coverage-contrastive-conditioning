import argparse
import tempfile
from pathlib import Path
from typing import Union

import jsonlines


def evaluate_word_level(config):
    from kiwi.lib.evaluate import evaluate_from_configuration

    print("Addition errors:")
    overtranslation_config = {
        "gold_files": {
            "target_tags": config["gold_files"]["target_tags"],
        },
        "predicted_files": {
            "target_tags": config["predicted_files"]["target_tags"],
        }
    }
    evaluate_from_configuration(overtranslation_config)
    print("Omission errors:")
    undertranslation_config = {
        "gold_files": {
            "source_tags": config["gold_files"]["source_tags"],
        },
        "predicted_files": {
            "source_tags": config["predicted_files"]["source_tags"],
        }
    }
    evaluate_from_configuration(undertranslation_config)


def evaluate_sentence_level(config):
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support

    print("Addition errors:")
    with open(config["gold_files"]["target_tags"]) as f:
        gold_overtranslations = ["BAD" in line for line in f]
    with open(config["predicted_files"]["target_tags"]) as f:
        predicted_overtranslations = ["BAD" in line for line in f]
    result = precision_recall_fscore_support(np.array(gold_overtranslations), np.array(predicted_overtranslations), average='binary')
    print("Precision\tRecall\tF1")
    print(f"{result[0]:.3f}\t{result[1]:.3f}\t{result[2]:.3f}")

    print("Omission errors:")
    with open(config["gold_files"]["source_tags"]) as f:
        gold_undertranslations = ["BAD" in line for line in f]
    with open(config["predicted_files"]["source_tags"]) as f:
        predicted_undertranslations = ["BAD" in line for line in f]
    result = precision_recall_fscore_support(np.array(gold_undertranslations), np.array(predicted_undertranslations), average='binary')
    print("Precision\tRecall\tF1")
    print(f"{result[0]:.3f}\t{result[1]:.3f}\t{result[2]:.3f}")


def main(
        nmt_predicted_source_tags_path: Union[Path, str],
        baseline_predicted_source_tags_labels_path: Union[Path, str],
        gold_source_tags_path: Union[Path, str],
        gold_target_tags_path: Union[Path, str],
):
    nmt_predicted_source_tags_path = Path(nmt_predicted_source_tags_path)
    assert nmt_predicted_source_tags_path.exists()
    baseline_predicted_source_tags_labels_path = Path(baseline_predicted_source_tags_labels_path)
    assert baseline_predicted_source_tags_labels_path.exists()
    gold_source_tags_path = Path(gold_source_tags_path)
    assert gold_source_tags_path.exists()
    gold_target_tags_path = Path(gold_target_tags_path)
    assert gold_target_tags_path.exists()

    baseline_config = {
        "gold_files": {
            "source_tags": str(gold_source_tags_path),
            "target_tags": str(gold_target_tags_path),
        },
        "predicted_files": {
            "source_tags": str(baseline_predicted_source_tags_labels_path),
            "target_tags": str(baseline_predicted_source_tags_labels_path.parent /
                               baseline_predicted_source_tags_labels_path.name.replace(".source_tags_labels",
                                                                                       ".target_tags_labels")),
        },
    }

    ours_config = {
        "gold_files": baseline_config["gold_files"],
        "predicted_files": {
            "source_tags": str(nmt_predicted_source_tags_path),
            "target_tags": str(nmt_predicted_source_tags_path).replace(".source_tags", ".tags"),
        },
    }

    single_sentences_indices = set()
    with jsonlines.open(Path(ours_config["predicted_files"]["source_tags"]).with_suffix(".jsonl")) as f:
        for i, line in enumerate(f):
            if not line["prediction"]["is_multi_sentence_input"]:
                single_sentences_indices.add(i)
    print(f"Number of samples used for evaluation: {len(single_sentences_indices)}")

    def extract_single_sentences(original_path: Path) -> Path:
        with open(original_path) as f_in, tempfile.NamedTemporaryFile("w", delete=False) as f_out:
            for i, line in enumerate(f_in):
                if i in single_sentences_indices:
                    f_out.write(line)
        return Path(f_out.name)

    baseline_config["gold_files"]["source_tags"] = str(
        extract_single_sentences(Path(baseline_config["gold_files"]["source_tags"])))
    baseline_config["gold_files"]["target_tags"] = str(
        extract_single_sentences(Path(baseline_config["gold_files"]["target_tags"])))
    baseline_config["predicted_files"]["source_tags"] = str(
        extract_single_sentences(Path(baseline_config["predicted_files"]["source_tags"])))
    baseline_config["predicted_files"]["target_tags"] = str(
        extract_single_sentences(Path(baseline_config["predicted_files"]["target_tags"])))
    ours_config["gold_files"] = baseline_config["gold_files"]
    ours_config["predicted_files"]["source_tags"] = str(
        extract_single_sentences(Path(ours_config["predicted_files"]["source_tags"])))
    ours_config["predicted_files"]["target_tags"] = str(
        extract_single_sentences(Path(ours_config["predicted_files"]["target_tags"])))

    print("Baseline word-level:")
    evaluate_word_level(baseline_config)
    print()
    print("Our approach word-level:")
    evaluate_word_level(ours_config)
    print()

    print("Baseline sentence-level:")
    evaluate_sentence_level(baseline_config)
    print()
    print("Our approach sentence-level:")
    evaluate_sentence_level(ours_config)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmt-predicted-source-tags-path")
    parser.add_argument("--baseline-predicted-source-tags-labels-path")
    parser.add_argument("--gold-source-tags-path")
    parser.add_argument("--gold-target-tags-path")
    args = parser.parse_args()
    main(args.nmt_predicted_source_tags_path,
         args.baseline_predicted_source_tags_labels_path, args.gold_source_tags_path, args.gold_target_tags_path)
