import argparse
import dataclasses
from pathlib import Path

import jsonlines
from tqdm import tqdm

from baseline.utils import convert_prediction_to_labels
from coverage.evaluator import CoverageEvaluator, CoverageResult
from data_generation.tokenizer import WordLevelQETokenizer
from translation_models import load_forward_and_backward_model


def main(model_name: str, language_pair: str, dataset_name: str):
    src_lang = language_pair.split("-")[0]
    tgt_lang = language_pair.split("-")[1]

    dataset_name = Path(dataset_name)
    src_path = dataset_name.parent / (dataset_name.name + ".src")
    if language_pair == "zh-en":
        src_path = dataset_name.parent / (dataset_name.name + ".src.cleaned.truncated")
    assert src_path.exists()
    tgt_path = dataset_name.parent / (dataset_name.name + ".mt")
    assert tgt_path.exists()

    forward_model, backward_model = load_forward_and_backward_model(model_name, src_lang, tgt_lang)

    predictions_dir = Path(__file__).parent.parent / "predictions"
    out_jsonl_path = predictions_dir / (dataset_name.name + f".{model_name}.jsonl")
    out_source_tags_path = predictions_dir / (dataset_name.name + f".{model_name}.source_tags")
    out_target_tags_path = predictions_dir / (dataset_name.name + f".{model_name}.tags")

    evaluator = CoverageEvaluator(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        forward_evaluator=forward_model,
        backward_evaluator=backward_model,
        batch_size=(1 if language_pair == "zh-en" else 2),
    )

    src_tokenizer = WordLevelQETokenizer(src_lang)
    tgt_tokenizer = WordLevelQETokenizer(tgt_lang)

    with open(src_path) as f_src, open(tgt_path) as f_tgt, \
            open(out_source_tags_path, "w") as f_out_src, open(out_target_tags_path, "w") as f_out_tgt, \
            jsonlines.open(out_jsonl_path, "w") as f_out_jsonl:
        for src, tgt in zip(tqdm(f_src), f_tgt):
            src_tokens = src.strip().split()
            tgt_tokens = tgt.strip().split()
            src_detokenized = src_tokenizer.detokenize(src_tokens)
            tgt_detokenized = tgt_tokenizer.detokenize(tgt_tokens)
            result = evaluator.detect_errors(
                src=src_detokenized,
                translation=tgt_detokenized,
            )
            source_tags, target_tags = convert_prediction_to_labels(
                src_len=len(src_tokens),
                tgt_len=len(tgt_tokens),
                prediction=result,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
            f_out_src.write(" ".join(source_tags) + "\n")
            f_out_tgt.write(" ".join(target_tags) + "\n")
            f_out_jsonl.write({
                "src": src.strip(),
                "tgt": tgt.strip(),
                "src_detokenized": src_detokenized,
                "tgt_detokenized": tgt_detokenized,
                "prediction": dataclasses.asdict(result),
                "source_tags": source_tags,
                "target_tags": target_tags,
            })

    src_out_path = out_jsonl_path.with_suffix(".source_tags")
    tgt_out_path = out_jsonl_path.with_suffix(".tags")

    with jsonlines.open(out_jsonl_path) as f_in, open(src_out_path, "w") as f_src, open(tgt_out_path, "w") as f_tgt:
        for i, line in enumerate(f_in):
            prediction = CoverageResult.from_dict(line["prediction"])
            src_labels, tgt_labels = convert_prediction_to_labels(
                src_len=len(line["src"].split()),
                tgt_len=len(line["tgt"].split()),
                prediction=prediction,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
            f_src.write(" ".join(src_labels) + "\n")
            f_tgt.write(" ".join(tgt_labels) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name")
    parser.add_argument("--language-pair")
    parser.add_argument("--dataset-name")
    args = parser.parse_args()
    main(args.model_name, args.language_pair, args.dataset_name)
