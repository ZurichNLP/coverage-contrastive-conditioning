import argparse
import dataclasses
import json

from tqdm import tqdm

from coverage.evaluator import CoverageEvaluator
from evaluation.utils import MqmDataset, MqmSample
from translation_models import load_forward_and_backward_model


def main(model_name: str, language_pair: str):
    src_lang = language_pair.split("-")[0]
    tgt_lang = language_pair.split("-")[1]

    dataset = MqmDataset(language_pair)

    forward_model, backward_model = load_forward_and_backward_model(model_name, src_lang, tgt_lang)

    evaluator = CoverageEvaluator(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        forward_evaluator=forward_model,
        backward_evaluator=backward_model,
        batch_size=1,
    )

    for sample in tqdm(dataset.load_samples(load_original_sequences=True)):
        sample: MqmSample = sample
        result = evaluator.detect_errors(
            src=sample.original_source,
            translation=sample.original_target,
        )
        print(json.dumps({
            "sample": dataclasses.asdict(sample),
            "prediction": dataclasses.asdict(result),
        }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name")
    parser.add_argument("--language-pair")
    args = parser.parse_args()
    main(args.model_name, args.language_pair)
