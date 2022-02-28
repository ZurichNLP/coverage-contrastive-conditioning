import argparse
from pathlib import Path
from typing import Union

import jsonlines
from tqdm import tqdm

from baseline.utils import SyntheticData
from coverage.utils import load_stanza_parser
from data_generation.utils import SyntheticExample
from translation_models import load_translation_model


def main(language_pair: str, src_path: Union[Path, str], out_path: Union[Path, str] = None, translation_model_name: str = "mbart"):
    src_lang = language_pair.split("-")[0]
    tgt_lang = language_pair.split("-")[1]

    src_path = Path(src_path)
    assert src_path.exists()
    out_path = out_path or src_path.with_suffix(".out.jsonl")

    parser = load_stanza_parser(src_lang)

    forward_model = load_translation_model(translation_model_name)

    with open(src_path) as f_src, jsonlines.open(out_path, "w") as f:
        for src_line in tqdm(f_src):
            sample = SyntheticExample(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                full_source=src_line.strip(),
                parser=parser,
                forward_model=forward_model,
            )
            if len(sample.full_source.split()) > 250:
                continue
            sample.parse_source()
            if not sample.is_single_sentence():
                continue
            sample.create_partial_source()
            sample.translate_full_source()
            sample.translate_partial_source()
            # Only keep negative samples or strict additions
            if sample.partial_source_is_strict() and not sample.partial_translation_is_strict():
                continue
            f.write(sample.to_dict())

    synthetic_data = SyntheticData(
        jsonl_path=out_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    synthetic_data.save_as_qe_data()

    # Postprocess Chinese data
    if src_lang == "zh":
        MAX_SEQ_LEN = 120
        input_path = synthetic_data.qe_src_path
        tags_path = synthetic_data.qe_tags_path
        output_path = synthetic_data.qe_src_path.parent / (synthetic_data.qe_src_path.name + ".cleaned.truncated")
        tags_output_path = synthetic_data.qe_tags_path.parent / (synthetic_data.qe_tags_path.name + ".cleaned.truncated")
        with open(input_path) as f_input, open(tags_path) as f_tags, \
                open(output_path, "w") as f_output, open(tags_output_path, "w") as f_tags_output:
            for src_line, tags_line in zip(f_input, f_tags):
                tags = tags_line.strip().split()
                for tag in tags:
                    assert tag in {"OK", "BAD"}, tag
                src_tokens = src_line.strip().split()
                if len(tags) == len(src_tokens):
                    f_output.write(" ".join(src_tokens[:MAX_SEQ_LEN]) + "\n")
                    f_tags_output.write(" ".join(tags[:MAX_SEQ_LEN]) + "\n")
                    continue
                output_tokens = []
                output_tags = []
                assert len(src_line) % 2 == 0
                for i, (char, tag) in enumerate(zip(src_line, tags)):
                    if i % 2 == 0:
                        if char.isspace():
                            continue
                        output_tokens.append(char)
                        output_tags.append(tag)
                    if len(output_tokens) >= MAX_SEQ_LEN:
                        break
                assert len(output_tags) == len(output_tokens)
                if output_tags:
                    f_output.write(" ".join(output_tokens) + "\n")
                    f_tags_output.write(" ".join(output_tags) + "\n")
                else:
                    f_output.write("<pad>" + "\n")
                    f_tags_output.write("OK" + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-pair")
    parser.add_argument("--src-path")
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--translation-model-name")
    args = parser.parse_args()
    main(args.language_pair, args.src_path, args.out_path, args.translation_model_name)
