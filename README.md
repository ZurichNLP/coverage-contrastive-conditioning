
Data and code for the paper ["As Little as Possible, as Much as Necessary: Detecting Over- and Undertranslations with Contrastive Conditioning"](https://openreview.net/pdf?id=txfPhtRZ_SW) (ACL 2022).
- Detect potential coverage errors simply using NMT models (e.g. from the Hugging Face Hub)
- A dataset of synthetic coverage errors in EN–DE and ZH–EN machine translations

## Installation

- Prerequisites: Python >= 3.8, PyTorch
- `pip install -r requirements.txt`
- Download dependency parsers for the languages used in the paper: 
```python
import stanza
stanza.download("en")
stanza.download("de")
stanza.download("zh-hans")
```

### QE baseline (requires separate Python environment)
- `pip install -r baseline/requirements-baseline.txt`


## Usage Examples

### Manual demonstration: Reproducing Figure 1
```python
import math
from translation_models import load_translation_model

translation_model = load_translation_model("mbart50-one-to-many")

src = "Please exit the plane after landing."
tgt = "Bitte verlassen Sie das Flugzeug."

partial_sources = [
    "exit the plane after landing.",  # "Please" deleted
    "Please exit after landing.",  # "the plane" deleted
    "Please exit the plane.",  # "after landing" deleted
]

scores = translation_model.score(
    src_lang="en",
    tgt_lang="de",
    source_sentences=[src] + partial_sources,
    hypothesis_sentences=(4 * [tgt]),
)
print(list(map(math.exp, scores)))
# [0.3387762759789361, 0.14397865706857962, 0.20369576359159763, 0.7212898669689275]
```

### Using the CoverageEvaluator class

```python
from coverage.evaluator import CoverageEvaluator
from translation_models import load_forward_and_backward_model

forward_model, backward_model = load_forward_and_backward_model("mbart50", src_lang="en", tgt_lang="de")

evaluator = CoverageEvaluator(
  src_lang="en",
  tgt_lang="de",
  forward_evaluator=forward_model,
  backward_evaluator=backward_model,
)
result = evaluator.detect_errors(
  src="Please exit the plane after landing.",
  translation="Bitte verlassen Sie das Flugzeug.",
)
print(result)
# "Omission errors: after landing"
```

## NMT models
This repo implements coverage error detection with the following NMT models:
- [mbart-large-50-one-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt) / [mbart-large-50-many-to-one-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-one-mmt) (used for the paper)
- [m2m100_418M](https://huggingface.co/facebook/m2m100_418M)
- [m2m100_1.2B](https://huggingface.co/facebook/m2m100_1.2B)

Custom NMT models can be added by subclassing `translation_models.TranslationModel`.

## Reproducing the results

### Segment-Level Comparison to Gold Data

#### Download data
- Download the following files from https://github.com/google/wmt-mqm-human-evaluation:
  - `mqm_newstest2020_ende.tsv`
  - `mqm_newstest2020_zhen.tsv`
- Place the files in `data/mqm`

#### Run contrastive conditioning
```shell
python -m evaluation.predict_mqm \
  --model-name mbart50 \
  --language-pair en-de \
  > predictions/out.jsonl
```

#### Evaluate contrastive conditioning
```shell
python -m evaluation.evaluate_mqm \
  --language-pair en-de \
  --split test \
  --predictions-path predictions/out.jsonl
```

#### Train the QE baseline
Activate the baseline Python environment
```shell
kiwi train baseline/config.en-de.yaml
```

#### Evaluate the baseline
```shell
python -m evaluation.evaluate_mqm_baseline \
  --language-pair en-de \
  --split test \
  --checkpoint-path <checkpoint_path> \
  --nmt-predictions-path predictions/out.jsonl
```

### Synthetic Data

#### Download dataset
- `wget https://files.ifi.uzh.ch/cl/archiv/2022/mutamur/coverage_data.zip`
- `unzip coverage_data.zip -d data/synthetic`

#### Alternative: Generate new synthetic data
Prepare monolingual sentence lines (e.g. download English text from http://data.statmt.org/news-crawl/)
```shell
python -m data_generation.generate \
  --language-pair en-de \
  --src-path <file containing source sentence lines> \
  --translation-model-name mbart50-one-to-many
```

#### Run contrastive conditioning on synthetic data
```shell
python -m evaluation.predict_synthetic_testset \
  --model-name mbart50 \
  --language-pair en-de \
  --dataset-name data/synthetic/en-de.test
```

#### Run QE baseline on synthetic data
Activate the baseline Python environment
```shell
kiwi predict baseline/predict.en-de.yaml system.load=<checkpoint_path>
```

#### Evaluate both on synthetic test set
```shell
python -m evaluation.evaluate_on_synthetic_data \
  --nmt-predicted-source-tags-path predictions/en-de.test.mbart50.source_tags \
  --baseline-predicted-source-tags-labels-path <path> \
  --gold-source-tags-path data/synthetic/en-de.test.source_tags \
  --gold-target-tags-path data/synthetic/en-de.test.tags
```

## License
- Code: MIT License
- Data: See READMEs in the respective subdirectories

## Citation
```bibtex
@inproceedings{vamvas-sennrich-2022-coverage,
    title = "As Little as Possible, as Much as Necessary: Detecting Over- and Undertranslations with Contrastive Conditioning",
    author = "Vamvas, Jannis  and
      Sennrich, Rico",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://openreview.net/pdf?id=txfPhtRZ_SW",
}
```