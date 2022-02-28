from typing import List

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Text2TextGenerationPipeline
from transformers.file_utils import PaddingStrategy
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from translation_models import TranslationModel
from translation_models.utils import batch


class Mbart50Model(TranslationModel):

    def __init__(self,
                 model_name_or_path: str = "facebook/mbart-large-50-one-to-many-mmt",
                 device=None,
                 *args,
                 **kwargs
                 ):
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
        model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.pipeline = Text2TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=device,
            *args, **kwargs,
        )

    def __str__(self):
        return self.model_name_or_path.replace("/", "_")

    @property
    def is_to_many_model(self):
        return "-to-many-" in self.model_name_or_path

    def set_language_pair(self, src_lang: str, tgt_lang):
        src_lang = get_mbart_language(src_lang)
        tgt_lang = get_mbart_language(tgt_lang)
        assert not self.is_to_many_model or src_lang == "en_XX"
        assert self.is_to_many_model or tgt_lang == "en_XX"
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.pipeline.tokenizer.src_lang = src_lang
        self.pipeline.tokenizer.tgt_lang = tgt_lang

    def _translate(self, sentences: List[str], beam: int = 5, batch_size=2, **kwargs) -> List[str]:
        results = self.pipeline(
            sentences,
            num_beams=beam,
            batch_size=batch_size,
            forced_bos_token_id=(self.pipeline.tokenizer.lang_code_to_id[self.tgt_lang] if self.is_to_many_model else None),
        )
        return [result["generated_text"] for result in results]

    def _score(self, source_sentences: List[str], hypothesis_sentences: List[str], batch_size=2) -> List[float]:
        padding_strategy = PaddingStrategy.LONGEST if batch_size > 1 else PaddingStrategy.DO_NOT_PAD
        scores = []
        for src_sentences, tgt_sentences in zip(batch(source_sentences, batch_size), batch(hypothesis_sentences, batch_size)):
            inputs = self.pipeline.tokenizer._batch_encode_plus(src_sentences, return_tensors="pt", padding_strategy=padding_strategy)
            with self.pipeline.tokenizer.as_target_tokenizer():
                labels = self.pipeline.tokenizer._batch_encode_plus(tgt_sentences, return_tensors="pt", padding_strategy=padding_strategy)
            inputs = self.pipeline.ensure_tensor_on_device(**inputs)
            labels = labels.to(self.pipeline.device)
            labels["input_ids"][labels["input_ids"] == self.pipeline.model.config.pad_token_id] = -100
            inputs["decoder_input_ids"] = shift_tokens_right(labels["input_ids"], self.pipeline.model.config.pad_token_id)
            output = self.pipeline.model(**inputs)
            batch_scores = torch.zeros(len(src_sentences), device=self.pipeline.device)
            for i in range(len(src_sentences)):
                loss = torch.nn.CrossEntropyLoss()(
                    output.logits[i][1:].view(-1, self.pipeline.model.config.vocab_size),
                    labels["input_ids"][i][1:].view(-1),
                )
                batch_scores[i] = -loss
            scores += batch_scores.tolist()
        assert len(scores) == len(source_sentences)
        return scores


def load_mbart_one_to_many(src_lang: str, tgt_lang: str, device: int = None) -> Mbart50Model:
    assert "en" in {src_lang, tgt_lang}
    assert src_lang != tgt_lang
    if src_lang == "en":
        model_name = "facebook/mbart-large-50-one-to-many-mmt"
    else:
        model_name = "facebook/mbart-large-50-many-to-one-mmt"
    return Mbart50Model(model_name_or_path=model_name, device=device)


def get_mbart_language(language: str):
    mbart_language_codes = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN",
                            "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO",
                            "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL",
                            "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF",
                            "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA",
                            "gl_ES", "sl_SI"]
    code_dict = {code.split("_")[0]: code for code in mbart_language_codes}
    if "_" in language:
        assert language in code_dict.values()
        return language
    else:
        return code_dict[language]
