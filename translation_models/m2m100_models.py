from typing import List

import torch
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers.file_utils import PaddingStrategy
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from translation_models import TranslationModel
from translation_models.utils import batch


class M2M100Model(TranslationModel):

    def __init__(self,
                 model_name_or_path: str = "facebook/m2m100_418M",
                 device=None,
                 *args,
                 **kwargs,
                 ):
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name_or_path)
        if device is not None:
            self.model = self.model.to(device)
        self.model.config.max_length = max(self.model.config.max_length, self.model.config.max_position_embeddings - 4)

    def __str__(self):
        return self.model_name_or_path.replace("/", "_")

    def set_language_pair(self, src_lang: str, tgt_lang):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

    def _translate(self, sentences: List[str], beam: int = 5, batch_size=2, **kwargs) -> List[str]:
        padding_strategy = PaddingStrategy.LONGEST if batch_size > 1 else PaddingStrategy.DO_NOT_PAD
        translations = []
        if len(sentences) / batch_size > 100:
            batch_iterator = tqdm(list(batch(sentences, batch_size)))
        else:
            batch_iterator = batch(sentences, batch_size)
        for src_sentences in batch_iterator:
            inputs = self.tokenizer._batch_encode_plus(src_sentences, return_tensors="pt",
                                                       padding_strategy=padding_strategy)
            inputs = inputs.to(self.model.device)
            generated_tokens = self.model.generate(**inputs,
                                                   forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
                                                   num_beams=beam)
            translations += self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translations

    def _score(self, source_sentences: List[str], hypothesis_sentences: List[str], batch_size=2) -> List[float]:
        padding_strategy = PaddingStrategy.LONGEST if batch_size > 1 else PaddingStrategy.DO_NOT_PAD
        scores = []
        batch_iterator = zip(batch(source_sentences, batch_size), batch(hypothesis_sentences, batch_size))
        for src_sentences, tgt_sentences in batch_iterator:
            inputs = self.tokenizer._batch_encode_plus(src_sentences, return_tensors="pt",
                                                       padding_strategy=padding_strategy)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer._batch_encode_plus(tgt_sentences, return_tensors="pt",
                                                           padding_strategy=padding_strategy)
            inputs = inputs.to(self.model.device)
            labels = labels.to(self.model.device)
            labels["input_ids"][labels["input_ids"] == self.model.config.pad_token_id] = -100
            inputs["decoder_input_ids"] = shift_tokens_right(labels["input_ids"], self.model.config.pad_token_id)
            output = self.model(**inputs)
            batch_scores = torch.zeros(len(src_sentences), device=self.model.device)
            for i in range(len(src_sentences)):
                loss = torch.nn.CrossEntropyLoss()(
                    output.logits[i][1:].view(-1, self.model.config.vocab_size),
                    labels["input_ids"][i][1:].view(-1),
                )
                batch_scores[i] = -loss
            scores += batch_scores.tolist()
        return scores
