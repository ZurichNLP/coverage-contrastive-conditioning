from typing import List, Tuple

import torch


class TranslationModel:

    def set_language_pair(self, src_lang: str, tgt_lang: str):
        raise NotImplementedError

    def translate(self, src_lang: str, tgt_lang: str, sentences: List[str], beam: int = 5, batch_size=2, **kwargs) -> List[str]:
        self.set_language_pair(src_lang, tgt_lang)
        translations = self._translate(sentences, beam=beam, batch_size=batch_size, **kwargs)
        assert len(translations) == len(sentences)
        return translations

    def _translate(self, sentences: List[str], beam: int = 5, batch_size=2, **kwargs) -> List[str]:
        raise NotImplementedError

    @torch.no_grad()
    def score(self, src_lang: str, tgt_lang: str, source_sentences: List[str], hypothesis_sentences: List[str], batch_size=2) -> List[float]:
        self.set_language_pair(src_lang, tgt_lang)
        assert len(source_sentences) == len(hypothesis_sentences)
        scores = self._score(source_sentences, hypothesis_sentences, batch_size)
        assert len(scores) == len(source_sentences)
        return scores

    def _score(self, source_sentences: List[str], hypothesis_sentences: List[str], batch_size=2) -> List[float]:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


def load_translation_model(name: str, device=0) -> TranslationModel:
    if name == "mbart50-one-to-many":
        from translation_models.mbart_models import Mbart50Model
        translation_model = Mbart50Model(model_name_or_path="facebook/mbart-large-50-one-to-many-mmt", device=device)
    elif name == "mbart50-many-to-one":
        from translation_models.mbart_models import Mbart50Model
        translation_model = Mbart50Model(model_name_or_path="facebook/mbart-large-50-many-to-one-mmt", device=device)
    elif name == "m2m100_418M":
        from translation_models.m2m100_models import M2M100Model
        translation_model = M2M100Model(model_name_or_path="facebook/m2m100_418M", device=device)
    elif name == "m2m100_1.2B":
        from translation_models.m2m100_models import M2M100Model
        translation_model = M2M100Model(model_name_or_path="facebook/m2m100_1.2B", device=device)
    else:
        raise NotImplementedError
    return translation_model


def load_forward_and_backward_model(name: str, src_lang: str, tgt_lang: str, forward_device=0, backward_device=1
                                    ) -> Tuple[TranslationModel, TranslationModel]:
    """
    Return the same model twice if it is a many-to-many model, otherwise return two "specialized" models
    """
    if name == "mbart50":
        from translation_models.mbart_models import load_mbart_one_to_many
        forward_model = load_mbart_one_to_many(src_lang, tgt_lang, forward_device)
        backward_model = load_mbart_one_to_many(tgt_lang, src_lang, backward_device)
    else:
        forward_model = load_translation_model(name, forward_device)
        backward_model = forward_model
    return forward_model, backward_model
