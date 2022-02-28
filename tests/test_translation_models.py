from unittest import TestCase

from translation_models import load_translation_model, load_forward_and_backward_model


class TranslationModelTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.src_lang = "en"
        cls.tgt_lang = "de"

    def setUp(self) -> None:
        raise NotImplementedError

    def test_translate(self):
        print(self.model.translate(self.src_lang, self.tgt_lang, ["This is a test."]))

    def test_translate_batched(self):
        print(self.model.translate(self.src_lang, self.tgt_lang, 8 * ["This is a test."]))

    def test_score(self):
        scores = self.model.score(
            self.src_lang, self.tgt_lang,
            source_sentences=(2 * ["This is a test."]),
            hypothesis_sentences=(["Dies ist ein Test.", "Diese Übersetzung ist komplett falsch."]),
        )
        self.assertIsInstance(scores[0], float)
        self.assertIsInstance(scores[1], float)
        scores = self.model.score(
            self.src_lang, self.tgt_lang,
            source_sentences=(2 * ["This is a test."]),
            hypothesis_sentences=(["Diese Übersetzung ist komplett falsch.", "Dies ist ein Test."]),
        )
        self.assertLess(scores[0], scores[1])
        scores = self.model.score(
            self.src_lang, self.tgt_lang,
            source_sentences=(2 * ["This is a test."]),
            hypothesis_sentences=(2 * ["Dies ist ein Test."]),
        )
        self.assertAlmostEqual(scores[0], scores[1], places=4)

    def test_score_batched(self):

        scores = self.model.score(
            self.src_lang, self.tgt_lang,
            source_sentences=(4 * ["This is a test."]),
            hypothesis_sentences=(["Diese Übersetzung ist komplett falsch", "Dies ist ein Test.", "Dies ist ein Test.", "Dies ist ein Test."]),
            batch_size=2,
        )
        self.assertLess(scores[0], scores[1])
        self.assertAlmostEqual(scores[2], scores[1], places=4)
        self.assertAlmostEqual(scores[3], scores[1], places=4)

        scores = self.model.score(
            self.src_lang, self.tgt_lang,
            source_sentences=(["This is a test.", "A translation that is completely wrong.", "This is a test.", "This is a test."]),
            hypothesis_sentences=(4 * ["Dies ist ein Test."]),
            batch_size=2,
        )
        self.assertLess(scores[1], scores[0])
        self.assertAlmostEqual(scores[2], scores[0], places=4)
        self.assertAlmostEqual(scores[3], scores[0], places=4)

        scores = self.model.score(
            self.src_lang, self.tgt_lang,
            source_sentences=(4 * ["This is a test."]),
            hypothesis_sentences=(["Dies ist ein Test.", "Dies ist ein Test.", ".", "Dies ist ein Test."]),
            batch_size=2,
        )
        self.assertAlmostEqual(scores[1], scores[0], places=4)
        self.assertLess(scores[2], scores[0])
        self.assertAlmostEqual(scores[3], scores[0], places=4)

        scores = self.model.score(
            self.src_lang, self.tgt_lang,
            source_sentences=(["This is a test.", "This is a test.", "This is a test.", "A translation that is completely wrong."]),
            hypothesis_sentences=(4 * ["Dies ist ein Test."]),
            batch_size=2,
        )
        self.assertAlmostEqual(scores[1], scores[0], places=4)
        self.assertAlmostEqual(scores[2], scores[0], places=4)
        self.assertLess(scores[3], scores[0])


class MbartTranslationModelTestCase(TranslationModelTestCase):

    def setUp(self) -> None:
        self.model = load_translation_model("mbart50-one-to-many")


class M2M100TranslationModelTestCase(TranslationModelTestCase):

    def setUp(self) -> None:
        self.model = load_translation_model("m2m100_418M")

class ForwardBackwardTestCase(TestCase):

    def setUp(self) -> None:
        self.model_names = [
            "mbart50",
            "m2m100_418M",
        ]
        self.src_lang = "en"
        self.tgt_lang = "de"

    def test_translate(self):
        for model_name in self.model_names:
            forward_model, backward_model = load_forward_and_backward_model(
                name=model_name,
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
            )
            print(forward_model.translate(self.src_lang, self.tgt_lang, ["This is a test."])[0])
            print(backward_model.translate(self.tgt_lang, self.src_lang, ["Dies ist ein Test."])[0])
