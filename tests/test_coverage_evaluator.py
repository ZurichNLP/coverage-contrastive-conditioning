import logging
from unittest import TestCase

from coverage.evaluator import CoverageEvaluator
from translation_models import load_forward_and_backward_model


class CoverageEvaluatorTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        forward_model, backward_model = load_forward_and_backward_model(cls.model_name, src_lang="en", tgt_lang="de")
        cls.evaluator = CoverageEvaluator(
            src_lang="en",
            tgt_lang="de",
            forward_evaluator=forward_model,
            backward_evaluator=backward_model,
        )

    def test_no_error(self):
        result = self.evaluator.detect_errors(
            src="Please exit the plane after landing.",
            translation="Bitte verlassen Sie das Flugzeug nach der Landung.",
        )
        self.assertFalse(result.addition_errors)
        self.assertFalse(result.omission_errors)
        self.assertEqual("en", result.src_lang)
        self.assertEqual("de", result.tgt_lang)
        self.assertFalse(result.is_multi_sentence_input)
        result = self.evaluator.detect_errors(
            src="Please exit the plane.",
            translation="Bitte verlassen Sie das Flugzeug.",
        )
        self.assertFalse(result.addition_errors)
        self.assertFalse(result.omission_errors)

    def test_addition_error(self):
        result = self.evaluator.detect_errors(
            src="Please exit the plane after landing.",
            translation="Bitte verlassen Sie das kleine Flugzeug nach der Landung.",
        )
        self.assertTrue(result.addition_errors)
        self.assertFalse(result.omission_errors)
        print(result)

    def test_omission_error(self):
        result = self.evaluator.detect_errors(
            src="Please exit the plane after landing.",
            translation="Bitte verlassen Sie das Flugzeug.",
        )
        self.assertFalse(result.addition_errors)
        self.assertTrue(result.omission_errors)
        print(result)

    def test_both(self):
        result = self.evaluator.detect_errors(
            src="Please exit the plane after landing.",
            translation="Bitte verlassen Sie das kleine Flugzeug.",
        )
        self.assertTrue(result.addition_errors)
        self.assertTrue(result.omission_errors)
        print(result)

    def test_multiple_sentences(self):
        with self.assertLogs(level=logging.WARNING):
            result = self.evaluator.detect_errors(
                src="This is a test. And here comes a second sentence.",
                translation="Dies ist ein Test. Und hier kommt noch ein zweiter Satz.",
            )
        self.assertTrue(result.is_multi_sentence_input)
        self.assertFalse(result.addition_errors)
        self.assertFalse(result.omission_errors)


class MbartCoverageEvaluatorTestCase(CoverageEvaluatorTestCase):
    model_name = "mbart50"


class M2M100CoverageEvaluatorTestCase(CoverageEvaluatorTestCase):
    model_name = "m2m100_418M"