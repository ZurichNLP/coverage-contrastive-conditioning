import stanza
from stanza import Pipeline
from stanza.models.common.doc import Word


def is_ignored_upos(word: Word):
    return word.upos in {
        "ADP",
        "AUX",
        "CCONJ",
        "DET",
        "PART",
        "PRON",
        "SCONJ",
        "PUNCT",
        "SYM",
    }


def load_stanza_parser(language: str) -> Pipeline:
    if language in {"zh"}:
        processors = "tokenize,pos,lemma,depparse"
    else:
        processors = "tokenize,mwt,pos,lemma,depparse"
    return stanza.Pipeline(language, processors=processors, verbose=False)
