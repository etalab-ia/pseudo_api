import copy
import itertools
from collections import defaultdict, Counter
from string import ascii_uppercase
from typing import Callable, List, Tuple

from flair.data import Token, Sentence
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesPunctNormalizer
from sqlitedict import SqliteDict

moses_detokenize = MosesDetokenizer(lang="fr")
import stopwatch

sw = stopwatch.StopWatch()


def read_db():
    stats = SqliteDict('./my_db.sqlite', autocommit=True)
    pass


def create_conll_output(sentences_tagged: List[Sentence]):
    tags = []
    conll_str: str = ""
    for sent_pred in sentences_tagged:
        tags.extend([s.tag for s in sent_pred.get_spans("ner")])
        for tok_pred in sent_pred:
            result_str = f"{moses_detokenize.tokenize([tok_pred.text])}\t{tok_pred.get_tag('ner').value}\t" \
                         f"{tok_pred.start_pos}\t{tok_pred.end_pos}"
            conll_str += result_str + "\n"
        conll_str += "\n"
    return conll_str, Counter(tags)


def prepare_output(text: str, tagger, word_tokenizer=None, output_type: str = "pseudonymized"):
    stats_dict = {}
    with sw.timer("root"):
        if not word_tokenizer:
            tokenizer = MOSES_TOKENIZER
        else:
            tokenizer = word_tokenizer

        text = [t.strip() for t in text.split("\n") if t.strip()]
        with sw.timer('model_annotation'):
            sentences_tagged = tagger.predict(sentences=text,
                                              mini_batch_size=32,
                                              embedding_storage_mode="none",
                                              use_tokenizer=tokenizer,
                                              verbose=True)

        if output_type == "conll":
            api_output, tags_stats = create_conll_output(sentences_tagged=sentences_tagged)
        elif output_type == "tagged":
            api_output, tags_stats = create_tagged_text(sentences_tagged=sentences_tagged)
        elif output_type == "pseudonymized":
            api_output, tags_stats = create_pseudonymized_text(sentences_tagged=sentences_tagged)

        # deal with stats
        stats_dict["nb_analyzed_sentences"] = len(text)
        stats_dict.update(tags_stats)
        return api_output, stats_dict


def update_stats(analysis_stats: dict, analysis_ner_stats: dict, time_info: stopwatch.AggregatedReport,
                 output_type: str):
    def update_averages(avg: float, size: int, value: float):
        return (size * avg + value) / (size + 1)

    def update_dict_values(old_dict: dict, new_dict: dict):
        for k, v in new_dict.items():
            if k in old_dict:
                old_dict[k] += v


    # get previous values
    old_nb_analyzed_documents = analysis_stats.get("nb_analyzed_documents", 0)
    old_nb_analyzed_sentences = analysis_stats.get("nb_analyzed_sentences", 0)
    old_output_types_freq = analysis_stats.get(f"output_type_{output_type}", 0)
    old_avg_time = analysis_stats.get("avg_time_per_doc", 0)
    old_avg_time_per_sent = analysis_stats.get("avg_time_per_sentence", 0)

    analysis_stats["nb_analyzed_documents"] = old_nb_analyzed_documents + 1
    analysis_stats["nb_analyzed_sentences"] = old_nb_analyzed_sentences + analysis_ner_stats.pop(
        "nb_analyzed_sentences")

    # add entities tags freqs
    update_dict_values(analysis_stats, analysis_ner_stats)
    # analysis_stats.update(analysis_ner_stats)

    # deal with time stats
    delta_ms, _, _ = time_info.aggregated_values["root"]
    analysis_stats["avg_time_per_doc"] = update_averages(old_avg_time,
                                                         old_nb_analyzed_documents, delta_ms)
    analysis_stats["avg_time_per_sentence"] = update_averages(old_avg_time_per_sent,
                                                          old_nb_analyzed_sentences,
                                                          delta_ms / analysis_stats[
                                                              "nb_analyzed_sentences"])

    analysis_stats[f"output_type_{output_type}"] = old_output_types_freq + 1


def create_tagged_text(sentences_tagged: List[Sentence]):
    # Iterate over the modified sentences to recreate the text (tagged)
    tagged_str = ""
    tags = []
    for sent in sentences_tagged:
        tags.extend([s.tag for s in sent.get_spans("ner")])
        temp_str = sent.to_tagged_string()
        tagged_str += temp_str + "\n\n"

    return tagged_str, Counter(tags)


def create_pseudonymized_text(sentences_tagged: List[Sentence]):
    singles = [f"{letter}..." for letter in ascii_uppercase]
    doubles = [f"{a}{b}..." for a, b in list(itertools.combinations(ascii_uppercase, 2))]
    pseudos = singles + doubles
    pseudo_entity_dict = {}
    sentences_pseudonymized = copy.deepcopy(sentences_tagged)
    tag_stats = defaultdict(int)

    # Replace the entities within the sentences themselves
    for id_sn, sent in enumerate(sentences_pseudonymized):
        for sent_span in sent.get_spans("ner"):
            if "LOC" in sent_span.tag:
                for id_tok in range(len(sent_span.tokens)):
                    tag_stats[sent_span.tokens[id_tok].get_tag('ner').value] += 1
                    sent_span.tokens[id_tok].text = "..."
            else:
                for id_tok, token in enumerate(sent_span.tokens):
                    tag_stats[token.get_tag('ner').value] += 1
                    replacement = pseudo_entity_dict.get(token.text.lower(), pseudos.pop(0))
                    pseudo_entity_dict[token.text.lower()] = replacement
                    sent_span.tokens[id_tok].text = replacement

    # Iterate over the modified sentences to recreate the text (pseudonymized)
    pseudonymized_str = ""
    for sent in sentences_pseudonymized:
        detokenized_str = moses_detokenize.detokenize([t.text for t in sent.tokens],
                                                      return_str=True)
        pseudonymized_str += detokenized_str + "\n\n"

    return pseudonymized_str, tag_stats


def create_api_output(sentences_tagged: List[Sentence]) -> Tuple[str, str]:
    "We create two output texts: tagged and pseudonyimzed"
    tagged_str = create_tagged_text(sentences_tagged=sentences_tagged)
    pseudonymized_str = create_pseudonymized_text(sentences_tagged=sentences_tagged)

    return tagged_str, pseudonymized_str


# ENTITIES = {"PER_PRENOM": "PRENOM", "PER_NOM": "NOM", "LOC": "ADRESSE"}

class MosesTokenizerSpans(MosesTokenizer):
    def __init__(self, lang="en", custom_nonbreaking_prefixes_file=None):
        MosesTokenizer.__init__(self, lang=lang,
                                custom_nonbreaking_prefixes_file=custom_nonbreaking_prefixes_file)
        self.lang = lang

    def span_tokenize(
            self,
            text,
            aggressive_dash_splits=False,
            escape=True,
            protected_patterns=None,
    ):
        # https://stackoverflow.com/a/35634472
        import re
        detokenizer = MosesDetokenizer(lang=self.lang)
        tokens = self.tokenize(text=text, aggressive_dash_splits=aggressive_dash_splits,
                               return_str=False, escape=escape,
                               protected_patterns=protected_patterns)
        tail = text
        accum = 0
        tokens_spans = []
        for token in tokens:
            detokenized_token = detokenizer.detokenize(tokens=[token],
                                                       return_str=True,
                                                       unescape=escape)
            escaped_token = re.escape(detokenized_token)

            m = re.search(escaped_token, tail)
            tok_start_pos, tok_end_pos = m.span()
            sent_start_pos = accum + tok_start_pos
            sent_end_pos = accum + tok_end_pos
            accum += tok_end_pos
            tail = tail[tok_end_pos:]

            tokens_spans.append((detokenized_token, (sent_start_pos, sent_end_pos)))
        return tokens_spans


def build_moses_tokenizer(tokenizer: MosesTokenizerSpans,
                          normalizer: MosesPunctNormalizer = None) -> Callable[[str], List[Token]]:
    """
    Wrap Spacy model to build a tokenizer for the Sentence class.
    :param model a Spacy V2 model
    :return a tokenizer function to provide to Sentence class constructor
    """
    try:
        from sacremoses import MosesTokenizer
        from sacremoses import MosesPunctNormalizer
    except ImportError:
        raise ImportError(
            "Please install sacremoses or better before using the Spacy tokenizer, otherwise you can use segtok_tokenizer as advanced tokenizer."
        )

    moses_tokenizer: MosesTokenizerSpans = tokenizer
    if normalizer:
        normalizer: MosesPunctNormalizer = normalizer

    def tokenizer(text: str) -> List[Token]:
        if normalizer:
            text = normalizer.normalize(text=text)
        doc = moses_tokenizer.span_tokenize(text=text, escape=False)
        previous_token = None
        tokens: List[Token] = []
        for word, (start_pos, end_pos) in doc:
            word: str = word
            token = Token(
                text=word, start_position=start_pos, whitespace_after=True
            )
            tokens.append(token)

            if (previous_token is not None) and (
                    token.start_pos - 1
                    == previous_token.start_pos + len(previous_token.text)
            ):
                previous_token.whitespace_after = False

            previous_token = token
        return tokens

    return tokenizer


MOSES_TOKENIZER = build_moses_tokenizer(tokenizer=MosesTokenizerSpans(lang="fr"),
                                        normalizer=MosesPunctNormalizer(lang="fr"))
