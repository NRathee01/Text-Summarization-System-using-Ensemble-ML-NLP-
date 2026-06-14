"""
text_summarization_ensemble.py
================================
End-to-end Ensemble Text Summarization System (single file)

Pipeline
--------
1. Preprocessing      : tokenization, stopword removal, lemmatization,
                         term-frequency based "semantic scoring"
2. Extractive stage   : R-Fuzzy - a Mamdani fuzzy inference system ranks
                         sentences on title similarity, length, term
                         weight, position, and numeric-data presence,
                         then selects the top-ranked sentences.
3. Abstractive stage  : BART (facebook/bart-large-cnn) rewrites the
                         extractive summary into a fluent, condensed
                         abstractive summary.
4. Baselines          : LexRank, LSA, Luhn, TextRank (via `sumy`) for
                         comparison.
5. Evaluation         : ROUGE-1 / ROUGE-2 / ROUGE-L benchmark table.

Setup
-----
    pip install torch transformers nltk scikit-fuzzy numpy scipy \
                sumy rouge-score lxml

Usage
-----
    python text_summarization_ensemble.py
"""

import re

import numpy as np
import nltk
import skfuzzy as fuzz
from skfuzzy import control as ctrl

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

from rouge_score import rouge_scorer


# ============================================================
# 1. PREPROCESSING
# ============================================================

for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()


def split_sentences(text):
    """Split raw text into a list of sentences."""
    return sent_tokenize(text)


def tokenize_and_clean(sentence, language="english"):
    """Lowercase, tokenize, remove stopwords/punctuation, lemmatize."""
    stop_words = set(stopwords.words(language))
    tokens = word_tokenize(sentence.lower())
    return [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in stop_words
    ]


def build_term_frequencies(sentences, language="english"):
    """Normalized (0-1) term-frequency table across all sentences."""
    freq = {}
    for sent in sentences:
        for tok in tokenize_and_clean(sent, language):
            freq[tok] = freq.get(tok, 0) + 1
    if not freq:
        return {}
    max_freq = max(freq.values())
    return {tok: count / max_freq for tok, count in freq.items()}


def semantic_score(sentence, term_freq, language="english"):
    """Average term-frequency weight of a sentence's content words."""
    tokens = tokenize_and_clean(sentence, language)
    if not tokens:
        return 0.0
    return sum(term_freq.get(tok, 0.0) for tok in tokens) / len(tokens)


def contains_number(sentence):
    """True if the sentence contains digits (often a factual signal)."""
    return bool(re.search(r"\d", sentence))


# ============================================================
# 2. R-FUZZY EXTRACTIVE SUMMARIZER
# ============================================================

class FuzzyExtractiveSummarizer:
    """
    Mamdani fuzzy inference system over five sentence features:
      - title_sim    : overlap with title tokens
      - sentence_len : normalized sentence length
      - term_weight  : semantic (TF-based) score
      - sentence_pos : position in document (earlier = higher)
      - numeric_data : presence of digits/statistics
    Output: 'importance' score in [0, 1], used to rank sentences.
    """

    def __init__(self, language="english"):
        self.language = language
        self.system = self._build_fuzzy_system()

    def _build_fuzzy_system(self):
        universe = np.arange(0, 1.01, 0.01)

        title_sim = ctrl.Antecedent(universe, "title_sim")
        sentence_len = ctrl.Antecedent(universe, "sentence_len")
        term_weight = ctrl.Antecedent(universe, "term_weight")
        sentence_pos = ctrl.Antecedent(universe, "sentence_pos")
        numeric_data = ctrl.Antecedent(universe, "numeric_data")
        importance = ctrl.Consequent(universe, "importance")

        for var in (title_sim, sentence_len, term_weight,
                    sentence_pos, numeric_data, importance):
            var["low"] = fuzz.trimf(var.universe, [0, 0, 0.5])
            var["medium"] = fuzz.trimf(var.universe, [0, 0.5, 1])
            var["high"] = fuzz.trimf(var.universe, [0.5, 1, 1])

        rules = [
            ctrl.Rule(term_weight["high"] & title_sim["high"], importance["high"]),
            ctrl.Rule(term_weight["high"] & sentence_pos["high"], importance["high"]),
            ctrl.Rule(numeric_data["high"] & term_weight["medium"], importance["high"]),
            ctrl.Rule(title_sim["high"] & sentence_pos["medium"], importance["high"]),
            ctrl.Rule(numeric_data["high"] & sentence_pos["high"], importance["high"]),
            ctrl.Rule(term_weight["medium"] & title_sim["medium"], importance["medium"]),
            ctrl.Rule(sentence_pos["high"] & term_weight["low"], importance["medium"]),
            ctrl.Rule(term_weight["medium"] & sentence_pos["medium"], importance["medium"]),
            ctrl.Rule(sentence_len["high"] & term_weight["low"], importance["low"]),
            ctrl.Rule(term_weight["low"] & title_sim["low"] & sentence_pos["low"], importance["low"]),
        ]

        return ctrl.ControlSystem(rules)

    def _sentence_features(self, sentences, title, term_freq):
        n = len(sentences)
        title_tokens = set(tokenize_and_clean(title, self.language)) if title else set()

        features = []
        for i, sent in enumerate(sentences):
            tokens = tokenize_and_clean(sent, self.language)

            t_weight = semantic_score(sent, term_freq, self.language)

            t_sim = (
                len(set(tokens) & title_tokens) / len(title_tokens)
                if title_tokens else 0.0
            )

            raw_len = len(sent.split())
            s_len = min(raw_len / 30.0, 1.0)

            s_pos = 1.0 - (i / max(n - 1, 1))

            num_flag = 1.0 if contains_number(sent) else 0.0

            features.append({
                "title_sim": t_sim,
                "sentence_len": s_len,
                "term_weight": t_weight,
                "sentence_pos": s_pos,
                "numeric_data": num_flag,
            })
        return features

    def score_sentences(self, sentences, title=""):
        term_freq = build_term_frequencies(sentences, self.language)
        features = self._sentence_features(sentences, title, term_freq)

        scores = []
        for feat in features:
            sim = ctrl.ControlSystemSimulation(self.system)
            for key, value in feat.items():
                sim.input[key] = float(np.clip(value, 0, 1))
            try:
                sim.compute()
                scores.append(sim.output["importance"])
            except Exception:
                scores.append(float(np.mean(list(feat.values()))))
        return scores

    def summarize(self, text, title="", ratio=0.4, min_sentences=2, max_sentences=15):
        """
        Returns:
            summary_text (str)
            summary_sentences (list[str])
            selected_indices (list[int]) - in document order
        """
        sentences = split_sentences(text)
        if len(sentences) <= min_sentences:
            return text, sentences, list(range(len(sentences)))

        scores = self.score_sentences(sentences, title)

        n_select = int(np.ceil(len(sentences) * ratio))
        n_select = max(min_sentences, min(n_select, max_sentences, len(sentences)))

        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        selected = sorted(ranked[:n_select])

        summary_sentences = [sentences[i] for i in selected]
        return " ".join(summary_sentences), summary_sentences, selected


# ============================================================
# 3. BART ABSTRACTIVE SUMMARIZER
# ============================================================

class BartAbstractiveSummarizer:
    """Wraps facebook/bart-large-cnn for abstractive summarization."""

    def __init__(self, model_name="facebook/bart-large-cnn", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def summarize(self, text, max_length=150, min_length=30, num_beams=4):
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# ============================================================
# 4. ENSEMBLE SUMMARIZER (R-Fuzzy + BART)
# ============================================================

class EnsembleSummarizer:
    """
    Stage 1 (extractive)  -> R-Fuzzy fuzzy-logic sentence ranking
    Stage 2 (abstractive) -> BART rewrites the extractive summary
    """

    def __init__(self, extractive_ratio=0.5, bart_model="facebook/bart-large-cnn"):
        self.extractive_ratio = extractive_ratio
        self.extractor = FuzzyExtractiveSummarizer()
        self.abstractor = BartAbstractiveSummarizer(model_name=bart_model)

    def summarize(self, text, title="", bart_max_length=150,
                   bart_min_length=30, return_stages=False):
        extractive_summary, sentences, indices = self.extractor.summarize(
            text, title=title, ratio=self.extractive_ratio
        )

        final_summary = self.abstractor.summarize(
            extractive_summary,
            max_length=bart_max_length,
            min_length=bart_min_length,
        )

        if return_stages:
            return {
                "extractive_summary": extractive_summary,
                "selected_sentence_indices": indices,
                "final_summary": final_summary,
            }
        return final_summary


# ============================================================
# 5. BASELINE SUMMARIZERS (LexRank, LSA, Luhn, TextRank)
# ============================================================

_BASELINE_SUMMARIZERS = {
    "LexRank": LexRankSummarizer,
    "LSA": LsaSummarizer,
    "Luhn": LuhnSummarizer,
    "TextRank": TextRankSummarizer,
}


def baseline_summarize(text, method="TextRank", sentence_count=3, language="english"):
    if method not in _BASELINE_SUMMARIZERS:
        raise ValueError(
            f"Unknown baseline method '{method}'. Choose from {list(_BASELINE_SUMMARIZERS)}"
        )
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = _BASELINE_SUMMARIZERS[method]()
    sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in sentences)


def run_all_baselines(text, sentence_count=3, language="english"):
    return {
        name: baseline_summarize(text, method=name, sentence_count=sentence_count, language=language)
        for name in _BASELINE_SUMMARIZERS
    }


# ============================================================
# 6. ROUGE EVALUATION
# ============================================================

def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def benchmark_summaries(reference, summaries):
    return {name: compute_rouge(reference, text) for name, text in summaries.items()}


def print_benchmark_table(results):
    header = f"{'Method':<28}{'ROUGE-1':>10}{'ROUGE-2':>10}{'ROUGE-L':>10}"
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        print(
            f"{name:<28}{scores['rouge1']:>10.4f}"
            f"{scores['rouge2']:>10.4f}{scores['rougeL']:>10.4f}"
        )


# ============================================================
# 7. DEMO / ENTRY POINT
# ============================================================

TITLE = "Global Shift Toward Renewable Energy"

DOCUMENT = """
Over the past decade, renewable energy sources such as solar and wind
have grown from a niche segment of the global power mix into a
mainstream pillar of electricity generation. In 2023, renewables
accounted for more than 30 percent of global electricity production
for the first time, driven largely by record additions of solar
photovoltaic capacity in China, the United States, and the European
Union. Falling technology costs have been the primary driver of this
transition. The cost of solar panels has dropped by roughly 90 percent
since 2010, while onshore wind costs have fallen by around 70 percent
over the same period. Battery storage costs have also declined sharply,
making it economically viable to pair renewable generation with storage
to smooth out supply fluctuations. Governments have reinforced this
market-driven shift with policy support, including tax credits, feed-in
tariffs, and binding national targets for emissions reductions. The
European Union, for example, has committed to sourcing at least 42.5
percent of its energy from renewable sources by 2030. Despite this
progress, significant challenges remain. Grid infrastructure in many
countries was designed for centralized fossil-fuel power plants and
requires substantial upgrades to handle distributed and intermittent
renewable generation. Supply chains for critical materials such as
lithium, cobalt, and rare-earth elements used in batteries and turbines
are also under strain, raising concerns about cost volatility and
geopolitical dependencies. Analysts broadly agree that continued
investment in grid modernization, energy storage, and domestic
manufacturing capacity will be essential if the current pace of
renewable adoption is to be sustained through the end of the decade.
""".strip()

REFERENCE_SUMMARY = (
    "Renewables surpassed 30 percent of global electricity generation in "
    "2023, driven by falling solar and wind costs and supportive "
    "government policy. However, outdated grid infrastructure and strained "
    "critical material supply chains remain major obstacles to sustaining "
    "this growth."
)


def main():
    print("Loading models (first run downloads BART weights, ~1.6GB)...\n")
    ensemble = EnsembleSummarizer(extractive_ratio=0.5)

    print("Running ensemble (R-Fuzzy + BART) summarizer...")
    result = ensemble.summarize(DOCUMENT, title=TITLE, return_stages=True)

    print("\n--- Stage 1: R-Fuzzy extractive summary ---")
    print(result["extractive_summary"])
    print(f"\nSelected sentence indices: {result['selected_sentence_indices']}")

    print("\n--- Stage 2: Final ensemble (BART) summary ---")
    print(result["final_summary"])

    print("\nRunning baseline summarizers (LexRank, LSA, Luhn, TextRank)...")
    baseline_summaries = run_all_baselines(DOCUMENT, sentence_count=3)

    all_summaries = {"Ensemble (R-Fuzzy + BART)": result["final_summary"]}
    all_summaries.update(baseline_summaries)

    print("\n--- ROUGE Benchmark vs. Reference Summary ---")
    results = benchmark_summaries(REFERENCE_SUMMARY, all_summaries)
    print_benchmark_table(results)


if __name__ == "__main__":
    main()
