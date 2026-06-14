# Text Summarization System using Ensemble ML (NLP)

An ensemble NLP summarization pipeline combining **R-Fuzzy** (fuzzy-logic
extractive summarization) and **BART** (transformer-based abstractive
summarization), benchmarked against standard extractive baselines using
ROUGE-1, ROUGE-2, and ROUGE-L.

## Architecture

```
Raw Document
     |
     v
[preprocessing.py] -> tokenization, stopword removal, lemmatization,
                       term-frequency based semantic scoring
     |
     v
[fuzzy_extractive.py] -> R-Fuzzy stage: a Mamdani fuzzy inference system
                          scores every sentence on title similarity,
                          length, term weight, position, and presence of
                          numeric data, then selects the top-ranked
                          sentences (extractive summary)
     |
     v
[abstractive_bart.py] -> BART (facebook/bart-large-cnn) rewrites the
                          extractive summary into a fluent, condensed
                          abstractive summary
     |
     v
[ensemble.py] -> orchestrates the two stages above (final output)
```

## Files

| File                  | Purpose                                                            |
|-----------------------|---------------------------------------------------------------------|
| `preprocessing.py`    | Tokenization, stopword removal, lemmatization, semantic scoring   |
| `fuzzy_extractive.py` | R-Fuzzy extractive summarizer (fuzzy logic sentence ranking)      |
| `abstractive_bart.py` | BART abstractive summarizer wrapper                                |
| `ensemble.py`         | Combines the fuzzy extractive + BART abstractive stages           |
| `baselines.py`        | Baseline summarizers (LexRank, LSA, Luhn, TextRank) via `sumy`     |
| `evaluate.py`         | ROUGE-1/2/L computation and benchmark table printing               |
| `main.py`             | End-to-end demo: runs ensemble + baselines + ROUGE benchmark       |

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads `facebook/bart-large-cnn` (~1.6GB) from
HuggingFace, so an internet connection is required initially. NLTK
corpora (punkt, stopwords, wordnet) are downloaded automatically.

## Usage

Run the included demo (sample document + reference summary):

```bash
python main.py
```

This will:
1. Run the R-Fuzzy extractive stage and print the selected sentences.
2. Feed that extractive summary into BART to produce the final
   ensemble summary.
3. Run LexRank, LSA, Luhn, and TextRank baselines on the same document.
4. Print a ROUGE-1 / ROUGE-2 / ROUGE-L benchmark table comparing all
   methods against the reference summary.

### Using it on your own text

```python
from ensemble import EnsembleSummarizer

ensemble = EnsembleSummarizer(extractive_ratio=0.5)

result = ensemble.summarize(
    text=my_document,
    title="My Document Title",
    return_stages=True,
)

print(result["extractive_summary"])  # R-Fuzzy stage output
print(result["final_summary"])       # Final BART-rewritten summary
```

## Notes on the R-Fuzzy stage

The fuzzy system uses five input features per sentence, each fuzzified
into `low` / `medium` / `high` triangular membership functions:

- **title_sim** – overlap between sentence tokens and title tokens
- **sentence_len** – normalized sentence length
- **term_weight** – average normalized term-frequency ("semantic score")
  of the sentence's content words
- **sentence_pos** – position in the document (earlier sentences score
  higher, reflecting typical lead-paragraph importance)
- **numeric_data** – whether the sentence contains digits/statistics

A rule base combines these into a single `importance` output, which is
defuzzified (centroid method) into a 0-1 score. The top sentences by
score (controlled by `extractive_ratio`) form the extractive summary
passed on to BART.

## Tuning

- `extractive_ratio` in `EnsembleSummarizer` controls how much of the
  source document is kept before the abstractive stage (lower = more
  aggressive filtering).
- `bart_max_length` / `bart_min_length` in `ensemble.summarize(...)`
  control the final summary length.
- Adjust the fuzzy rule base in `fuzzy_extractive.py` to emphasize
  different features for domain-specific documents (e.g. legal or
  financial text where numeric data matters more).
