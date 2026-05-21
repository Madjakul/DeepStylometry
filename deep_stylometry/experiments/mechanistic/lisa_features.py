# deep_stylometry/experiments/mechanistic/lisa_features.py
"""Phase 1a: LISA-style linguistic feature extraction."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed vocabulary lists
# ---------------------------------------------------------------------------

_FUNCTION_WORDS_150 = [
    "a", "about", "above", "across", "after", "again", "against", "all", "almost",
    "along", "already", "also", "although", "always", "am", "among", "an", "and",
    "another", "any", "are", "around", "as", "at", "away", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "come",
    "could", "did", "do", "does", "doing", "done", "down", "during", "each",
    "either", "else", "enough", "even", "ever", "every", "except", "few", "for",
    "from", "further", "get", "got", "had", "has", "have", "having", "he", "hence",
    "her", "here", "herself", "him", "himself", "his", "how", "however", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "know", "let",
    "like", "made", "many", "may", "me", "might", "more", "most", "much", "must",
    "my", "myself", "near", "neither", "never", "no", "nor", "not", "now", "of",
    "off", "often", "on", "once", "only", "or", "other", "our", "out", "over",
    "own", "part", "rather", "really", "same", "she", "should", "since", "so",
    "some", "still", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "though", "through", "thus", "to",
    "too", "toward", "under", "unless", "until", "up", "upon", "us", "very",
    "was", "we", "were", "what", "when", "where", "whether", "which", "while",
    "who", "whom", "will", "with", "within", "would", "yet", "you", "your",
]

_HEDGE_WORDS = [
    "might", "could", "perhaps", "possibly", "seems", "suggest", "suggests",
    "suggested", "appear", "appears", "appeared", "indicate", "indicates",
    "indicated", "possibly", "probably", "likely", "unlikely", "uncertain",
    "assume", "assumes", "supposed", "believe", "believes", "think", "thinks",
    "generally", "approximately", "roughly", "somewhat",
]

_DISCOURSE_MARKERS = [
    "however", "therefore", "moreover", "furthermore", "consequently", "thus",
    "hence", "nevertheless", "nonetheless", "accordingly", "additionally",
    "alternatively", "besides", "conversely", "finally", "firstly", "secondly",
    "thirdly", "in addition", "in contrast", "in fact", "in particular",
    "in summary", "in conclusion", "on the other hand", "on the contrary",
    "for example", "for instance", "that is", "in other words", "as a result",
    "as such", "even so", "at the same time",
]

# POS bigram vocabulary (top 50) — deterministic ordering
_POS_BIGRAM_VOCAB = [
    "NN_NN", "DT_NN", "IN_DT", "NN_IN", "JJ_NN", "NNS_IN", "IN_NN", "VBZ_DT",
    "VBZ_NN", "DT_JJ", "NN_VBZ", "IN_NNS", "NNS_NN", "NN_NNS", "VBD_DT",
    "RB_JJ", "JJ_NNS", "VBN_IN", "NNP_NNP", "VB_DT", "NN_CC", "CC_DT",
    "VBG_NN", "DT_NNS", "PRP_VBZ", "PRP_VBD", "MD_VB", "IN_JJ", "WDT_VBZ",
    "NN_MD", "TO_VB", "VBP_DT", "VBP_NN", "DT_NN_NN", "NNS_VBP", "RB_VBN",
    "WP_VBZ", "VBN_DT", "NNS_VBD", "JJR_NN", "VBZ_JJ", "DT_VBN", "NN_VBD",
    "NNS_CC", "RB_RB", "IN_VBG", "VBG_IN", "CC_NN", "NN_TO", "VB_NN",
]


# ---------------------------------------------------------------------------
# Lazy model singletons
# ---------------------------------------------------------------------------

_SPACY_NLP = None
_SAT_SPLITTER = None


def _get_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


def _get_sat_splitter():
    global _SAT_SPLITTER
    if _SAT_SPLITTER is None:
        try:
            from wtpsplit import SaT
            _SAT_SPLITTER = SaT("sat-3l")
        except Exception:
            _SAT_SPLITTER = False
    return _SAT_SPLITTER if _SAT_SPLITTER is not False else None


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Sentence splitting: wtpsplit if available, else simple regex."""
    splitter = _get_sat_splitter()
    if splitter is not None:
        try:
            return splitter.split(text)
        except Exception:
            pass
    # Regex fallback
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s.strip()]


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def _function_words(tokens: List[str]) -> np.ndarray:
    """150-dim: normalised count of each function word."""
    word_lower = [t.lower() for t in tokens]
    total = max(len(tokens), 1)
    vec = np.zeros(len(_FUNCTION_WORDS_150), dtype=np.float32)
    for i, fw in enumerate(_FUNCTION_WORDS_150):
        vec[i] = word_lower.count(fw) / total
    return vec


def _sentence_length(text: str) -> np.ndarray:
    """5-dim: mean, median, std, min, max sentence lengths in tokens."""
    sents = _split_sentences(text)
    lens = [len(s.split()) for s in sents if s.strip()]
    if not lens:
        return np.zeros(5, dtype=np.float32)
    return np.array([
        np.mean(lens), np.median(lens), np.std(lens),
        np.min(lens), np.max(lens),
    ], dtype=np.float32)


def _punctuation_density(text: str) -> np.ndarray:
    """9-dim: normalised counts of punctuation marks."""
    chars = list(text)
    n = max(len(chars), 1)
    marks = [",", ";", ":", ".", "?", "!", "—", "(", ")"]
    return np.array([text.count(m) / n for m in marks], dtype=np.float32)


def _capitalization(tokens: List[str]) -> np.ndarray:
    """2-dim: ratio of capitalised words, ratio of all-caps tokens."""
    if not tokens:
        return np.zeros(2, dtype=np.float32)
    n = len(tokens)
    cap = sum(1 for t in tokens if t and t[0].isupper()) / n
    all_caps = sum(1 for t in tokens if t.isupper() and len(t) > 1) / n
    return np.array([cap, all_caps], dtype=np.float32)


def _type_token_ratio(tokens: List[str]) -> np.ndarray:
    """3-dim: TTR at window sizes 50, 100, full."""
    def ttr(toks):
        if not toks:
            return 0.0
        return len(set(toks)) / len(toks)
    w50 = ttr(tokens[:50])
    w100 = ttr(tokens[:100])
    full = ttr(tokens)
    return np.array([w50, w100, full], dtype=np.float32)


def _word_length(tokens: List[str]) -> np.ndarray:
    """2-dim: mean and std word length in characters."""
    word_lens = [len(t) for t in tokens if t.strip()]
    if not word_lens:
        return np.zeros(2, dtype=np.float32)
    return np.array([np.mean(word_lens), np.std(word_lens)], dtype=np.float32)


def _hedging(tokens: List[str]) -> np.ndarray:
    """1-dim: hedge count / token count."""
    word_lower = [t.lower() for t in tokens]
    total = max(len(tokens), 1)
    count = sum(word_lower.count(h) for h in _HEDGE_WORDS)
    return np.array([count / total], dtype=np.float32)


def _citations(text: str) -> np.ndarray:
    """1-dim: citation marker count / total chars."""
    patterns = [
        r"\([A-Z][a-z]+(?:\s*(?:&|and)\s*[A-Z][a-z]+)*,\s*\d{4}\)",
        r"\[\d+\]",
        r"et al\.",
    ]
    total = max(len(text), 1)
    count = sum(len(re.findall(p, text)) for p in patterns)
    return np.array([count / total], dtype=np.float32)


def _pos_bigrams(text: str, use_spacy: bool = True) -> Optional[np.ndarray]:
    """50-dim: top-50 POS bigram frequencies (spaCy or NLTK)."""
    try:
        if use_spacy:
            try:
                nlp = _get_spacy()
            except (OSError, Exception):
                raise ImportError("spacy model not found")
            doc = nlp(text[:2000])
            pos_tags = [token.pos_ for token in doc]
        else:
            import nltk
            from nltk import pos_tag, word_tokenize
            tokens = word_tokenize(text[:2000])
            pos_tags = [tag for _, tag in pos_tag(tokens)]

        bigram_counts: Dict[str, int] = {}
        for i in range(len(pos_tags) - 1):
            key = f"{pos_tags[i]}_{pos_tags[i+1]}"
            bigram_counts[key] = bigram_counts.get(key, 0) + 1

        total = max(sum(bigram_counts.values()), 1)
        vec = np.zeros(len(_POS_BIGRAM_VOCAB), dtype=np.float32)
        for i, bigram in enumerate(_POS_BIGRAM_VOCAB):
            vec[i] = bigram_counts.get(bigram, 0) / total
        return vec
    except ImportError:
        return None


def _discourse_markers(text: str, n_sentences: int) -> np.ndarray:
    """1-dim: discourse marker count / sentence count."""
    text_lower = text.lower()
    count = sum(text_lower.count(m) for m in _DISCOURSE_MARKERS)
    n = max(n_sentences, 1)
    return np.array([count / n], dtype=np.float32)


def _dependency_depth(text: str) -> Optional[np.ndarray]:
    """1-dim: mean dependency tree depth (spaCy)."""
    try:
        nlp = _get_spacy()

        def _depth(token):
            d = 0
            t = token
            while t.head != t:
                t = t.head
                d += 1
                if d > 100:
                    break
            return d

        doc = nlp(text[:2000])
        depths = [_depth(token) for token in doc]
        if not depths:
            return np.array([0.0], dtype=np.float32)
        return np.array([np.mean(depths)], dtype=np.float32)
    except (ImportError, OSError, Exception):
        return None


# ---------------------------------------------------------------------------
# Concatenated feature vector
# ---------------------------------------------------------------------------

def extract_lisa_features(text: str) -> Tuple[np.ndarray, List[str]]:
    """Extract LISA feature vector and corresponding feature names.

    Returns (vector, names) where names[i] labels vector[i].
    """
    tokens = text.split()
    sents = _split_sentences(text)
    n_sents = len(sents)

    parts: List[np.ndarray] = []
    names: List[str] = []

    # 1. Function words (150)
    fw = _function_words(tokens)
    parts.append(fw)
    names.extend([f"fw_{w}" for w in _FUNCTION_WORDS_150])

    # 2. Sentence length (5)
    sl = _sentence_length(text)
    parts.append(sl)
    names.extend(["sl_mean", "sl_median", "sl_std", "sl_min", "sl_max"])

    # 3. Punctuation density (9)
    pd = _punctuation_density(text)
    parts.append(pd)
    names.extend(["punct_comma", "punct_semi", "punct_colon", "punct_period",
                  "punct_question", "punct_exclaim", "punct_emdash",
                  "punct_open_paren", "punct_close_paren"])

    # 4. Capitalisation (2)
    cap = _capitalization(tokens)
    parts.append(cap)
    names.extend(["cap_first", "cap_all"])

    # 5. Type-token ratio (3)
    ttr = _type_token_ratio(tokens)
    parts.append(ttr)
    names.extend(["ttr_50", "ttr_100", "ttr_full"])

    # 6. Word length (2)
    wl = _word_length(tokens)
    parts.append(wl)
    names.extend(["wl_mean", "wl_std"])

    # 7. Hedging (1)
    hd = _hedging(tokens)
    parts.append(hd)
    names.append("hedging")

    # 8. Citations (1)
    ct = _citations(text)
    parts.append(ct)
    names.append("citations")

    # 9. POS bigrams (50 or skip)
    pb = _pos_bigrams(text)
    if pb is not None:
        parts.append(pb)
        names.extend([f"posbg_{bg}" for bg in _POS_BIGRAM_VOCAB])

    # 10. Discourse markers (1)
    dm = _discourse_markers(text, n_sents)
    parts.append(dm)
    names.append("discourse")

    # 11. Dependency depth (1 or skip)
    dd = _dependency_depth(text)
    if dd is not None:
        parts.append(dd)
        names.append("dep_depth")

    vec = np.concatenate(parts, axis=0)
    return vec, names


def get_feature_names() -> List[str]:
    """Return feature names from a dummy extraction."""
    _, names = extract_lisa_features("This is a test sentence.")
    return names


# ---------------------------------------------------------------------------
# Batch extraction and corpus building
# ---------------------------------------------------------------------------

def extract_corpus_features(
    texts: List[str],
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Extract LISA features for a list of texts.

    Returns (array of shape (N, D), feature_names).
    """
    from tqdm import tqdm

    feature_names: Optional[List[str]] = None
    rows: List[np.ndarray] = []

    it = tqdm(texts, desc="LISA features") if show_progress else texts
    for text in it:
        vec, names = extract_lisa_features(text)
        if feature_names is None:
            feature_names = names
        rows.append(vec)

    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []

    arr = np.stack(rows, axis=0)
    return arr, feature_names or []


# ---------------------------------------------------------------------------
# Phase 1a entry point
# ---------------------------------------------------------------------------

def build_lisa_corpus(
    cfg: "MechanisticConfig",  # noqa: F821
    probe_set: List[Dict],
    resume: bool = False,
) -> None:
    """Build probe-train and probe-eval LISA corpora and save as parquet."""
    import pandas as pd
    from deep_stylometry.experiments.mechanistic.io_utils import (
        output_root,
        output_exists,
        phase1_lisa_path,
    )

    root = output_root(cfg)
    features_path = phase1_lisa_path(root, "features.parquet")
    probe_corpus_path = phase1_lisa_path(root, "probe_corpus.parquet")

    if resume and output_exists(features_path, probe_corpus_path):
        logger.info("Phase 1a: LISA features already cached.")
        return

    import datasets as hf_datasets
    import numpy as np

    rng = np.random.default_rng(cfg.io.seed)

    # Collect probe-set passage ids to exclude
    probe_doc_ids = set()
    for entry in probe_set:
        probe_doc_ids.update([
            entry.get("anchor_doc_id", ""),
            entry.get("positive_doc_id", ""),
            entry.get("negative_doc_id", ""),
        ])

    logger.info("Loading base-4 train split for probe-train corpus...")
    train_ds = hf_datasets.load_dataset(
        "almanach/halvest-contrastive",
        name=cfg.base_data_subset,
        split="train",
    )

    logger.info("Loading base-4 valid split for probe-eval corpus...")
    valid_ds = hf_datasets.load_dataset(
        "almanach/halvest-contrastive",
        name=cfg.base_data_subset,
        split="valid",
    )

    def _collect_texts(ds, n_target):
        texts, doc_ids, author_ids, domains = [], [], [], []
        for i, row in enumerate(ds):
            if len(texts) >= n_target:
                break
            doc_id = row.get("query_id", str(i))
            if doc_id in probe_doc_ids:
                continue
            text = row.get("query", "")
            if not text:
                continue
            texts.append(text)
            doc_ids.append(doc_id)
            author_ids.append(row.get("query_authorids", []))
            domains.append(row.get("query_domain", "unknown"))
        return texts, doc_ids, author_ids, domains

    train_texts, train_doc_ids, train_author_ids, train_domains = _collect_texts(
        train_ds, cfg.lisa.probe_train_size
    )
    eval_texts, eval_doc_ids, eval_author_ids, eval_domains = _collect_texts(
        valid_ds, cfg.lisa.probe_eval_size
    )

    logger.info(
        "Extracting LISA features: train=%d, eval=%d",
        len(train_texts), len(eval_texts),
    )

    train_feats, feat_names = extract_corpus_features(train_texts)
    eval_feats, _ = extract_corpus_features(eval_texts)

    def _to_df(texts, doc_ids, author_ids, domains, feats, split_name):
        df = pd.DataFrame(feats, columns=feat_names)
        df.insert(0, "doc_id", doc_ids)
        df.insert(1, "domain", domains)
        df.insert(2, "split", split_name)
        df.insert(3, "text_preview", [t[:200] for t in texts])
        return df

    train_df = _to_df(train_texts, train_doc_ids, train_author_ids, train_domains,
                      train_feats, "train")
    eval_df = _to_df(eval_texts, eval_doc_ids, eval_author_ids, eval_domains,
                     eval_feats, "eval")
    combined_df = pd.concat([train_df, eval_df], ignore_index=True)

    combined_df.to_parquet(features_path, index=False)
    logger.info("Saved LISA features to %s", features_path)

    # Probe corpus: just eval split with feature vectors
    eval_df.to_parquet(probe_corpus_path, index=False)
    logger.info("Saved probe corpus to %s", probe_corpus_path)
