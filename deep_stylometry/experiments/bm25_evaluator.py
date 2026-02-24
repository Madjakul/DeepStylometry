# deep_stylometry/evaluation/bm25_evaluator.py
"""BM25 evaluation using rank_bm25 + ranx.

Accepts either raw strings or pre-tokenized input_ids (from a HF
tokenizer). When using input_ids, tokens are decoded back to strings —
this gives you the subword "semi-lemmatization" from the tokenizer for
free.
"""

import logging
from typing import Dict, List, Optional, Union

from ranx import Qrels, Run, evaluate
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class BM25Evaluator:

    def __init__(
        self,
        k: int = 100,
        metrics: Optional[List[str]] = None,
    ) -> None:
        self.k = k
        self.metrics = metrics or [
            "mrr@10",
            "mrr@100",
            "ndcg@10",
            "ndcg@100",
            "recall@10",
            "recall@100",
        ]

    def evaluate(
        self,
        queries: Union[List[str], List[List[int]]],
        corpus: Union[List[str], List[List[int]]],
        hard_qrels: Dict[str, Dict[str, int]],
        soft_qrels: Optional[Dict[str, Dict[str, int]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        corpus_ids: Optional[List[str]] = None,
        query_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        queries / corpus
            Either raw strings or lists of token ids.
            If token ids, ``tokenizer`` must be provided to decode them.
        tokenizer
            HF tokenizer — used to convert input_ids → string tokens
            for BM25.  Each id is decoded individually so BM25 sees
            subword units (e.g. "run", "##ning") as separate terms.
        """
        corpus_ids = corpus_ids or [f"d_{i}" for i in range(len(corpus))]
        query_ids = query_ids or [f"q_{i}" for i in range(len(queries))]

        # Convert to List[List[str]] for BM25
        tokenized_corpus = self._to_str_tokens(corpus, tokenizer, "corpus")
        tokenized_queries = self._to_str_tokens(queries, tokenizer, "queries")

        bm25 = BM25Okapi(tokenized_corpus)

        run_dict: Dict[str, Dict[str, float]] = {}
        for qid, tok_query in tqdm(
            zip(query_ids, tokenized_queries), total=len(queries), desc="BM25/retrieve"
        ):
            scores = bm25.get_scores(tok_query)
            top_idx = scores.argsort()[::-1][: self.k]
            run_dict[qid] = {corpus_ids[i]: float(scores[i]) for i in top_idx}

        run = Run(run_dict)
        mrr_m = [m for m in self.metrics if m.startswith("mrr")]
        soft_m = [m for m in self.metrics if not m.startswith("mrr")]

        results = {}
        if mrr_m:
            results.update(evaluate(Qrels(hard_qrels), run, mrr_m))
        if soft_m:
            qrels_obj = Qrels(soft_qrels) if soft_qrels else Qrels(hard_qrels)
            results.update(evaluate(qrels_obj, run, soft_m))

        logger.info(f"BM25: {results}")
        return results

    @staticmethod
    def _to_str_tokens(
        data: Union[List[str], List[List[int]]],
        tokenizer: Optional[PreTrainedTokenizerBase],
        label: str,
    ) -> List[List[str]]:
        """Convert input to List[List[str]] for BM25."""
        # input_ids -> decode each token individually to a string
        assert (
            tokenizer is not None
        ), f"Got input_ids for {label} but no tokenizer provided"
        result = []
        special_ids = set(tokenizer.all_special_ids)
        for ids in tqdm(data, desc=f"BM25/decode {label}"):
            tokens = [
                tokenizer.decode([tid]).strip() for tid in ids if tid not in special_ids
            ]
            # Filter empty strings from decode
            result.append([t for t in tokens if t])
        return result
