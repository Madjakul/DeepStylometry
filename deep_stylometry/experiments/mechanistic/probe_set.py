# deep_stylometry/experiments/mechanistic/probe_set.py
"""Phase 0: build the curated triplet probe set."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deep_stylometry.experiments.mechanistic.config import MechanisticConfig
from deep_stylometry.experiments.mechanistic.io_utils import (
    output_exists,
    phase0_path,
    output_root,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenisation helper (uses same settings as halvest_datamodule)
# ---------------------------------------------------------------------------

def _tokenize_text(text: str, tokenizer) -> List[int]:
    return tokenizer(
        text,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=False,
    )["input_ids"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_hub_split(subset: str, split: str) -> Any:
    """Load a HALvest-Contrastive split from the Hub."""
    import datasets as hf_datasets
    logger.info("Loading almanach/halvest-contrastive %s %s from Hub...", subset, split)
    return hf_datasets.load_dataset(
        "almanach/halvest-contrastive", name=subset, split=split
    )


# ---------------------------------------------------------------------------
# Author-set inventory builder
# ---------------------------------------------------------------------------

def _scalar_domain(raw) -> str:
    """Normalise a domain value that may be a list to a plain string."""
    if isinstance(raw, list):
        return raw[0] if raw else "unknown"
    return raw or "unknown"


def _build_inventory(ds: Any) -> Dict[str, Any]:
    """Extract per-document inventory from a HALvest split.

    Returns a dict keyed by frozenset of author ids.  Each value is a list of
    dicts with keys: text, domain, doc_id, authorids.
    """
    inventory: Dict[frozenset, List[Dict]] = defaultdict(list)

    for i, row in enumerate(ds):
        # Query (anchor) side
        q_authors = frozenset(row.get("query_authorids") or [])
        q_domain = _scalar_domain(row.get("query_domain", "unknown"))
        q_text = row.get("query", "")
        q_doc_id = row.get("query_id", str(i))

        if q_authors and q_text:
            inventory[q_authors].append({
                "text": q_text,
                "domain": q_domain,
                "doc_id": q_doc_id,
                "authorids": list(q_authors),
                "row_index": i,
            })

        # Positive side — different document, same author set
        p_authors = frozenset(row.get("pos_authorids") or [])
        p_domain = _scalar_domain(row.get("pos_domain", q_domain))
        p_text = row.get("positive", "")
        p_doc_id = row.get("pos_id", f"{i}_pos")

        if p_authors and p_text and p_authors == q_authors:
            inventory[p_authors].append({
                "text": p_text,
                "domain": p_domain,
                "doc_id": p_doc_id,
                "authorids": list(p_authors),
                "row_index": i,
            })

    return dict(inventory)


def _dedup_by_doc_id(docs: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for d in docs:
        if d["doc_id"] not in seen:
            seen.add(d["doc_id"])
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Rotation pool selection
# ---------------------------------------------------------------------------

def _select_rotation_pool(
    inventory: Dict[frozenset, List[Dict]],
    pool_size: int,
    min_docs: int,
    rng: random.Random,
) -> List[frozenset]:
    candidates = [
        (aset, docs)
        for aset, docs in inventory.items()
        if len(_dedup_by_doc_id(docs)) >= min_docs
    ]
    if not candidates:
        return []
    # Sort by frequency descending, then pick top pool_size
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    pool = [aset for aset, _ in candidates[:pool_size]]
    return pool


# ---------------------------------------------------------------------------
# Tier builders
# ---------------------------------------------------------------------------

def _make_tier_a_entries(
    rotation_pool: List[frozenset],
    inventory: Dict[frozenset, List[Dict]],
    n_per_tier: int,
    tokenizer,
    target_length: int,
    pair_tol: int,
    global_tol: int,
    rng: random.Random,
) -> List[Dict]:
    """Tier A: anchor/positive same set, negative disjoint same domain."""
    entries = []
    all_sets = list(inventory.keys())

    for anchor_set in rotation_pool:
        docs = _dedup_by_doc_id(inventory[anchor_set])
        if len(docs) < 2:
            continue

        for i, anchor_doc in enumerate(docs):
            if len(entries) >= n_per_tier:
                break
            # Positive: same author set, different document
            pos_candidates = [d for d in docs if d["doc_id"] != anchor_doc["doc_id"]]
            if not pos_candidates:
                continue
            pos_doc = rng.choice(pos_candidates)

            # Negative: disjoint author set, same domain
            anchor_domain = anchor_doc["domain"]
            neg_pool = [
                d
                for aset, docs_n in inventory.items()
                if aset.isdisjoint(anchor_set)
                for d in docs_n
                if d["domain"] == anchor_domain
            ]
            if not neg_pool:
                # Relax domain constraint
                neg_pool = [
                    d
                    for aset, docs_n in inventory.items()
                    if aset.isdisjoint(anchor_set)
                    for d in docs_n
                ]
            if not neg_pool:
                continue
            neg_doc = rng.choice(neg_pool)

            entry = _build_entry(
                "A", len(entries) + 1,
                anchor_doc, pos_doc, neg_doc,
                tokenizer, target_length, pair_tol, global_tol,
            )
            if entry is not None:
                entries.append(entry)

        if len(entries) >= n_per_tier:
            break

    return entries[:n_per_tier]


def _make_tier_b_entries(
    rotation_pool: List[frozenset],
    inventory: Dict[frozenset, List[Dict]],
    n_per_tier: int,
    tokenizer,
    target_length: int,
    pair_tol: int,
    global_tol: int,
    rng: random.Random,
) -> List[Dict]:
    """Tier B: negative shares exactly one author with anchor."""
    entries = []

    for anchor_set in rotation_pool:
        docs = _dedup_by_doc_id(inventory[anchor_set])
        if len(docs) < 2:
            continue

        # Find sets that intersect anchor_set by exactly one member
        partial_sets = [
            aset
            for aset in inventory.keys()
            if len(aset & anchor_set) == 1 and aset != anchor_set
        ]
        if not partial_sets:
            continue

        for i, anchor_doc in enumerate(docs):
            if len(entries) >= n_per_tier:
                break
            pos_candidates = [d for d in docs if d["doc_id"] != anchor_doc["doc_id"]]
            if not pos_candidates:
                continue
            pos_doc = rng.choice(pos_candidates)

            pset = rng.choice(partial_sets)
            neg_candidates = inventory[pset]
            if not neg_candidates:
                continue
            neg_doc = rng.choice(neg_candidates)

            entry = _build_entry(
                "B", len(entries) + 1,
                anchor_doc, pos_doc, neg_doc,
                tokenizer, target_length, pair_tol, global_tol,
            )
            if entry is not None:
                entries.append(entry)

        if len(entries) >= n_per_tier:
            break

    return entries[:n_per_tier]


def _make_tier_c_entries(
    rotation_pool: List[frozenset],
    inventory: Dict[frozenset, List[Dict]],
    n_per_tier: int,
    tokenizer,
    target_length: int,
    pair_tol: int,
    global_tol: int,
    rng: random.Random,
) -> List[Dict]:
    """Tier C: anchor in domain D1, positive by same author in domain D2."""
    entries = []

    for anchor_set in rotation_pool:
        docs = _dedup_by_doc_id(inventory[anchor_set])
        if len(docs) < 2:
            continue

        # Organise docs by domain
        by_domain: Dict[str, List[Dict]] = defaultdict(list)
        for d in docs:
            by_domain[d["domain"]].append(d)

        # Need at least 2 domains
        domains = [dom for dom, ds in by_domain.items() if len(ds) >= 1]
        if len(domains) < 2:
            continue

        for anchor_doc in docs:
            if len(entries) >= n_per_tier:
                break
            d1 = anchor_doc["domain"]
            other_domains = [dom for dom in domains if dom != d1]
            if not other_domains:
                continue
            d2 = rng.choice(other_domains)
            pos_candidates = by_domain[d2]
            pos_doc = rng.choice(pos_candidates)

            # Negative: disjoint author set in D1
            neg_pool = [
                d
                for aset, docs_n in inventory.items()
                if aset.isdisjoint(anchor_set)
                for d in docs_n
                if d["domain"] == d1
            ]
            if not neg_pool:
                neg_pool = [
                    d
                    for aset, docs_n in inventory.items()
                    if aset.isdisjoint(anchor_set)
                    for d in docs_n
                ]
            if not neg_pool:
                continue
            neg_doc = rng.choice(neg_pool)

            entry = _build_entry(
                "C", len(entries) + 1,
                anchor_doc, pos_doc, neg_doc,
                tokenizer, target_length, pair_tol, global_tol,
            )
            if entry is not None:
                entries.append(entry)

        if len(entries) >= n_per_tier:
            break

    return entries[:n_per_tier]


# ---------------------------------------------------------------------------
# Length filtering
# ---------------------------------------------------------------------------

def _build_entry(
    tier: str,
    idx: int,
    anchor_doc: Dict,
    pos_doc: Dict,
    neg_doc: Dict,
    tokenizer,
    target_length: int,
    pair_tol: int,
    global_tol: int,
) -> Optional[Dict]:
    a_ids = _tokenize_text(anchor_doc["text"], tokenizer)
    p_ids = _tokenize_text(pos_doc["text"], tokenizer)
    n_ids = _tokenize_text(neg_doc["text"], tokenizer)

    len_a, len_p, len_n = len(a_ids), len(p_ids), len(n_ids)

    # Pair tolerance: |pos - neg| <= pair_tol
    if abs(len_p - len_n) > pair_tol:
        return None

    # Global tolerance: |pos - target| and |neg - target| within global_tol
    if abs(len_p - target_length) > global_tol:
        return None
    if abs(len_n - target_length) > global_tol:
        return None

    return {
        "tier": tier,
        "triplet_id": f"tier{tier}_{idx:04d}",
        "anchor_text": anchor_doc["text"],
        "anchor_authors": anchor_doc["authorids"],
        "anchor_domain": anchor_doc["domain"],
        "anchor_doc_id": anchor_doc["doc_id"],
        "anchor_token_len": len_a,
        "positive_text": pos_doc["text"],
        "positive_authors": pos_doc["authorids"],
        "positive_domain": pos_doc["domain"],
        "positive_doc_id": pos_doc["doc_id"],
        "positive_token_len": len_p,
        "negative_text": neg_doc["text"],
        "negative_authors": neg_doc["authorids"],
        "negative_domain": neg_doc["domain"],
        "negative_doc_id": neg_doc["doc_id"],
        "negative_token_len": len_n,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_statistics(entries: List[Dict]) -> Dict:
    tiers = ["A", "B", "C"]
    stats: Dict[str, Any] = {"counts": {}}
    for t in tiers:
        t_entries = [e for e in entries if e["tier"] == t]
        stats["counts"][f"tier_{t}"] = len(t_entries)
        if t_entries:
            pos_lens = [e["positive_token_len"] for e in t_entries]
            neg_lens = [e["negative_token_len"] for e in t_entries]
            all_lens = pos_lens + neg_lens
            stats[f"tier_{t}_length_mean"] = float(np.mean(all_lens))
            stats[f"tier_{t}_length_std"] = float(np.std(all_lens))

    # Tier B overlap breakdown
    b_entries = [e for e in entries if e["tier"] == "B"]
    if b_entries:
        overlap_sizes = Counter()
        for e in b_entries:
            n = len(set(e["anchor_authors"]) & set(e["negative_authors"]))
            overlap_sizes[n] += 1
        stats["tier_b_overlap_distribution"] = dict(overlap_sizes)

    stats["total"] = len(entries)
    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_probe_set(cfg: MechanisticConfig, resume: bool = False) -> List[Dict]:
    root = output_root(cfg)
    probe_set_file = phase0_path(root, "probe_set.json")
    stats_file = phase0_path(root, "statistics.json")

    if resume and output_exists(probe_set_file, stats_file):
        logger.info("Phase 0: loading cached probe set from %s", probe_set_file)
        with open(probe_set_file) as f:
            return json.load(f)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    rng = random.Random(cfg.io.seed)

    # Try validation split first
    ds = _load_hub_split(cfg.base_data_subset, "valid")
    inventory = _build_inventory(ds)

    rotation_pool = _select_rotation_pool(
        inventory,
        cfg.probe_set.rotation_pool_size,
        cfg.probe_set.min_documents_per_set,
        rng,
    )
    augmented_from_train = False

    if len(rotation_pool) < cfg.probe_set.rotation_pool_size:
        logger.warning(
            "Validation has only %d qualifying author sets; "
            "augmenting with training split.",
            len(rotation_pool),
        )
        train_ds = _load_hub_split(cfg.base_data_subset, "train")
        train_inventory = _build_inventory(train_ds)
        for aset, docs in train_inventory.items():
            if aset not in inventory:
                inventory[aset] = docs
            else:
                inventory[aset].extend(docs)
        rotation_pool = _select_rotation_pool(
            inventory,
            cfg.probe_set.rotation_pool_size,
            cfg.probe_set.min_documents_per_set,
            rng,
        )
        augmented_from_train = True

    logger.info("Rotation pool size: %d", len(rotation_pool))

    ps = cfg.probe_set
    shared_kwargs = dict(
        inventory=inventory,
        n_per_tier=ps.n_per_tier,
        tokenizer=tokenizer,
        target_length=ps.target_length,
        pair_tol=ps.pair_tolerance,
        global_tol=ps.global_tolerance,
        rng=rng,
    )

    tier_a: List[Dict] = []
    tier_b: List[Dict] = []
    tier_c: List[Dict] = []

    for tier_name, make_fn in [
        ("A", _make_tier_a_entries),
        ("B", _make_tier_b_entries),
        ("C", _make_tier_c_entries),
    ]:
        base_gtol = ps.global_tolerance
        base_ptol = ps.pair_tolerance
        max_gtol = base_gtol * 4
        tier_entries: List[Dict] = []

        for step_i, gtol in enumerate(range(base_gtol, max_gtol + 1, 5)):
            ptol = int(base_ptol * (1.5 ** step_i))
            if step_i > 0:
                logger.info(
                    "Tier %s: loosening tolerances to global_tol=%d pair_tol=%d "
                    "(have %d/%d).",
                    tier_name, gtol, ptol, len(tier_entries), ps.n_per_tier,
                )
            tier_entries = make_fn(
                rotation_pool,
                **{**shared_kwargs, "global_tol": gtol, "pair_tol": ptol},
            )
            if len(tier_entries) >= ps.n_per_tier:
                break

        if len(tier_entries) < 10:
            logger.warning(
                "Tier %s: only %d entries after full tolerance escalation.",
                tier_name, len(tier_entries),
            )

        if tier_name == "A":
            tier_a = tier_entries
        elif tier_name == "B":
            tier_b = tier_entries
        else:
            tier_c = tier_entries

    all_entries = tier_a + tier_b + tier_c
    stats = _compute_statistics(all_entries)
    stats["rotation_pool_size"] = len(rotation_pool)
    stats["augmented_from_train"] = augmented_from_train

    logger.info(
        "Probe set: A=%d, B=%d, C=%d  (total=%d)",
        len(tier_a), len(tier_b), len(tier_c), len(all_entries),
    )

    with open(probe_set_file, "w") as f:
        json.dump(all_entries, f, indent=2)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Saved probe set to %s", probe_set_file)
    return all_entries
