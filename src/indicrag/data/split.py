"""Data Ingestion and Leakage-Safe Cluster Partitioning Module.

Partitions IndicSQuAD Hindi data into Train, Dev, and Test splits.
Enforces zero article, title, context, and question leakage across partitions
by grouping content-linked articles into connected-component clusters.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PartitionMetadata:
    """Audit metadata for the partitioned dataset."""

    seed: int
    total_raw_records: int
    total_clusters: int
    train_clusters: int
    dev_clusters: int
    test_clusters: int
    train_records: int
    dev_records: int
    test_records: int
    benchmark_dev_sampled: int
    benchmark_test_sampled: int
    zero_title_leakage: bool
    zero_context_leakage: bool
    zero_question_leakage: bool
    zero_id_leakage: bool


def build_article_clusters(records: list[dict[str, Any]]) -> list[list[str]]:
    """Build connected components of articles sharing any context, question, or normalized title."""
    adj = defaultdict(set)
    titles = set()
    context_to_titles = defaultdict(set)
    question_to_titles = defaultdict(set)

    for r in records:
        t = r.get("title", "").strip()
        if not t:
            continue
        titles.add(t)
        ctx = r.get("context", "").strip()
        q = r.get("question", "").strip()
        if ctx:
            context_to_titles[ctx].add(t)
        if q:
            question_to_titles[q].add(t)

    # Link articles that share a context paragraph
    for _, linked_titles in context_to_titles.items():
        t_list = sorted(linked_titles)
        for i in range(len(t_list)):
            for j in range(i + 1, len(t_list)):
                adj[t_list[i]].add(t_list[j])
                adj[t_list[j]].add(t_list[i])

    # Link articles that share a question
    for _, linked_titles in question_to_titles.items():
        t_list = sorted(linked_titles)
        for i in range(len(t_list)):
            for j in range(i + 1, len(t_list)):
                adj[t_list[i]].add(t_list[j])
                adj[t_list[j]].add(t_list[i])

    visited = set()
    clusters = []
    for t in sorted(titles):
        if t not in visited:
            cluster = set()
            queue = [t]
            visited.add(t)
            while queue:
                curr = queue.pop(0)
                cluster.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(sorted(cluster))

    return clusters


def partition_dataset(
    raw_data_path: str,
    output_dir: str = "data/processed",
    seed: int = 42,
    dev_cluster_count: int = 4,
    test_cluster_count: int = 4,
    target_dev_samples: int = 150,
    target_test_samples: int = 75,
) -> PartitionMetadata:
    """Deterministically partition IndicSQuAD Hindi data into Train, Dev, and Test sets."""
    with open(raw_data_path, encoding="utf-8") as f:
        records = json.load(f)

    # 1. Normalize records and add answerability label
    normalized_records = []
    for r in records:
        ans_text = r.get("answers", {}).get("text", [])
        is_impossible = len(ans_text) == 0
        norm_r = {
            "id": r["id"],
            "title": r.get("title", "").strip(),
            "context": r.get("context", "").strip(),
            "question": r.get("question", "").strip(),
            "answers": r.get("answers", {"text": [], "answer_start": []}),
            "is_impossible": is_impossible,
            "script": "hindi",
            "synthetic_hinglish_id": f"{r['id']}_hinglish",
        }
        normalized_records.append(norm_r)

    # 2. Build leakage-safe connected clusters
    clusters = build_article_clusters(normalized_records)

    # 3. Deterministically shuffle and allocate clusters
    rng = random.Random(seed)
    shuffled_clusters = list(clusters)
    rng.shuffle(shuffled_clusters)

    test_clusters = shuffled_clusters[:test_cluster_count]
    dev_clusters = shuffled_clusters[test_cluster_count : test_cluster_count + dev_cluster_count]
    train_clusters = shuffled_clusters[test_cluster_count + dev_cluster_count :]

    test_titles = {t for c in test_clusters for t in c}
    dev_titles = {t for c in dev_clusters for t in c}
    train_titles = {t for c in train_clusters for t in c}

    # Verify disjoint titles
    assert not (test_titles & dev_titles), "Leakage: test and dev titles overlap!"
    assert not (test_titles & train_titles), "Leakage: test and train titles overlap!"
    assert not (dev_titles & train_titles), "Leakage: dev and train titles overlap!"

    # 4. Partition records
    train_records = [r for r in normalized_records if r["title"] in train_titles]
    dev_records = [r for r in normalized_records if r["title"] in dev_titles]
    test_records = [r for r in normalized_records if r["title"] in test_titles]

    # Assign cluster_id to each record
    title_to_cluster_id = {}
    for idx, c in enumerate(clusters):
        for t in c:
            title_to_cluster_id[t] = f"cluster_{idx:03d}"

    for r in train_records:
        r["cluster_id"] = title_to_cluster_id[r["title"]]
        r["partition"] = "train"
    for r in dev_records:
        r["cluster_id"] = title_to_cluster_id[r["title"]]
        r["partition"] = "dev"
    for r in test_records:
        r["cluster_id"] = title_to_cluster_id[r["title"]]
        r["partition"] = "test"

    # 5. Extract balanced evaluation benchmark subsets (Proposal A)
    # Dev: target 150 (75 ans, 75 unans)
    dev_ans = [r for r in dev_records if not r["is_impossible"]]
    dev_unans = [r for r in dev_records if r["is_impossible"]]
    rng.shuffle(dev_ans)
    rng.shuffle(dev_unans)
    dev_sample_size_half = target_dev_samples // 2
    benchmark_dev = (
        dev_ans[:dev_sample_size_half]
        + dev_unans[: (target_dev_samples - dev_sample_size_half)]
    )
    rng.shuffle(benchmark_dev)

    # Test: target 75 (38 ans, 37 unans)
    test_ans = [r for r in test_records if not r["is_impossible"]]
    test_unans = [r for r in test_records if r["is_impossible"]]
    rng.shuffle(test_ans)
    rng.shuffle(test_unans)
    test_ans_target = (target_test_samples + 1) // 2
    test_unans_target = target_test_samples - test_ans_target
    benchmark_test = (
        test_ans[:test_ans_target]
        + test_unans[:test_unans_target]
    )
    rng.shuffle(benchmark_test)

    # 6. Verify full zero-leakage assertions
    train_contexts = {r["context"] for r in train_records}
    dev_contexts = {r["context"] for r in dev_records}
    test_contexts = {r["context"] for r in test_records}

    train_questions = {r["question"] for r in train_records}
    dev_questions = {r["question"] for r in dev_records}
    test_questions = {r["question"] for r in test_records}

    train_ids = {r["id"] for r in train_records}
    dev_ids = {r["id"] for r in dev_records}
    test_ids = {r["id"] for r in test_records}

    zero_titles = bool(
        not (test_titles & dev_titles)
        and not (test_titles & train_titles)
        and not (dev_titles & train_titles)
    )
    zero_contexts = bool(
        not (test_contexts & dev_contexts)
        and not (test_contexts & train_contexts)
        and not (dev_contexts & train_contexts)
    )
    zero_questions = bool(
        not (test_questions & dev_questions)
        and not (test_questions & train_questions)
        and not (dev_questions & train_questions)
    )
    zero_ids = bool(
        not (test_ids & dev_ids)
        and not (test_ids & train_ids)
        and not (dev_ids & train_ids)
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "hindi_train.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    with open(out_path / "hindi_dev.json", "w", encoding="utf-8") as f:
        json.dump(dev_records, f, ensure_ascii=False, indent=2)

    with open(out_path / "hindi_test.json", "w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)

    with open(out_path / "benchmark_dev_150.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_dev, f, ensure_ascii=False, indent=2)

    with open(out_path / "benchmark_test_75.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_test, f, ensure_ascii=False, indent=2)

    # 7. Build Search Corpora (unannotated passages only, zero labels/questions)
    def extract_passages(recs: list[dict[str, Any]], part_label: str) -> list[dict[str, Any]]:
        seen_texts = set()
        passages = []
        import hashlib
        for r in recs:
            ctx = r["context"].strip()
            if ctx and ctx not in seen_texts:
                seen_texts.add(ctx)
                pid = f"psg_{hashlib.sha256(ctx.encode('utf-8')).hexdigest()[:12]}"
                passages.append({
                    "passage_id": pid,
                    "title": r["title"],
                    "text": ctx,
                    "source_partition": part_label,
                })
        return passages

    train_passages = extract_passages(train_records, "train")
    dev_passages = extract_passages(dev_records, "dev")
    test_passages = extract_passages(test_records, "test")

    corpus_background = list(train_passages)
    corpus_dev = train_passages + dev_passages
    corpus_test = train_passages + test_passages

    with open(out_path / "corpus_background.json", "w", encoding="utf-8") as f:
        json.dump(corpus_background, f, ensure_ascii=False, indent=2)

    with open(out_path / "corpus_dev.json", "w", encoding="utf-8") as f:
        json.dump(corpus_dev, f, ensure_ascii=False, indent=2)

    with open(out_path / "corpus_test.json", "w", encoding="utf-8") as f:
        json.dump(corpus_test, f, ensure_ascii=False, indent=2)

    meta = PartitionMetadata(
        seed=seed,
        total_raw_records=len(records),
        total_clusters=len(clusters),
        train_clusters=len(train_clusters),
        dev_clusters=len(dev_clusters),
        test_clusters=len(test_clusters),
        train_records=len(train_records),
        dev_records=len(dev_records),
        test_records=len(test_records),
        benchmark_dev_sampled=len(benchmark_dev),
        benchmark_test_sampled=len(benchmark_test),
        zero_title_leakage=zero_titles,
        zero_context_leakage=zero_contexts,
        zero_question_leakage=zero_questions,
        zero_id_leakage=zero_ids,
    )

    with open(out_path / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    return meta


if __name__ == "__main__":
    raw_val = "data/raw/Hindi_val.json"
    if os.path.exists(raw_val):
        print(f"Running partitioning on {raw_val}...")
        meta = partition_dataset(raw_val)
        print("Partitioning complete:")
        for k, v in asdict(meta).items():
            print(f"  {k}: {v}")
    else:
        print(f"File not found: {raw_val}")
