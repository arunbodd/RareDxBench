#!/usr/bin/env python3
"""
Gene Prioritization Benchmark Runner for Claude Haiku 5.5
=========================================================
Runs Claude Haiku 5.5 on the RareDxBench variant ranking benchmark using the canonical
prompts in data/ranking/ranking_prompts.jsonl and ground truth answers in data/ranking/ranking_answers.jsonl.

Ablation / Information content:
  - Phenotypes: Positive & Negative HPO terms with descriptive labels
  - Suspected Disease ID
  - Candidate Genes: Explicit HGNC gene symbols + Ensembl IDs + observed genomic positions & annotations

Usage:
  python scripts/run_haiku_ranking.py --limit 200 --detach
"""

import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx not installed. Run: pip install httpx")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()
except ImportError:
    pass

# Ensure SSL_CERT_FILE points to a valid cert bundle (override stale global .env paths)
try:
    import certifi
    if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
        os.environ["SSL_CERT_FILE"] = certifi.where()
    SSL_VERIFY = certifi.where() if os.path.exists(certifi.where()) else True
except ImportError:
    SSL_VERIFY = True

# Strip trailing "(ENSGXXX)" parentheticals or annotations from predicted gene strings
_ID_IN_PARENS = re.compile(r'^([A-Za-z0-9\-_]+)(?:\s*\([^)]+\))?')
_CANDIDATE_LINE_RE = re.compile(r'^\s*([A-Za-z0-9\-_]+)\s*\((ENSG\d+|[A-Za-z0-9\-_]+)\)', re.M)

DEFAULT_MODEL = "anthropic/claude-haiku-5-5"

DEFAULT_PROMPTS = REPO_ROOT / "data" / "ranking" / "ranking_prompts.jsonl"
DEFAULT_ANSWERS = REPO_ROOT / "data" / "ranking" / "ranking_answers.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "ranking" / "Results_Claude_Haiku_5_5"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "haiku_200_predictions.jsonl"


def normalize_symbol(gene: str) -> str:
    """Extract and normalize HGNC symbol from string."""
    if not gene:
        return ""
    g = gene.strip()
    m = _ID_IN_PARENS.match(g)
    if m:
        g = m.group(1).strip()
    return g.upper()


def extract_gene_map_from_prompt(prompt: str) -> dict[str, str]:
    """Map Ensembl IDs and symbols observed in the Candidate Genes block to HGNC symbols."""
    mapping = {}
    for sym, ens in _CANDIDATE_LINE_RE.findall(prompt):
        s = sym.strip()
        e = ens.strip()
        if e.startswith("ENSG"):
            mapping[e] = s
        if s.startswith("ENSG"):
            mapping[s] = e
        mapping[s.upper()] = s
        mapping[e.upper()] = s
    return mapping


def load_dataset(prompts_path: Path, answers_path: Path, limit: int = 0) -> list[dict]:
    """Load canonical prompts and merge with answer truth labels."""
    answers = {}
    if answers_path.exists():
        with open(answers_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ans = json.loads(line)
                answers[int(ans["id"])] = ans

    dataset = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = int(row["id"])
            prompt = row["prompt"]
            ans = answers.get(pid, {})

            # Map from prompt
            gene_map = extract_gene_map_from_prompt(prompt)

            # Resolve true genes
            true_ensg_list = ans.get("true_genes", [])
            true_symbols = []
            for eg in true_ensg_list:
                sym = gene_map.get(eg, eg)
                true_symbols.append(sym)

            dataset.append({
                "patient_id": pid,
                "prompt": prompt,
                "disease_id": ans.get("disease_id", ""),
                "true_genes_ensg": true_ensg_list,
                "true_genes": true_symbols,
                "category": ans.get("category", "unknown"),
                "broad_category": ans.get("broad_category", "unknown"),
                "in_gene_gene_kg": ans.get("in_gene_gene_kg"),
                "in_disease_gene_kg": ans.get("in_disease_gene_kg"),
                "all_candidate_genes": ans.get("all_candidate_genes", []),
            })
            if limit and len(dataset) >= limit:
                break

    return dataset


async def call_aiva(client: httpx.AsyncClient, api_url: str, api_key: str, model: str, prompt: str, max_retries: int = 5) -> str | None:
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"message": prompt, "model": model, "stream": "false"}

    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST", api_url, data=data, headers=headers, timeout=180.0) as resp:
                if resp.status_code == 429:
                    delay = 10 * attempt
                    print(f"  [429 Rate Limit] Backing off {delay}s (attempt {attempt}/{max_retries})...", file=sys.stderr)
                    await asyncio.sleep(delay)
                    continue

                if 500 <= resp.status_code < 600:
                    delay = 5 * attempt
                    print(f"  [HTTP {resp.status_code}] Retrying in {delay}s (attempt {attempt}/{max_retries})...", file=sys.stderr)
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    print(f"  API error {resp.status_code}: {body[:300]}", file=sys.stderr)
                    return None

                text = ""
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        ev = json.loads(payload)
                        if ev.get("type") in ("text_delta", "final_output"):
                            text += ev.get("content", "")
                    except json.JSONDecodeError:
                        pass
                return text

        except Exception as exc:
            if attempt == max_retries:
                print(f"  Request failed after {max_retries} attempts: {exc}", file=sys.stderr)
                return None
            await asyncio.sleep(5 * attempt)

    return None


def parse_predictions(response_text: str) -> tuple[list[str], dict[str, float]]:
    """Parse JSON response containing predicted_genes list and confidence scores."""
    genes: list[str] = []
    scores: dict[str, float] = {}

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(response_text):
        if response_text[idx] != "{":
            idx += 1
            continue
        try:
            parsed, span = decoder.raw_decode(response_text[idx:])
        except json.JSONDecodeError:
            idx += 1
            continue

        if isinstance(parsed, dict) and "predicted_genes" in parsed:
            pg = parsed["predicted_genes"]
            if isinstance(pg, list):
                for entry in pg:
                    if isinstance(entry, dict):
                        gene = entry.get("gene") or entry.get("symbol") or entry.get("name")
                        scr = entry.get("score")
                        if gene:
                            sym = normalize_symbol(str(gene))
                            genes.append(sym)
                            if scr is not None:
                                try:
                                    scores[sym] = float(scr)
                                except (ValueError, TypeError):
                                    pass
                    elif isinstance(entry, str):
                        genes.append(normalize_symbol(entry))
            break
        idx += max(span, 1)

    # De-duplicate preserving ranking order
    seen = set()
    deduped = []
    for g in genes:
        if g and g not in seen:
            seen.add(g)
            deduped.append(g)

    return deduped, scores


def find_rank(predicted: list[str], true_genes: list[str]) -> int | None:
    """Return 1-based rank of the true gene in predicted list."""
    true_set = {normalize_symbol(g) for g in true_genes if g}
    for i, g in enumerate(predicted, 1):
        if normalize_symbol(g) in true_set:
            return i
    return None


def compute_benchmark_metrics(results: list[dict], at_ks: tuple[int, ...] = (1, 5, 10, 20, 50)) -> dict:
    """Compute standard gene ranking benchmark metrics."""
    ranks = []
    found_at = {k: 0 for k in at_ks}
    reciprocal_ranks = []
    dcg_list = []
    no_prediction = 0
    retention_count = 0

    per_category = defaultdict(list)

    for r in results:
        predicted = r.get("predicted", [])
        true_genes = r.get("true_genes", [])
        cat = r.get("category", "unknown")

        if not predicted:
            no_prediction += 1
            per_category[cat].append(None)
            reciprocal_ranks.append(0.0)
            dcg_list.append(0.0)
            continue

        rk = find_rank(predicted, true_genes)
        per_category[cat].append(rk)

        if rk is not None:
            retention_count += 1
            ranks.append(rk)
            reciprocal_ranks.append(1.0 / rk)
            dcg_list.append(1.0 / math.log2(rk + 1))
            for k in at_ks:
                if rk <= k:
                    found_at[k] += 1
        else:
            reciprocal_ranks.append(0.0)
            dcg_list.append(0.0)

    n = len(results)
    if n == 0:
        return {}

    idcg = 1.0
    metrics = {
        "n_patients": n,
        "no_prediction": no_prediction,
        "retention_rate": round(retention_count / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "ndcg@10": round(sum(dcg_list) / (n * idcg), 4),
        "mean_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "median_rank": float(sorted(ranks)[len(ranks) // 2]) if ranks else None,
    }
    for k in at_ks:
        metrics[f"recall@{k}"] = round(found_at[k] / n, 4)

    cat_metrics = {}
    for cat, cat_ranks in per_category.items():
        n_cat = len(cat_ranks)
        cm = {"n": n_cat}
        for k in at_ks:
            found_k = sum(1 for rk in cat_ranks if rk is not None and rk <= k)
            cm[f"recall@{k}"] = round(found_k / n_cat, 4)
        cat_metrics[cat] = cm
    metrics["per_category"] = cat_metrics

    return metrics


async def run_benchmark(dataset: list[dict], output_path: Path, model: str, concurrency: int = 3, dry_run: bool = False):
    api_url = os.getenv("AIVA_API_URL", "https://api.aivaportal.com/agents/chat/stream").rstrip("/")
    if "/agents/chat/stream" not in api_url:
        api_url = f"{api_url}/agents/chat/stream"
    api_key = os.getenv("AIVA_API_KEY") or os.getenv("AIVA_API_TOKEN", "")

    if not api_key:
        sys.exit("ERROR: AIVA_API_TOKEN / AIVA_API_KEY is not set.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done_ids = set()
    results = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(int(r["patient_id"]))
                    results.append(r)
                except Exception:
                    pass
        if done_ids:
            print(f"Resuming run: {len(done_ids)} patients already completed.")

    remaining = [p for p in dataset if p["patient_id"] not in done_ids]
    print(f"Patients to process: {len(remaining)} / {len(dataset)}")
    print(f"Model: {model}")
    print(f"Concurrency: {concurrency}")

    if not remaining:
        print("All requested patients are already completed.")
        return results

    sem = asyncio.Semaphore(concurrency)
    counter = {"done": 0}
    total = len(remaining)

    async with httpx.AsyncClient(verify=SSL_VERIFY) as client:
        with open(output_path, "a", encoding="utf-8") as fout:
            async def process_patient(patient: dict):
                pid = patient["patient_id"]
                prompt = patient["prompt"]
                true_genes = patient["true_genes"]

                if dry_run:
                    print(f"\n[Dry Run] Patient {pid}:")
                    print(prompt[:400] + "...\n")
                    return None

                async with sem:
                    resp_text = await call_aiva(client, api_url, api_key, model, prompt)

                predicted, scores = parse_predictions(resp_text or "")
                rank = find_rank(predicted, true_genes)

                rec = {
                    "patient_id": pid,
                    "disease_id": patient["disease_id"],
                    "ablation": "hpo+vcf+genes+disease",
                    "model": model,
                    "true_genes": true_genes,
                    "true_genes_ensg": patient["true_genes_ensg"],
                    "predicted": predicted,
                    "predicted_scores": scores,
                    "rank_of_true_gene": rank,
                    "category": patient["category"],
                    "broad_category": patient["broad_category"],
                    "in_gene_gene_kg": patient["in_gene_gene_kg"],
                    "in_disease_gene_kg": patient["in_disease_gene_kg"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                counter["done"] += 1
                top_gene = predicted[0] if predicted else "NONE"
                top_score = scores.get(top_gene, 0.0)
                true_gene_str = true_genes[0] if true_genes else "UNKNOWN"
                true_score = scores.get(true_gene_str, 0.0)

                status = f"RANK_{rank}" if rank is not None else "MISSED"
                if rank == 1:
                    status = "✓ RANK_1"

                in_model = "✓" if rank is not None else "✗"
                print(f"  [{counter['done']}/{total}] Patient {pid}: Top={top_gene}(s={top_score:.2f}) | "
                      f"True={true_gene_str}(rank={rank}, s={true_score:.2f}, in_model={in_model}) | {status}", flush=True)

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                results.append(rec)
                return rec

            tasks = [process_patient(p) for p in remaining]
            await asyncio.gather(*tasks)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Claude Haiku 5.5 on Gene Prioritization Benchmark")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS, help="Path to ranking_prompts.jsonl")
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS, help="Path to ranking_answers.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output predictions JSONL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model identifier")
    parser.add_argument("--limit", type=int, default=200, help="Number of patients to evaluate (0 = all)")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent request limit")
    parser.add_argument("--detach", action="store_true", help="Launch detached background daemon")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt preview without calling API")
    args = parser.parse_args()

    # If --detach is requested, spawn background process and exit
    if args.detach:
        output_dir = args.output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "haiku_ranking_run.log"
        pid_path = output_dir / "haiku_ranking_run.pid"

        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--prompts", str(args.prompts.resolve()),
            "--answers", str(args.answers.resolve()),
            "--output", str(args.output.resolve()),
            "--model", args.model,
            "--limit", str(args.limit),
            "--concurrency", str(args.concurrency),
        ]

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Detaching background run (limit={args.limit})\n")
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        with open(pid_path, "w", encoding="utf-8") as f:
            f.write(str(proc.pid))

        print(f"Detached background process started with PID {proc.pid}")
        print(f"Log:    {log_path}")
        print(f"Output: {args.output}")
        return

    print("=" * 70)
    print("STARTING CLAUDE HAIKU 5.5 GENE PRIORITIZATION RUN")
    print(f"Model:       {args.model}")
    print(f"Ablation:    hpo+vcf+genes+disease (Canonical RareDxBench Prompts)")
    print(f"Prompts:     {args.prompts}")
    print(f"Answers:     {args.answers}")
    print(f"Output:      {args.output}")
    print(f"Limit:       {args.limit}")
    print(f"Concurrency: {args.concurrency}")
    print("=" * 70, flush=True)

    dataset = load_dataset(args.prompts, args.answers, limit=args.limit)
    print(f"Loaded {len(dataset)} patients from {args.prompts}.")

    results = asyncio.run(run_benchmark(dataset, args.output, args.model, concurrency=args.concurrency, dry_run=args.dry_run))

    if not args.dry_run and results:
        print("\nComputing aggregate metrics...")
        metrics = compute_benchmark_metrics(results)
        summary_path = args.output.parent / f"{args.output.stem}_evaluation_summary.json"
        summary_payload = {
            "model": args.model,
            "ablation": "hpo+vcf+genes+disease",
            "n_evaluated": len(results),
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        print("\n" + "=" * 70)
        print("GENE PRIORITIZATION BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Patients Evaluated : {metrics.get('n_patients')}")
        print(f"Retention Rate     : {metrics.get('retention_rate', 0):.1%}")
        print(f"Recall@1           : {metrics.get('recall@1', 0):.1%}")
        print(f"Recall@5           : {metrics.get('recall@5', 0):.1%}")
        print(f"Recall@10          : {metrics.get('recall@10', 0):.1%}")
        print(f"Recall@20          : {metrics.get('recall@20', 0):.1%}")
        print(f"MRR                : {metrics.get('mrr')}")
        print(f"NDCG@10            : {metrics.get('ndcg@10')}")
        print(f"Mean Rank          : {metrics.get('mean_rank')}")
        print(f"Median Rank        : {metrics.get('median_rank')}")
        print(f"Evaluation summary saved to: {summary_path}")
        print("=" * 70)


if __name__ == "__main__":
    main()
