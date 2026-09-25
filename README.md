# H-MoE-TKG

Reference implementation of **Adaptive Modality Routing for Temporal Knowledge Graph Forecasting with Heterogeneous Mixture-of-Experts**.

The implementation follows the manuscript: a structural topology expert, temporal Transformer expert, frozen-language-model semantic adapter, two-layer query router, DistMult full-entity decoder, diversity loss, and entropy-based routing regularizer. The default configuration uses soft routing, so all three experts are evaluated for every query.

## Repository layout

- `src/hmoe_tkg/`: model, temporal-history construction, losses, metrics, and training code.
- `scripts/preprocess_icews.py`: convert raw ICEWS split files to the canonical numeric format.
- `scripts/cache_text_features.py`: cache frozen BERT features for observed `(subject, relation)` query pairs.
- `scripts/train.py`: train one seed with validation-based early stopping.
- `scripts/evaluate.py`: time-aware filtered full-entity evaluation.
- `scripts/run_seeds.py`: run seeds `1, 7, 21, 42, 2026`.
- `scripts/summarize_runs.py`: mean, sample standard deviation, and optional paired significance test.
- `scripts/plot_routing_umap.py`: routing-space UMAP with the manuscript parameters.
- `configs/`: configurations for ICEWS14, ICEWS18, and ICEWS05-15.
- `tests/`: a self-contained synthetic-data smoke test.

## Installation

The exact manuscript environment is recorded in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate hmoe-tkg
pip install -e . --no-deps
```

Alternatively, create a Python 3.10 environment and install the pinned requirements:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

Install the CUDA-compatible PyTorch and DGL wheels for the target machine if the generic packages selected by `pip` do not match the local CUDA runtime.

## Data format

Raw split files must contain one quadruple per line:

```text
subject<TAB>relation<TAB>object<TAB>timestamp
```

The fields may be labels or integer identifiers. Preprocessing creates numeric `train.tsv`, `valid.tsv`, and `test.tsv`, plus entity/relation label maps and metadata. Vocabulary is learned from the training split by default; unseen validation/test identifiers cause an explicit error rather than silently introducing future vocabulary.

```bash
python scripts/preprocess_icews.py \
  --train raw/ICEWS14/train.txt \
  --valid raw/ICEWS14/valid.txt \
  --test raw/ICEWS14/test.txt \
  --output data/ICEWS14/processed
```

Optional label files contain `identifier<TAB>human readable label`. Cache the frozen semantic representations with:

```bash
python scripts/cache_text_features.py \
  --data-dir data/ICEWS14/processed \
  --output data/ICEWS14/semantic_cache.pt \
  --model bert-base-uncased --max-length 64
```

## Training and evaluation

```bash
python scripts/train.py --config configs/icews14.yaml --seed 42
python scripts/evaluate.py \
  --config configs/icews14.yaml \
  --checkpoint outputs/icews14/seed_42/best.pt \
  --split test
```

Run all five manuscript seeds and summarize them:

```bash
python scripts/run_seeds.py --config configs/icews14.yaml
python scripts/summarize_runs.py --runs outputs/icews14/seed_*/test_metrics.json
```

The evaluator performs full-entity ranking and removes other true objects for the same `(subject, relation, timestamp)` query before calculating MRR and Hits@1/3/10.

## Reproducibility notes

- Query history is restricted to facts with timestamps strictly earlier than the query timestamp.
- The structural expert uses the four latest historical snapshots and at most ten neighbors per relation by default.
- Temporal sequences are left-padded to 20 events and use causal and padding masks.
- BERT is frozen; only the semantic adapter is optimized with the graph model.
- Random seeds are propagated to Python, NumPy, PyTorch, CUDA, and DGL when installed.
- Each output directory records the resolved configuration, best checkpoint, validation metrics, test metrics, and routing weights.

The synthetic smoke test does not reproduce paper results; it only checks that preprocessing, forward/backward computation, checkpointing, and filtered evaluation execute correctly.
