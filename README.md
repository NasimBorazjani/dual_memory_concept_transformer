# Dual Memory Concept Transformer

Research prototype. The code trains a GPT-2 style decoder that augments token-level processing with sentence representations stored in short-term and long-term memory buffers.

## Repository Tour
- `model.py` – dual-memory transformer core and memory utilities.
- `train.py` – end-to-end training loop, dataset caching, and evaluation helpers.
- `inference_rollout.py` – reproduces training-time memory behaviour for inspection.
- `binding_eval.py`, `eval_hotpot.py` – behavioural probes aligned with the paper.
- `create_datasets.py` & `sentence_splitter.py` – data preparation utilities.

Sample datasets live under `data_wikitext/` and `hotpot_data/`. Large training corpora (e.g. `hotpot_train_v1.1.json`) are intentionally excluded from Git; place them locally if you need full runs.

## Getting Started
1. Install dependencies from `requirements.txt` (or replicate the environment used during experiments).
2. Launch training:
   ```bash
   python train.py --config your_config.json
   ```
3. Use `inference_rollout.py` or the evaluation scripts to inspect sentence-level behaviour.

Feel free to adapt the configuration and memory settings while iterating on the accompanying manuscript.
