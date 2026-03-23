# Q2: Disentangled Speaker Representation

## Paper
"Disentangled Representation Learning for Environment-agnostic Speaker Recognition"
https://arxiv.org/abs/2406.14559

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m q2.train
# or
python q2/train.py
```

Config: `q2/configs/model.yaml`

By default uses a synthetic dataset. To use LibriSpeech, set `data_dir` in the config and ensure the dataset is downloaded.

Checkpoints are saved to `q2/checkpoints/best_model.pt`.

## Evaluation

```bash
python q2/eval.py
```

Requires a trained checkpoint at `q2/checkpoints/best_model.pt`.

Outputs:
- `q2/results/metrics.json` — EER and TAR@FAR=0.01
- `q2/results/eer_curve.png` — FAR vs FRR curve

## Results

After running eval.py, results are in `q2/results/`:
- `metrics.json` — numeric metrics
- `eer_curve.png` — detection error tradeoff curve

## Checkpoint Correspondence

| Checkpoint | Dataset | EER | TAR@FAR=0.01 |
|---|---|---|---|
| `q2/checkpoints/best_model.pt` | Synthetic / LibriSpeech | see metrics.json | see metrics.json |
