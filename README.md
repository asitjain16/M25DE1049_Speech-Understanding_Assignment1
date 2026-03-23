# Speech Understanding Assignment

Python implementation covering three questions on speech processing.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── mfcc_manual.py          # Q1: Manual MFCC pipeline
├── leakage_snr.py          # Q1: Spectral leakage & SNR analysis
├── voiced_unvoiced.py      # Q1: Voiced/unvoiced boundary detection
├── phonetic_mapping.py     # Q1: Forced alignment & RMSE
├── data/
│   ├── manifest.csv        # Q1 audio file manifest
│   ├── leakage_plots/      # Generated leakage/SNR plots
│   ├── vuv_plots/          # Generated boundary visualisations
│   └── textgrids/          # Generated TextGrid alignments
├── examples/               # Input audio files (0.wav, 1.wav, ...)
├── q2/
│   ├── train.py            # Q2: Disentangled speaker model training
│   ├── eval.py             # Q2: EER & TAR@FAR evaluation
│   ├── configs/model.yaml  # Q2: Training/eval config
│   ├── results/            # Q2: Generated metrics & plots
│   └── q2_readme.md        # Q2: Reproduction instructions
├── q3/
│   ├── audit.py            # Q3: Dataset bias audit
│   ├── privacymodule.py    # Q3: Privacy-preserving PyTorch module
│   ├── pp_demo.py          # Q3: Privacy transformation demo
│   ├── train_fair.py       # Q3: Fairness-loss ASR training
│   ├── evaluation_scripts/
│   │   └── eval_quality.py # Q3: Audio quality evaluation (SNR/spectral proxy)
│   ├── examples/           # Q3: Before/after audio pairs
│   └── results/            # Q3: Audit plots & quality metrics
└── src/Wav2TextGrid/       # Existing forced-alignment package
```

## Question 1: Cepstral Feature Extraction & Phoneme Boundary Detection

### Manual MFCC
```bash
python mfcc_manual.py
```
Runs on `examples/0.wav`, prints MFCC shape.

### Spectral Leakage & SNR
```bash
python leakage_snr.py
```
Prints comparison table and saves plots to `data/leakage_plots/`.

### Voiced/Unvoiced Detection
```bash
python voiced_unvoiced.py
```
Detects boundaries in `examples/0.wav`, saves visualisation to `data/vuv_plots/`.

### Phonetic Mapping & RMSE
```bash
python phonetic_mapping.py
```
Runs forced alignment on `examples/0.wav` and `examples/1.wav`, prints RMSE.

## Question 2: Disentangled Speaker Representation

See `q2/q2_readme.md` for full instructions.

```bash
# Train
python -m q2.train

# Evaluate
python q2/eval.py
```

## Question 3: Ethical Auditing & Privacy Preservation

### Bias Audit
```bash
python q3/audit.py
```
Audits Common Voice metadata, saves plots to `q3/results/`.

### Privacy Demo
```bash
python q3/pp_demo.py
```
Transforms `examples/0.wav` (male_old → female_young), saves to `q3/examples/`.

### Fairness Training
```bash
python q3/train_fair.py
```
Trains a simple ASR model with FairnessLoss on synthetic data.

### Quality Evaluation
```bash
python q3/evaluation_scripts/eval_quality.py
```
Evaluates SNR and spectral distortion of transformed audio pairs.
