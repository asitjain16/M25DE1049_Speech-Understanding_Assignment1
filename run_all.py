import sys
import os
import logging
import numpy as np

logging.disable(logging.CRITICAL)
SEP = "=" * 70

def banner(title):
    print()
    print(SEP)
    print("  " + title)
    print(SEP)

# ── QUESTION 1 ──────────────────────────────────────────────────────────────
banner("QUESTION 1: Cepstral Feature Extraction & Phoneme Boundary Detection")

banner("Q1.1 -- Manual MFCC Pipeline (no librosa)")
from mfcc_manual import extract_mfcc
for wav in ["examples/0.wav", "examples/1.wav"]:
    mfccs = extract_mfcc(wav)
    print("  {} -> shape={} | all_finite={}".format(
        wav, mfccs.shape, bool(np.all(np.isfinite(mfccs)))))

banner("Q1.2 -- Spectral Leakage & SNR (Rectangular / Hamming / Hanning)")
from leakage_snr import analyze_windows
from scipy.io import wavfile
sr, sig = wavfile.read("examples/0.wav")
if sig.ndim > 1:
    sig = sig.mean(axis=1)
sig = sig.astype(np.float64) / np.iinfo(sig.dtype).max
results = analyze_windows(sig, sr)
print("  {:<14} {:>10}   {:>14}".format("Window", "SNR (dB)", "Leakage Ratio"))
print("  " + "-" * 44)
for w, v in results.items():
    print("  {:<14} {:>10.2f}   {:>14.6f}".format(w, v["snr_db"], v["leakage_ratio"]))

banner("Q1.3 -- Voiced/Unvoiced/Silence Boundary Detection")
from voiced_unvoiced import detect_boundaries, visualize_boundaries
segs = detect_boundaries("examples/0.wav")
print("  examples/0.wav -> {} segments detected".format(len(segs)))
print("  {:>8}  {:>8}  {:>6}  {}".format("Start", "End", "Dur", "Label"))
print("  " + "-" * 38)
for s in segs[:12]:
    print("  {:8.3f}  {:8.3f}  {:6.3f}  {}".format(
        s.start_time, s.end_time, s.duration, s.label))
if len(segs) > 12:
    print("  ... ({} more segments)".format(len(segs) - 12))
visualize_boundaries("examples/0.wav", segs, "data/vuv_plots/0_boundaries.png")
print("  Visualisation saved -> data/vuv_plots/0_boundaries.png")

banner("Q1.4 -- Phonetic Mapping & RMSE (Wav2TextGrid forced alignment)")
from phonetic_mapping import run_q1_pipeline
for wav, lab in [("examples/0.wav", "examples/0.lab"),
                 ("examples/1.wav", "examples/1.lab")]:
    r = run_q1_pipeline(wav, lab, "data")
    print("  {}".format(wav))
    print("    Manual segments   : {}".format(len(r["manual_segments"])))
    print("    Reference segments: {}".format(len(r["reference_segments"])))
    print("    RMSE              : {:.4f} s".format(r["rmse"]))

print()
print(SEP)
print("  Q1 COMPLETE")
print(SEP)

# ── QUESTION 2 ──────────────────────────────────────────────────────────────
banner("QUESTION 2: Disentangled Speaker Representation (arxiv 2406.14559)")

banner("Q2.1 -- Training DisentangledModel (synthetic data, 5 epochs)")
import torch
import yaml
from q2.train import build_model, build_dataloaders, train_epoch

with open("q2/configs/model.yaml") as f:
    config = yaml.safe_load(f)

config["training"]["n_epochs"] = 5
config["training"]["checkpoint_dir"] = "q2/checkpoints"
os.makedirs("q2/checkpoints", exist_ok=True)

device = torch.device("cpu")
model = build_model(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
train_loader, val_loader = build_dataloaders(config)

for epoch in range(1, 6):
    losses = train_epoch(model, train_loader, optimizer, model.speaker_classifier, config)
    print("  Epoch {}/5 | total={:.4f}  recon={:.4f}  cls={:.4f}  dis={:.4f}".format(
        epoch, losses["total"], losses["recon"], losses["cls"], losses["dis"]))

ckpt_path = "q2/checkpoints/best_model.pt"
torch.save({"model_state_dict": model.state_dict(), "config": config}, ckpt_path)
print("  Checkpoint saved -> {}".format(ckpt_path))

banner("Q2.2 -- Evaluation: EER & TAR@FAR=0.01")
from q2.eval import extract_embeddings, compute_eer, compute_tar_at_far
model.eval()
embeddings, labels = extract_embeddings(model, val_loader)
eer = compute_eer(embeddings, labels)
tar = compute_tar_at_far(embeddings, labels, far=0.01)
print("  Embeddings shape : {}".format(embeddings.shape))
print("  EER              : {:.4f}  ({:.1f}%)".format(eer, eer * 100))
print("  TAR @ FAR=0.01   : {:.4f}  ({:.1f}%)".format(tar, tar * 100))

print()
print(SEP)
print("  Q2 COMPLETE")
print(SEP)

# ── QUESTION 3 ──────────────────────────────────────────────────────────────
banner("QUESTION 3: Ethical Auditing & Privacy Preservation")

banner("Q3.1 -- Bias Audit (Common Voice / synthetic fallback)")
from q3.audit import load_dataset_metadata, compute_representation_stats, generate_audit_plots
df = load_dataset_metadata(max_samples=2000)
report = compute_representation_stats(df)
generate_audit_plots(report, out_dir="q3/results")
print("  Gender distribution:")
for k, v in sorted(report.gender_distribution.items(), key=lambda x: -x[1]):
    print("    {:<20} {:>6.1f}%".format(k, v * 100))
print("  Age distribution (top 5):")
for k, v in sorted(report.age_distribution.items(), key=lambda x: -x[1])[:5]:
    print("    {:<20} {:>6.1f}%".format(k, v * 100))
print("  Documentation debt items : {}".format(len(report.documentation_debt_items)))
print("  Underrepresented groups  : {}".format(len(report.underrepresented_groups)))

banner("Q3.2 -- PrivacyModule forward pass (male_old -> female_young)")
from q3.privacymodule import PrivacyModule, ATTRIBUTE_MAP
pm = PrivacyModule(input_dim=80, latent_dim=64, n_attributes=4)
x = torch.randn(4, 80)
src = torch.tensor([ATTRIBUTE_MAP["male_old"]] * 4)
tgt = torch.tensor([ATTRIBUTE_MAP["female_young"]] * 4)
out = pm(x, src, tgt)
print("  Input shape      : {}".format(tuple(x.shape)))
print("  Output shape     : {}".format(tuple(out.shape)))
print("  Shape preserved  : {}".format(out.shape == x.shape))

banner("Q3.3 -- Privacy Demo (pp_demo.py)")
import importlib.util
spec_mod = importlib.util.spec_from_file_location("pp_demo", "q3/pp_demo.py")
pp = importlib.util.module_from_spec(spec_mod)
spec_mod.loader.exec_module(pp)

banner("Q3.4 -- Fairness Loss Training (3 epochs, synthetic data)")
logging.disable(logging.NOTSET)
from q3.train_fair import SimpleASRModel, FairnessLoss, train_with_fairness
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(42)
N, INPUT_DIM, N_CLASSES, N_GROUPS = 200, 40, 10, 3
feats = torch.randn(N, INPUT_DIM)
targets = torch.randint(0, N_CLASSES, (N,))
group_ids = torch.randint(0, N_GROUPS, (N,))
dataset = TensorDataset(feats, targets, group_ids)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
asr_model = SimpleASRModel(INPUT_DIM, N_CLASSES)
fl = FairnessLoss(["group_0", "group_1", "group_2"], lambda_fair=0.1)
opt = torch.optim.Adam(asr_model.parameters(), lr=1e-3)
train_with_fairness(asr_model, loader, opt, fl, n_epochs=3)
asr_model.eval()
with torch.no_grad():
    logits = asr_model(feats)
    final = fl(logits, targets, group_ids)
print("  Final fairness loss: {:.4f}".format(final.item()))
logging.disable(logging.CRITICAL)

banner("Q3.5 -- Audio Quality Evaluation (SNR / Spectral Distortion proxy)")
spec2 = importlib.util.spec_from_file_location("eval_quality", "q3/evaluation_scripts/eval_quality.py")
eq = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(eq)

print()
print(SEP)
print("  Q3 COMPLETE")
print(SEP)

print()
print(SEP)
print("  ALL QUESTIONS COMPLETE")
print(SEP)
