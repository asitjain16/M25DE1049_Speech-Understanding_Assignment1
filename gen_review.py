from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import os

os.makedirs("q2", exist_ok=True)
OUT = "q2/review.pdf"

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

styles = getSampleStyleSheet()

title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=15, spaceAfter=4,
    textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
sub_s = ParagraphStyle("S", parent=styles["Normal"], fontSize=9.5, alignment=TA_CENTER,
    textColor=colors.grey, spaceAfter=2)
h1_s = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=12, spaceBefore=12,
    spaceAfter=3, textColor=colors.HexColor("#16213e"))
h2_s = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=10.5, spaceBefore=7,
    spaceAfter=2, textColor=colors.HexColor("#0f3460"))
body_s = ParagraphStyle("B", parent=styles["Normal"], fontSize=9.5, leading=14,
    spaceAfter=5, alignment=TA_JUSTIFY)
bullet_s = ParagraphStyle("BL", parent=styles["Normal"], fontSize=9.5, leading=14,
    spaceAfter=3, leftIndent=14, alignment=TA_JUSTIFY)

def H1(t): return Paragraph(t, h1_s)
def H2(t): return Paragraph(t, h2_s)
def P(t):  return Paragraph(t, body_s)
def B(t):  return Paragraph("&#8226;  " + t, bullet_s)
def HR():  return HRFlowable(width="100%", thickness=0.5,
               color=colors.HexColor("#cccccc"), spaceAfter=5)
def SP(h=5): return Spacer(1, h)

story = []

story += [
    Paragraph("Technical Critical Review", title_s),
    Paragraph("Disentangled Representation Learning for Environment-agnostic Speaker Recognition", title_s),
    Paragraph("Nam et al., Interspeech 2024 — arXiv:2406.14559", sub_s),
    SP(3), HR(), SP(3),
]

# 1. Problem Statement
story += [
    H1("1. Problem Statement"),
    P("Speaker recognition systems must identify individuals from their voice, yet every "
      "recording is a mixture of speaker-specific traits (age, gender, accent, vocal tract "
      "shape) and environmental factors (background noise, room reverberation, microphone "
      "characteristics). When the recording environment at test time differs from training "
      "conditions — the <i>environment mismatch</i> problem — state-of-the-art speaker "
      "embeddings degrade significantly, as demonstrated by the VoxSRC 2022/2023 and "
      "VC-Mix benchmarks."),
    P("Existing mitigations such as data augmentation with MUSAN noise and simulated RIRs "
      "reduce but do not eliminate this vulnerability, because they do not explicitly "
      "separate environmental information from the learned representation. The paper "
      "addresses this gap by proposing a post-hoc disentanglement framework that can be "
      "applied on top of any existing speaker embedding extractor without modifying its "
      "architecture."),
    SP(3), HR(),
]

# 2. Method
story += [
    H1("2. Method"),
    H2("2.1 Framework Overview"),
    P("The proposed framework wraps any pre-trained speaker embedding extractor with a "
      "lightweight auto-encoder that acts as a disentangler. Given an input embedding "
      "e_ij from the extractor, the encoder compresses it into a bottleneck vector e^z_ij "
      "(dim 1024 for ResNet-34, 512 for ECAPA-TDNN), which is then split equally into a "
      "speaker component e^spk and an environment component e^env. The decoder "
      "reconstructs the original embedding from the concatenation of both components."),
    H2("2.2 Triplet Batch Construction"),
    P("Each mini-batch index contains three utterances from the same speaker: x_i1 and "
      "x_i2 from the same video session (same environment), and x_i3 from a different "
      "video (different environment). Identical augmentation is applied to x_i1 and x_i2, "
      "while a different augmentation is applied to x_i3. This controlled construction "
      "provides a natural supervision signal for environment disentanglement."),
    H2("2.3 Objective Functions (5 losses)"),
    P("The total loss is a weighted sum of five terms:"),
    B("<b>L_recons</b> (reconstruction): L1 distance between input and reconstructed "
      "embedding, summed over all three triplet members. Prevents information collapse "
      "during disentanglement."),
    B("<b>L_spk</b> (speaker classification): Combined angular prototypical loss + "
      "softmax cross-entropy on e^spk. Ensures the speaker component retains "
      "discriminative speaker identity."),
    B("<b>L_env_env</b> (environment triplet): Triplet loss on e^env, pulling same-"
      "environment representations together and pushing different-environment ones apart."),
    B("<b>L_env_spk(G)</b> (adversarial): Environment triplet loss applied to e^spk "
      "through a Gradient Reversal Layer (GRL). Forces e^spk to be uninformative about "
      "the environment by reversing gradients during backpropagation."),
    B("<b>L_corr</b> (correlation): Mean Absolute Pearson Correlation between e^spk and "
      "e^env, minimising statistical dependence between the two components."),
    P("L_total = lambda_S * L_spk + lambda_R * L_recons + lambda_E * L_env_env "
      "+ lambda_adv * L_env_spk(G) + lambda_C * L_corr, "
      "with lambda_adv = 0.5 and all other lambdas = 1."),
    H2("2.4 Code Swapping"),
    P("During reconstruction, the speaker components e^spk_i2 and e^spk_i3 are swapped "
      "before being passed to the decoder. This cross-environment reconstruction forces "
      "the model to encode environment-invariant speaker identity in e^spk, since the "
      "decoder must reconstruct a valid embedding using a speaker component from a "
      "different environment."),
    SP(3), HR(),
]

# 3. Strengths
story += [
    H1("3. Strengths"),
    B("<b>Plug-and-play design:</b> The framework requires no modification to the "
      "underlying speaker network. It operates purely on the output embeddings, making "
      "it compatible with any existing extractor (demonstrated on ResNet-34 and "
      "ECAPA-TDNN)."),
    B("<b>Multi-objective training stability:</b> The reconstruction loss explicitly "
      "penalises information loss, addressing the well-known instability of pure "
      "adversarial DRL (GRL-only baseline shows high variance across seeds)."),
    B("<b>Strong empirical results:</b> Up to 16% EER improvement on wild-environment "
      "benchmarks (VoxSRC22/23, VC-Mix) and ~12% improvement on standard VoxCeleb1 "
      "sets, with lower standard deviation than the GRL-only baseline."),
    B("<b>Principled disentanglement:</b> The combination of adversarial learning, "
      "correlation minimisation, and code swapping provides three complementary "
      "mechanisms targeting the same goal, reducing the risk of any single mechanism "
      "failing."),
    B("<b>Reproducibility:</b> Code and model weights are publicly released at "
      "github.com/kaistmm/voxceleb-disentangler."),
    SP(3), HR(),
]

# 4. Weaknesses
story += [
    H1("4. Weaknesses"),
    B("<b>Equal split assumption:</b> The bottleneck is always split 50/50 between "
      "e^spk and e^env. There is no principled justification for this ratio; in "
      "practice, environmental information may require more or fewer dimensions than "
      "speaker information depending on the recording conditions."),
    B("<b>No explicit environment labels:</b> The framework relies entirely on video "
      "session proximity as a proxy for environment similarity. This is a weak "
      "supervision signal — two utterances from the same video may still have "
      "different noise levels, and utterances from different videos may share "
      "similar environments."),
    B("<b>Hyperparameter sensitivity:</b> Five loss weights (lambda values) must be "
      "tuned. The paper fixes lambda_adv = 0.5 and all others to 1 without ablation "
      "across different values, leaving open the question of sensitivity."),
    B("<b>Computational cost:</b> Training requires a single NVIDIA RTX 4090 (24 GB) "
      "for ~300 epochs. The triplet batch construction and three discriminators add "
      "significant overhead compared to a standard speaker recognition pipeline."),
    B("<b>Limited backbone diversity:</b> Only two backbones (ResNet-34, ECAPA-TDNN) "
      "are evaluated. Generalisation to transformer-based extractors (e.g., "
      "WavLM-based systems) is not demonstrated."),
    SP(3), HR(),
]

# 5. Assumptions
story += [
    H1("5. Key Assumptions"),
    B("<b>Environment = video session:</b> The framework assumes that utterances from "
      "the same video share the same acoustic environment. This holds for VoxCeleb "
      "(YouTube celebrity interviews) but may not generalise to other datasets."),
    B("<b>Linear separability of speaker and environment:</b> The auto-encoder uses a "
      "single fully-connected layer for both encoder and decoder. This implicitly "
      "assumes that speaker and environment information are linearly separable in the "
      "embedding space of the pre-trained extractor."),
    B("<b>Fixed dimensionality split:</b> The 50/50 split assumes equal information "
      "content in both components, which is unlikely to hold in general."),
    B("<b>Augmentation as environment proxy:</b> Different augmentation types "
      "(reverb vs. noise) are used to simulate environment mismatch. Real-world "
      "environment variation is far more complex and may not be well-captured by "
      "MUSAN + RIR augmentation alone."),
    SP(3), HR(),
]

# 6. Experimental Validity
story += [
    H1("6. Experimental Validity"),
    P("The experimental setup is generally rigorous. The authors train all models "
      "three times with different random seeds and report mean and standard deviation, "
      "which is good practice for assessing training stability. Six evaluation sets "
      "are used, covering both standard (Vox1-O/E/H) and wild-environment "
      "(VoxSRC22/23, VC-Mix) conditions."),
    P("However, several concerns limit the strength of the conclusions:"),
    B("The ablation study is limited. The paper does not systematically ablate "
      "individual loss components (e.g., removing L_corr or L_env_env alone) to "
      "quantify each contribution. It is unclear which components drive the "
      "improvement."),
    B("The GRL baseline [7] is the only DRL comparison. No comparison is made "
      "against other recent environment-robust methods (e.g., domain adversarial "
      "training, style transfer, or noise-robust pooling)."),
    B("The VC-Mix dataset is introduced by the same research group, which raises "
      "a potential evaluation bias — the benchmark may be better suited to the "
      "proposed method's assumptions."),
    B("Inference cost is not reported. Since the framework adds an auto-encoder "
      "at inference time, latency and memory overhead should be quantified for "
      "real-time deployment scenarios."),
    SP(3), HR(),
]

# 7. Proposed Improvement
story += [
    H1("7. Proposed Improvement: Variational Bottleneck with Adaptive Split"),
    H2("7.1 Motivation"),
    P("The fixed 50/50 split between e^spk and e^env is the most arbitrary design "
      "choice in the paper. In practice, the amount of environmental information "
      "encoded in a speaker embedding varies with recording conditions. A rigid "
      "split may force the model to allocate too many dimensions to environment "
      "in clean conditions (wasting capacity) or too few in noisy conditions "
      "(leaving residual environment in e^spk)."),
    H2("7.2 Proposed Change"),
    P("Replace the fixed split with a <b>Variational Information Bottleneck (VIB)</b> "
      "that learns the optimal split ratio end-to-end. Specifically:"),
    B("Encode e^z into two Gaussian distributions: mu_spk, sigma_spk (speaker) "
      "and mu_env, sigma_env (environment), with learnable dimensionality "
      "controlled by a sparsity prior."),
    B("Add a KL-divergence term to the loss: L_KL = KL(q(e^spk|e^z) || p(e^spk)) "
      "+ KL(q(e^env|e^z) || p(e^env)), encouraging each component to use only "
      "as many dimensions as needed."),
    B("Use a learnable gating vector g in [0,1]^D (sigmoid-activated) that "
      "soft-assigns each dimension of e^z to either the speaker or environment "
      "component, replacing the hard 50/50 split."),
    H2("7.3 Expected Benefit"),
    P("This modification allows the model to dynamically allocate more capacity "
      "to the speaker component when environmental variation is low, and more to "
      "the environment component when noise is high. The VIB framework also "
      "provides a principled information-theoretic justification for the "
      "disentanglement, replacing the heuristic equal split."),
    H2("7.4 Evaluation"),
    P("The improvement can be evaluated by: (1) comparing EER on VoxSRC22/23 and "
      "VC-Mix against the original fixed-split model; (2) visualising the learned "
      "gating vector g to verify that the model assigns meaningful dimensions to "
      "each component; (3) measuring performance under varying SNR conditions to "
      "confirm adaptive capacity allocation. The implementation is in q2/train.py "
      "as the DisentangledModel with configurable content_dim and speaker_dim."),
    SP(5), HR(),
    Paragraph(
        "Reference: Nam et al., 'Disentangled Representation Learning for "
        "Environment-agnostic Speaker Recognition', Interspeech 2024, arXiv:2406.14559.",
        ParagraphStyle("ref", parent=styles["Normal"], fontSize=8.5,
                       textColor=colors.grey, alignment=TA_CENTER)
    ),
]

doc.build(story)
print("Generated: " + OUT)
