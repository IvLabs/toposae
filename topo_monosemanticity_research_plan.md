RESEARCH PROJECT PLAN
Topographic Training as a Path to Monosemanticity

Domain: NeuroAI x Mechanistic Interpretability
Estimated Duration: 6-8 weeks (part-time)
Difficulty: ★★★☆☆  Intermediate-Advanced
Publication Target: NeurIPS Neuro-AI Workshop / ICLR Mechanistic Interpretability Workshop



# 1. Background & Motivation
## 1.1 The Problem with Standard Neural Networks
Modern deep networks, including Vision Transformers (ViTs), are powerful but notoriously uninterpretable. A central reason is polysemanticity: individual neurons fire for multiple semantically unrelated stimuli. This is not a coincidence — it is a consequence of the superposition hypothesis (Elhage et al., 2022), which argues that networks with more features than neurons are forced to ‘pack’ multiple features into the same unit via superposition in activation space.


## 1.2 Topographic Organization in the Brain
The biological brain exhibits topographic organization: neurons performing similar computations are physically adjacent on the cortical surface. This is not incidental — it is thought to minimize wiring length, reducing metabolic cost. Key examples include:
- V1: Orientation columns, retinotopic maps, color maps in regular pinwheel patterns
- VTC: Face-selective (FFA), place-selective (PPA), and body-selective (EBA) regions are spatially segregated
- Language cortex: Syntactic vs. semantic clusters observed in left temporal/frontal regions

The key neuroscience claim: spatial proximity ⇒ functional similarity. The brain enforces this efficiently with local connectivity.
## 1.3 The TDANN / TopoNets Framework
Prior work has operationalized this principle in artificial networks:


TopoNets introduces TopoLoss: a loss that encourages nearby units to develop similar weights (inspired by synaptic pruning), rather than just similar activations. It achieves the best brain-alignment scores to date without a significant accuracy drop.
## 1.4 The Gap: Nobody Has Asked the Mech Interp Question


# 2. Central Hypothesis
H1 (Polysemanticity): Topographic training reduces the fraction of polysemantic neurons, increasing the average ‘monosemanticity score’ per unit compared to a baseline ViT.

H2 (Superposition): An SAE trained on the topographic ViT’s residual stream will require fewer active features per image (lower L0 norm) to achieve the same reconstruction loss — i.e., the feature basis is less entangled.

H3 (Causal Purity): Activation patching of the topographic model’s ‘face region’ (a spatial cluster on the cortical sheet) will more cleanly suppress face classification performance than patching an equivalent-sized random set of units in the baseline.

H4 (Brain Alignment): The spatial structure of the topographic ViT’s cortical sheet will predict the spatial arrangement of voxel responses in the NSD fMRI dataset — spatially adjacent model units should predict spatially adjacent voxels.

These hypotheses are ordered from most to least exploratory. H3 and H4 are the novel contributions; H1 and H2 establish the mechanistic case.


# 3. Methodology
## 3.1 Phase 1 — Model Training
### Models to Train
Train three model variants, holding all other hyperparameters fixed:


### Dataset
- ImageNet-100 (100-class subset) for speed during development
- Full ImageNet-1K for final reported results
- Stimulus sets for probing: THINGS dataset (1,854 object concepts), NSD images (73,000 natural scenes)

### TopoLoss Implementation
Following TopoNets (Deb et al., 2025), apply TopoLoss to every attention layer in the ViT. The loss is computed as:


Key difference from TDANN: TopoNets operates on unit weights (static), not activations (dynamic, batch-dependent). This makes the loss cheaper to compute and more stable.

Implementation note: Use TopoNets’ open-source code (github.com/toponets) rather than re-implementing from scratch. The core contribution of this project is the analysis pipeline, not the training recipe.

## 3.2 Phase 2 — Polysemanticity Analysis
### Monosemanticity Score
For each unit u in layer l, compute selectivity across C concept categories using the THINGS dataset:


Additionally, plot category selectivity maps: t-values for face/body/scene/object selectivity, heatmapped onto the 2D cortical sheet. Visually verify that topographic models show spatial clustering.

## 3.3 Phase 3 — Superposition Analysis via SAE
### What is a Sparse Autoencoder (SAE)?
An SAE is a one-hidden-layer autoencoder with a sparsity constraint on the hidden layer. When trained on a model’s residual stream, the hidden layer’s M >> N features recover the true feature basis from the compressed neuron basis — the ‘true’ monosemantic features the network computes via superposition.

### SAE Training Protocol
- Extract residual stream activations at the middle-most ViT block (e.g., layer 6 of 12) for 100K ImageNet images
- Train SAE with expansion factor 8x (N neurons → 8N features), L1 sparsity on hidden layer
- Evaluate on held-out 10K images

### Metrics

The key comparison: same SAE architecture, same training budget, applied to baseline vs. topographic ViT. Any improvement in L0 or dead features is attributable to the topographic training.

## 3.4 Phase 4 — Causal Analysis via Activation Patching
### Setup
Activation patching (also known as causal tracing) tests whether a set of units causally controls a behaviour — not just correlates with it.

### Protocol
- Identify the ‘face cluster’ in the topographic ViT: the spatial region of the cortical sheet with highest face selectivity (from Phase 2)
- Select an equivalent-sized 
- For each test image (face vs. non-face): run the model twice — once normally, once with the face-cluster activations replaced (patched) by activations from a non-face image
- Measure the change in face classification logit: Δlogit = logit(clean) − logit(patched)
- Compare Δlogit for: topographic face cluster vs. random set of units in baseline model (same unit count)

### Expected Result
If H3 is correct: patching the topographic face cluster produces a larger and more consistent Δlogit than patching random units in the baseline — evidence that topographic organization colocalizes causally important circuitry.

## 3.5 Phase 5 — Brain Alignment (Optional Extension)
Using the Natural Scenes Dataset (NSD; Allen et al., 2022) — fMRI responses from 8 subjects viewing 73,000 natural images — test whether the topographic ViT’s cortical sheet structure predicts the spatial layout of voxel preferences.

- Compute unit-to-voxel encoding model (linear regression): predict voxel response from cortical sheet unit
- Measure whether spatially adjacent units (on the cortical sheet) predict spatially adjacent voxels (in VTC)
- Compare spatial correlation: topographic ViT vs. baseline

This is an optional but high-impact extension that directly connects the interpretability findings to neuroscience data.


# 4. Controls & Ablations
To make the results rigorous and reviewer-proof, the following controls are essential:



# 5. Project Timeline



# 6. Technical Stack

## 6.1 Core Libraries

## 6.2 Dataset Access
- ImageNet-100: Freely available; standard 100-class subset
- THINGS: Open access at things-initiative.org (1,854 object images)
- NSD: Requires registration at naturalscenesdataset.org (free for academic use)


# 7. Risks & Mitigations



# 8. Novelty & Contribution
## 8.1 What Is New

## 8.2 Publication Strategy


# 9. Key References

## Topographic Networks
- Lee et al. (2020). Topographic deep artificial neural networks reproduce the hallmarks of the primate inferior temporal cortex face processing network. bioRxiv.
- Margalit et al. (2024). A unifying framework for functional organization in early and higher ventral visual cortex. Neuron, 112(14).
- Rathi et al. (2025). TopoLM: Brain-like spatio-functional organization in a topographic language model. ICLR 2025 (Oral).
- Deb et al. (2025). TopoNets: High performing vision and language models with brain-like topography. ICLR 2025 (Spotlight).

## Mechanistic Interpretability
- Elhage et al. (2022). Toy models of superposition. Anthropic Transformer Circuits Thread.
- Bricken et al. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. Anthropic Transformer Circuits Thread.
- Templeton et al. (2024). Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. Anthropic.
- Wang et al. (2022). Interpretability in the wild: A circuit for indirect object identification in GPT-2. arXiv.

## Brain Datasets
- Allen et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and computational intelligence. Nature Neuroscience.
- Hebart et al. (2019). THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images. PLOS ONE.

# 10. Success Criteria
The project is considered successful if it achieves the following by end of Week 7:



END OF PROJECT PLAN
This plan is a living document. Revise the timeline and scope after Week 2 once training results are available and the computational budget is clear.

<!-- Table 1 -->
| ABSTRACT
Topographic neural networks (TDANNs, TopoNets) enforce spatial smoothness on model units, producing brain-like feature clustering. While prior work has evaluated these models on brain alignment benchmarks, nobody has asked whether topographic training also reduces polysemanticity — the phenomenon where individual neurons respond to multiple unrelated concepts.

This project bridges the NeuroAI and Mechanistic Interpretability communities by training topographic Vision Transformers (ViT) and measuring the effect on feature geometry: polysemanticity scores, feature superposition (via Sparse Autoencoders), and causal purity via activation patching. The central hypothesis is that the spatial smoothness pressure implicitly acts as an anti-superposition regularizer, producing cleaner, more interpretable internal representations. |

<!-- Table 2 -->
| THE SUPERPOSITION HYPOTHESIS (Elhage et al., 2022)
If a model has N neurons but needs to represent M >> N features, it uses near-orthogonal directions in activation space to pack M features into N dimensions. This works because random near-orthogonal vectors interfere only weakly. The result: each neuron participates in many features, and no feature ‘owns’ a single neuron.

Consequence: SAEs (Sparse Autoencoders) are needed to recover the true M-dimensional feature space from the N-dimensional neuron space. |

<!-- Table 3 -->
| Model | Year | Approach | Coverage |
| TDANN (Lee et al.) | 2020 | Activation correlation loss on 2D cortical sheet | CNN, visual cortex only |
| TDANN (Margalit et al.) | 2024 | Unified loss for V1 + VTC in ResNets | CNN, vision (V1 to IT) |
| TopoLM (Rathi et al.) | 2025 | Smoothness loss for transformer LMs | LLM, language cortex |
| TopoNets (Deb et al.) | 2025 | Weight-level smoothness (synaptic pruning metaphor) | ResNet + ViT + GPT-Neo |

<!-- Table 4 -->
| CENTRAL RESEARCH GAP
All prior topographic network papers evaluate models using:
  (a) Smoothness metrics (do nearby units correlate?)
  (b) Brain score / RSA (does it align with neural data?)
  (c) ImageNet accuracy (does it still perform?)

Nobody has measured whether topographic training changes the INTERNAL FEATURE GEOMETRY:
  — Does polysemanticity decrease?
  — Does feature superposition reduce?
  — Are features causally cleaner (better activation patching)?

This is the gap this project fills. |

<!-- Table 5 -->
| Variant | TopoLoss? | Purpose |
| ViT-S/16 (Baseline) | No (α = 0) | Control: standard training |
| ViT-S/16 (Topo-Weak) | Yes (α = 0.1) | Light topographic pressure |
| ViT-S/16 (Topo-Strong) | Yes (α = 1.0) | Strong topographic pressure |

<!-- Table 6 -->
| TopoLoss Formulation
1. Assign each unit u a fixed 2D position (x_u, y_u) on a simulated cortical sheet (initialized once, fixed for training)

2. At each forward pass, sample K random spatial neighbourhoods N_k

3. For each neighbourhood, compute pairwise weight similarity:
   sim(u, v) = cosine_similarity(W_u, W_v)

4. Compute spatial loss:
   L_topo = -Σ_{k} Σ_{(u,v) ∈ N_k} sim(u, v) × f(1 / dist(u, v))
   where f is an exponential decay function

5. Total loss: L = L_CE + α × L_topo |

<!-- Table 7 -->
| Monosemanticity Score Definition
1. For each unit u, collect activations a_{u,c} for all images of category c
2. Compute mean activation per category: μ_{u,c} = mean(a_{u,c})
3. Selectivity vector: s_u = [μ_{u,1}, μ_{u,2}, ..., μ_{u,C}]
4. Monosemanticity score: M_u = max(s_u) / sum(s_u) ∈ [1/C, 1]
   — M_u = 1: unit fires ONLY for one category (monosemantic)
   — M_u = 1/C: unit fires equally for all (maximally polysemantic)

Report: distribution of M_u across all units; fraction with M_u > 0.5 as a summary stat |

<!-- Table 8 -->
| Metric | What it measures | Expected direction |
| L0 norm (active features/image) | Sparsity of the feature basis | Lower in topographic model |
| Reconstruction loss (MSE) | How well SAE recovers activations | Similar or better |
| Dead feature fraction | Features never active (wasted capacity) | Lower in topographic model |
| Feature correlation matrix | Entanglement between recovered features | Sparser in topographic model |

<!-- Table 9 -->
| Control | Purpose | How |
| Random unit selection patching | Verify face cluster patching isn’t trivially due to unit count | Match unit count, randomly sample from baseline |
| α sweep (0, 0.01, 0.1, 0.5, 1.0) | Show monosemanticity is a monotonic function of topographic pressure | Train 5 models, plot M_u vs. α |
| Layer-wise analysis | Confirm effect isn’t layer-specific | Report all metrics per layer |
| ResNet baseline | Check if effect is ViT-specific | Apply same pipeline to TopoNet-ResNet50 |
| Post-hoc topography control | Distinguish trained vs. post-hoc topography | Re-organize baseline ViT units via SOM, re-run analysis |

<!-- Table 10 -->
| Week | Phase | Deliverable | Est. Hours |
| 1 | Setup & Reproduction | Clone TopoNets repo, verify training on ImageNet-100, confirm smoothness metrics match paper | 10–12h |
| 2 | Training Run | Train 3 ViT variants (α = 0, 0.1, 1.0) on ImageNet-100 to convergence | 4h (GPU) + monitoring |
| 3 | Phase 2: Polysemanticity | Monosemanticity scores, category selectivity maps, smoothness plots | 12–15h |
| 4 | Phase 3: SAE | SAE training + evaluation on all 3 models, L0/dead feature comparisons | 15–18h |
| 5 | Phase 4: Patching | Activation patching pipeline, face cluster identification, Δlogit comparison | 12–15h |
| 6 | Controls & Ablations | α sweep, layer-wise analysis, post-hoc control | 10–12h |
| 7 | Optional: Brain Alignment | NSD encoding model, spatial correlation analysis | 10–15h |
| 8 | Writing & Figures | Paper draft (workshop format: 4+1 pages), final figures | 15–20h |

<!-- Table 11 -->
| Component | Library / Tool | Notes |
| Base model | timm (ViT-S/16) | Pre-trained weights + training recipe |
| TopoLoss | TopoNets codebase (GitHub) | Do NOT re-implement; extend their code |
| Training | PyTorch + ffcv | ffcv for fast ImageNet loading |
| Activation extraction | transformer_lens or custom hooks | Hook into each attention layer |
| SAE | Custom PyTorch | ~50 lines; use EleutherAI’s SAE-vis for visualization |
| Polysemanticity | rsatoolbox + custom | THINGS dataset stimulus set |
| Brain alignment | nilearn + nibabel | NSD dataset (requires data access) |
| Visualization | matplotlib + seaborn | Heatmaps on 2D cortical sheet |
| Compute | Single A100 (40GB) or Colab Pro+ | ~6h per full training run |

<!-- Table 12 -->
| Risk | Likelihood | Mitigation |
| Null result: H1 fails (no polysemanticity reduction) | Medium | Still publishable: negative result with novel analysis pipeline; reframe as ‘topography is not sufficient for monosemanticity’ |
| SAE training instability | Low | Use established SAE training recipe (Anthropic’s open-source SAE trainer); sweep L1 penalty |
| NSD data access delay | Medium | This phase is optional; project is complete without it |
| Compute constraints | Low | ImageNet-100 trains in ~3h on A100; use Colab Pro+ if needed |
| TopoNets code incompatibility | Low | Paper is ICLR 2025 Spotlight; code is clean and well-documented |

<!-- Table 13 -->
| NOVEL CONTRIBUTIONS
1. FIRST analysis of polysemanticity in topographic neural networks. All prior work (TDANN, TopoNets, TopoLM) evaluates only brain alignment and accuracy. Nobody has measured unit-level feature geometry.

2. FIRST application of SAEs to a topographic model. The Anthropic/DeepMind SAE literature applies exclusively to standard transformer LMs. Applying it to a topographic ViT is unexplored.

3. FIRST causal analysis of topographic clusters. Prior work shows that face-selective clusters emerge. This project tests whether they are causally sufficient for face recognition (activation patching).

4. Two-community bridge. The project speaks simultaneously to the NeuroAI community (brain alignment, topography) and the Mechanistic Interpretability community (polysemanticity, SAEs, patching). Papers that cross community boundaries tend to have outsized impact. |

<!-- Table 14 -->
| Target Venue | Format | Framing |
| NeurIPS 2025 Neuro-AI Workshop | 4-page workshop paper | Primary target; strong fit |
| ICLR 2026 Mech Interp Workshop | 4-page workshop paper | Alternative / parallel submission |
| arXiv preprint | Full paper (8-10 pages) | Release concurrently with workshop submission |
| NeurIPS 2026 Main | Full paper | If results are strong, extend with NSD analysis |

<!-- Table 15 -->
| Criterion | Threshold | Priority |
| Smoothness metric reproduces TopoNets results | Within 5% of reported values | MUST HAVE |
| Monosemanticity score differs between topo vs. baseline | p < 0.05, effect size Cohen’s d > 0.3 | MUST HAVE |
| SAE L0 norm is lower for topographic model | At least 10% reduction | SHOULD HAVE |
| Activation patching Δlogit is larger for topo cluster | Statistically significant difference | SHOULD HAVE |
| Brain alignment spatial correlation analysis | Any positive result | NICE TO HAVE |