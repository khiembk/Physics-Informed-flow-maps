# CLAUDE.md — Physics-Informed Flow Maps (Research Context)

This file captures the research goals, design decisions, and implementation roadmap for this project, for use by Claude Code in future sessions.

---

## Project Goal

Build a **physics-informed few-step flow model** for state-to-state PDE prediction:

```
x_0 → x_T
```

The model must be both accurate (low relative L2) and **physically admissible** (satisfying hard constraints at the predicted final state).

The codebase extends the Boffi et al. flow map self-distillation framework (`py/common/losses.py`, `py/common/flow_map.py`) with physics-informed training.

---

## Two-Phase Training

### Phase 1 — Physics-Informed Interpolant Path

Learn a corrected interpolation path between `x_0` and `x_1`:

```
x_t = ψ_φ(t, x_0, x_1) = (1-t)x_0 + t*x_1 + α(t) * φ(t, x_0, x_1)
```

where `α(t) = t(1-t)` enforces `ψ_φ(0,·) = x_0` and `ψ_φ(1,·) = x_1`.

The path velocity is:
```
∂_t ψ_φ = (x_1 - x_0) + α̇(t) φ + α(t) ∂_t φ
```

**Loss:**
```
L_path(φ) = L_phy(φ) + λ_sm * L_sm(φ)
```

- `L_phy`: weighted physics residual `E[w(t) * ||R(x_t)||²]`, weight `w(t) = w_0 + t*α` increases toward t=1
- `L_sm`: spatial smoothness of the marginal velocity field `u_φ(x,t)`

The correction network `φ` is a learned model (same MLP/UNet architecture as the flow map). **φ is frozen after Phase 1.**

---

### Phase 2 — Few-Step Flow Model (LSD Loss)

Train a flow map `v_θ(x_s, s, t)` on the frozen physics-informed path.

**Parameterization:**
```
X_{s,t}(x_s) = x_s + (t-s) * v_θ(x_s, s, t)
```

**Training uses LSD (Lagrangian Self-Distillation)**, which enforces the tangent condition:
```
∂_t X(s,t,x_s) = b(t, X(s,t,x_s))
```

Two sources of the teacher velocity `b`:

| Batch portion | Source of `b` | Loss term |
|---|---|---|
| Diagonal (`s=t`) | Frozen Phase 1 path: `∂_t ψ_φ(t, x_0, x_1)` | Flow matching |
| Off-diagonal (`s<t`) | EMA of current model: `v_θ_ema(X_{s,t}(x_s), t, t)` | LSD self-distillation |

**Why LSD over PSD/ESD for Phase 2:**
- PSD requires two-step composition — expensive for high-dim PDE states
- ESD requires spatial Jacobian through the network — numerically fragile
- LSD matches time derivative directly — stable, efficient, maps cleanly to the tangent condition

**Self-consistency loss** (from paper, same as LSD off-diagonal):
```
L_consist = E_{s,t}[|| v_θ(x_s,s,t) + ∂_t v_θ(x_s,s,t) - sg(v_θ(x_t,t,t)) ||²]
```

The existing `lsd_term` in `py/common/losses.py:154` implements this directly. For Phase 2, swap `teacher_params` to evaluate `b` from the frozen Phase 1 model on the diagonal portion.

---

## Benchmark Suite — 4 PDE Systems

All systems: state-to-state prediction on `[0,1]²`, periodic BC, 64×64 grid.
Data format: `NPZ`, shape `[N, C, H, W]`.

### A. Incompressible Navier-Stokes / Boussinesq

**State:** `[c, u_x, u_y]` — scalar + velocity

**PDE:**
```
∂_t c + u·∇c = κ Δc
∂_t u + (u·∇)u = -∇p + ν Δu + c*b
∇·u = 0
```

**Final-state constraints:**
- `∇·u_T = 0` → metric: `DivErr_u = ||div(u_pred)||_2 / (||∇u_pred||_2 + ε)`
- Scalar mass: `|∫c_T - ∫c_0| / (|∫c_0| + ε)`

**Constraint type:** Differential (pointwise in Fourier space)

**Config:** `configs/navier_stokes_2d.yaml`, `ν=0.001`, `κ=0.0005`, `T=0.1`

---

### B. Incompressible MHD

**State:** `[u_x, u_y, B_x, B_y]` — velocity + magnetic field

**PDE:**
```
∂_t u + u·∇u = -∇(p + |B|²/2) + B·∇B + ν Δu
∂_t B + u·∇B = B·∇u + η ΔB
∇·u = 0,  ∇·B = 0
```

**Final-state constraints (two):**
- `∇·u_T = 0` → `DivErr_u`
- `∇·B_T = 0` → `DivErr_B`

**Constraint type:** Two simultaneous differential constraints. Harder than NS.

**Init trick:** Initialize via stream function ψ and vector potential A so `u = ∇⊥ψ`, `B = ∇⊥A` — guarantees both divergence-free at t=0.

**Config:** `configs/mhd_2d.yaml`, `ν=0.001`, `η=0.001`, `T=0.05`

---

### C. Multiphase / Two-Phase Flow

**State:** `[P, S_w, S_o]` — pressure + water/oil saturation

**PDE (Darcy-based):**
```
φ ∂_t S + ∇·f_w(S) u_t = q_w
u_t = -λ(S) k ∇P
S_o = 1 - S_w
```

**Final-state constraints:**
- Bounds: `0 ≤ S_i ≤ 1` → `BoundErr_S = mean(relu(-S) + relu(S-1))`
- Simplex: `S_w + S_o = 1` → `SimplexErr = mean(|S_w + S_o - 1|)`

**Constraint type:** Algebraic simplex — no spatial derivatives. Pointwise per grid cell.

**Note:** Since `S_o = 1 - S_w` by definition, the model can predict both and be penalized for simplex violation, or predict one and hard-project. Store both channels for the constraint metric.

**Config:** `configs/multiphase_2d.yaml`, `T=0.1`

---

### D. 2D Shallow Water Equations

**State:** `[η, m_x, m_y]` — water height + momentum (`m = η*v`)

**PDE (conservative form):**
```
∂_t η + ∂_x m_x + ∂_y m_y = 0
∂_t m_x + ∂_x(m_x²/η + g η²/2) + ∂_y(m_x m_y/η) = ν Δu_x
∂_t m_y + ∂_x(m_x m_y/η) + ∂_y(m_y²/η + g η²/2) = ν Δu_y
```

**Final-state constraints:**
- Positivity: `η_T ≥ 0` → `NegHeight = mean(relu(-η_pred))`
- Mass conservation: `|∫η_T - ∫η_0| / (|∫η_0| + ε)`

**Constraint type:** Pointwise inequality + global integral conservation. Mixed local/global.

**Config:** `configs/shallow_water_2d.yaml`, `g=1.0`, `ν=0.002`, `T=0.05`

**Init:** `η_0 = 1 + 0.1 * GRF(x,y)`, clipped to `η_0 > 0.1`

---

## Constraint Type Summary

| System | Constraint type | Spatial operator | Difficulty |
|--------|----------------|-----------------|------------|
| NS/Boussinesq | Differential (∇·u=0) | Spectral divergence | Medium |
| MHD | Two differential (∇·u=0, ∇·B=0) | Spectral divergence × 2 | Hard |
| Multiphase | Algebraic simplex (0≤S≤1, ΣS=1) | None — pointwise algebraic | Medium-hard |
| Shallow water | Positivity (η≥0) + global mass | Pointwise + integral | Medium |

---

## Evaluation Metrics

For each system, report:

```python
rel_l2_global   = ||x_pred - x_true||_2 / ||x_true||_2
rel_l2_avg      = mean over channels of (||c_pred - c_true|| / ||c_true||)
main_constraint = system-specific (DivErr / BoundErr+SimplexErr / NegHeight+MassErr)
invalid_frac    = fraction of grid cells violating hard constraints
```

Comparison baselines from the claude_doc report table: FNO, PINO, CViT, MeanFlow/CoupledFlow.

---

## Model Size Budget & Recommended Configs

Parameter counts below are exact (analytically derived from the MPConv/Dense weight formulas — no bias in MPConv, bias in Dense).

### Grid-based PDEs (64×64, C channels)

**Flow map v_θ: EDM2 UNet — ~12.28M params (C=3), ~12.29M (C=4)**

```python
config.network.network_type = "edm2"
config.network.img_resolution = 64
config.network.img_channels = C          # 3 for NS/SW/Multiphase, 4 for MHD
config.network.rescale = 0.5             # sigma_data
config.network.use_weight = False
config.network.logvar_channels = 128
config.network.use_bfloat16 = False
config.network.label_dim = 0
config.network.unet_kwargs = ml_collections.ConfigDict({
    "model_channels": 64,
    "channel_mult": (1, 2, 4),           # channels: [64, 128, 256]
    "num_blocks": 1,                     # 1 residual block per resolution
    "attn_resolutions": (16,),           # self-attention at 16×16 (bottom scale)
    "channel_mult_noise": None,
    "channel_mult_emb": None,
    "block_kwargs": {"dropout": 0.0},
})
```

| System | C | v_θ params |
|--------|---|-----------|
| NS/Boussinesq | 3 | 12,283,984 |
| MHD | 4 | 12,285,136 |
| Multiphase | 3 | 12,283,984 |
| Shallow water | 3 | 12,283,984 |

**Path encoding φ: tiny EDM2 UNet — ~0.72M params (all systems)**

φ takes (x_0, x_1) concatenated → 2*C input channels, outputs C channels.
Requires a minor architecture modification: use `img_channels = 2*C` for the UNet input
but override `out_conv` to MPConv(cout=64, C, 3×3). See implementation note below.

```python
config.phi_network = ml_collections.ConfigDict()
config.phi_network.network_type = "edm2"
config.phi_network.img_resolution = 64
config.phi_network.img_channels = 2 * C  # concatenated x_0, x_1
config.phi_network.output_channels = C   # actual output channels (needs arch mod)
config.phi_network.rescale = 0.5
config.phi_network.use_weight = False
config.phi_network.logvar_channels = 128
config.phi_network.use_bfloat16 = False
config.phi_network.label_dim = 0
config.phi_network.unet_kwargs = ml_collections.ConfigDict({
    "model_channels": 16,
    "channel_mult": (1, 2, 4),           # channels: [16, 32, 64]
    "num_blocks": 1,
    "attn_resolutions": (),              # no attention — keeps it small
    "channel_mult_noise": None,
    "channel_mult_emb": None,
    "block_kwargs": {"dropout": 0.0},
})
```

**Architecture note for φ:** The existing `EDM2FlowMapUNet.out_conv` is `MPConv(cout, img_channels, (3,3))`.
Since img_channels=2*C but desired output is C, add a `PathEncodingUNet` wrapper (or subclass) that
replaces `out_conv` with `MPConv(cout, C_output, (3,3))`. The encoder still processes 2*C+1 input channels
(the +1 is the constant channel appended in the forward pass). Only the final output projection changes.

### Low-dimensional / toy PDEs (checker, MLP)

**Flow map v_θ: MLP — 1.05M params (d=2)**

```python
# Existing checker config — unchanged
config.network.network_type = "mlp"
config.network.n_hidden = 4
config.network.n_neurons = 512
config.network.output_dim = d
```

**Path encoding φ: smaller MLP — 0.59M params (d=2)**

φ input dimension = 2*d + 1 (t, x_0, x_1). Dense layers include bias.

```python
config.phi_network.network_type = "mlp"
config.phi_network.n_hidden = 4
config.phi_network.n_neurons = 384
config.phi_network.output_dim = d
# input_dim must be set to 2*d+1 in the PathEncodingMLP class
```

### Parameter count verification

```python
# Run after initialization to verify
from jax.flatten_util import ravel_pytree
print(f"v_theta params: {ravel_pytree(params)[0].size:,}")
print(f"phi params:     {ravel_pytree(phi_params)[0].size:,}")
```

---

## Implementation Roadmap

1. **Phase 1:** Implement the correction network `φ` and `L_path` loss
   - Add `physics_path_term` to `py/common/losses.py`
   - Add `R(x_t)` (PDE residual) per system in `py/common/physics_residuals.py`
   - Add `φ` network alongside or as a wrapper of `FlowMap`

2. **Phase 2:** Wire LSD loss to use frozen Phase 1 `φ` as teacher on diagonal terms
   - Modify `setup_loss` to accept a frozen `phi_params` argument
   - Diagonal `b` = `∂_t ψ_φ(t, x_0, x_1)` from Phase 1 model
   - Off-diagonal `b` = EMA of current `v_θ` (existing LSD logic unchanged)

3. **Datasets:** Build 4 PDE simulators in `physics_constrained_pde_bench/`
   - Spectral utilities: `src/common/spectral.py`
   - Constraint metrics: `src/common/metrics.py`
   - Solvers: `src/solvers/{navier_stokes,mhd,multiphase,shallow_water}_2d.py`
   - CLI: `src/generate_dataset.py --config configs/<system>.yaml`

4. **Evaluation:** `src/evaluate_final_state.py` — loads predictions + ground truth, computes all metrics, outputs JSON

---

## File Map (current)

```
py/
  common/
    losses.py          — LSD/PSD/ESD loss implementations (extend for Phase 2)
    flow_map.py        — FlowMap wrapper; partial_t/partial_s via jvp
    interpolant.py     — Stochastic interpolants (extend for physics-informed path)
    datasets.py        — Dataset loaders (add PDE loaders here)
    edm2_net.py        — EDM2 UNet architecture
    network_utils.py   — Network setup
    loss_args.py       — Diagonal/off-diagonal batch splitting
  configs/
    checker.py         — Template config (copy for each PDE)
  launchers/
    learn.py           — Main training loop
claude_doc/
  physics_constrained_coupled_pde_benchmarks.md  — Benchmark spec
  CLAUDE.md           — This file
papers/our/
  main.tex            — Draft paper
```
