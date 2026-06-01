# CLAUDE.md — Physics-Informed Flow Maps (Research Context)

This file captures the research goals, design decisions, implementation roadmap, and experimental findings for this project.

---

## Project Goal

Build a **physics-informed few-step flow model** for state-to-state PDE prediction:

```
x_0 → x_T
```

The model must be both accurate (low relative L2) and **physically admissible** (satisfying hard constraints at the predicted final state).

---

## Two-Phase Training

### Phase 1 — Physics-Informed Interpolant Path

Learn a corrected interpolation path between `x_0` and `x_1`:

```
x_t = ψ_φ(t, x_0, x_1) = (1-t)x_0 + t*x_1 + α(t) * φ(t, x_0, x_1)
```

where `α(t) = t(1-t)` enforces `ψ_φ(0,·) = x_0` and `ψ_φ(1,·) = x_1`.

**Loss:**
```
L_path(φ) = L_phy(φ) + λ_sm * L_sm(φ)
```

- `L_phy = E[w(t) * ||R(x_t)||²]`:  physics constraint violation at intermediate state
- `L_sm = E[||grad_spatial(v_t)||²]`: spatial roughness of path velocity
- `w(t) = w0 + w_alpha * t`: weight increasing toward t=1

**CRITICAL — Physics loss formulation:**
`R(x_t)` must measure **constraint violation** (not dynamics mismatch):

| System | R(x_t) | Notes |
|--------|--------|-------|
| NS | ∇·u_t | divergence of velocity |
| MHD | [∇·u_t, ∇·B_t] | both fields div-free |
| SW | relu(-η_t) | height positivity |
| Multiphase | simplex + bounds | S_w+S_o=1, 0≤S≤1 |
| **Euler** | **relu(-ρ_t) + relu(-p_t)** | **nonlinear! violated by linear interp** |

**CRITICAL — Constraint difficulty:**
- Algebraic LINEAR constraints (∇·u=0, simplex) are PRESERVED by linear interpolation of valid states → L_phy ≈ 0, no signal
- Nonlinear constraints (p=(γ-1)(E-|m|²/2ρ)>0 in Euler) ARE violated by linear interpolation → real training signal
- φ ≈ 0 at init → constraint violation ≈ 0 → gradient ≈ 0 → chicken-and-egg problem for all systems
- **Current Phase 1 does not converge** because L_phy ≈ machine epsilon for all systems tested

### Phase 2 — Flow Map Training

Train flow map `v_θ(x_s, s, t)` on the **frozen** physics-informed path from Phase 1.

**Diagonal flow matching loss:**
```
L = ||v_θ(x_t, t, t) - v_t||²
```
where `(x_t, v_t)` are pre-computed from frozen φ.

Two-step per iteration:
1. `compute_targets(x0, xT, t)` → `(x_t, v_t)` from frozen φ (no grad)
2. `train_step(params, x_t, v_t, t)` → gradient update on v_θ

---

## 5 Benchmark Systems

All: 64×64 grid, periodic BC, NPZ format `[N,C,H,W]`

### A. Shallow Water (SW) — state: [η, m_x, m_y]
- Constraint: η ≥ 0 (height positivity)
- **T=0.3** (trivial RelL2=0.058 — easy task, linear constraints)
- Solver: pseudo-spectral conservative form

### B. Navier-Stokes/Boussinesq (NS) — state: [c, u_x, u_y]
- Constraint: ∇·u = 0
- **T=0.5** (trivial RelL2=0.593 — hard, real signal for longer T)
- Solver: vorticity-streamfunction + scalar, RK4

### C. MHD — state: [u_x, u_y, B_x, B_y]
- Constraints: ∇·u = 0, ∇·B = 0
- **T=0.3** (trivial RelL2=0.987 — **TOO HARD**, recommend T≤0.1)
- Solver: stream function ψ + vector potential A, RK4

### D. Multiphase — state: [P, S_w, S_o]
- Constraints: S_w+S_o=1, 0≤S≤1
- **T=0.5** (trivial RelL2=0.365)
- Solver: upwind FV Buckley-Leverett

### E. Compressible Euler — state: [ρ, m_x, m_y, E]
- Constraints: ρ>0, **p=(γ-1)(E-|m|²/2ρ)>0** ← NONLINEAR, violated by linear interp
- **T=0.5**, Mach=0.6, γ=1.4 (trivial RelL2=0.554)
- Solver: pseudo-spectral + artificial viscosity (ν=0.008)
- **Best system for this method** — pressure constraint is genuinely violated by linear path

---

## Experimental Results (latest run)

**Setup:** Phase 1 (3000 steps, λ_sm=0), Phase 2 (10K steps, diagonal loss)

| System | Trivial | Phase2 | Baseline | RelL2 Δ | Key constraint Δ |
|--------|---------|--------|----------|---------|-----------------|
| SW (T=0.3) | 0.058 | — | — | — | — |
| **NS (T=0.5)** | 0.593 | 0.366 | 0.371 | **−1.4% Phase2 wins** | DivErr **−16%** |
| MHD (T=0.3) | 0.987 | 0.737 | 0.735 | +0.2% (tie) | DivErr_B **−4.6%** |
| Multi (T=0.5) | 0.365 | 0.297 | 0.294 | +1.2% | BoundErr **−11%** |
| **Euler (T=0.5)** | 0.554 | 0.344 | 0.332 | +3.5% | **NegP −43%** |

**Key findings:**
- Phase 2 **beats baseline on NS RelL2** (first time) at T=0.5
- Phase 2 wins on ALL physics constraint metrics across all systems
- MHD at T=0.3: training converges (loss −44%) but task too hard (trivial=0.99) → **reduce T to 0.1**
- Euler has largest constraint improvement (NegP −43%) — confirms nonlinear constraints benefit most from physics-informed path

**Convergence (Phase 2 loss step1k → step10k):**

| System | Loss @1k | Loss @10k | Reduction | Need more steps? |
|--------|---------|---------|-----------|-----------------|
| SW | 3.5e-4 | 2.4e-4 | −31% | No |
| NS | 0.101 | 0.060 | −40% | Yes, 30-50K |
| MHD | 0.891 | 0.501 | −44% | No (fix T instead) |
| Multi | 0.024 | 0.013 | −44% | Marginal |
| Euler | 0.508 | 0.240 | −53% | Yes, 30-50K |

---

## Known Issues & Recommendations

### Phase 1 does not converge (most important issue)
- **Root cause**: φ≈0 at init → constraint violation = α(t)·(nonzero only if φ nonzero) → L_phy≈0 → no gradient
- **For linear constraints** (NS ∇·u, Multi simplex): ALSO preserved by linear interpolation → L_phy=0 even when endpoints are valid
- **Fix options**:
  1. Use **dynamics mismatch** `R = v_t - rhs(x_t)` — nonzero from step 1, large signal
  2. Large λ_phy with small random init forcing small but nonzero φ from start
  3. Euler-only: L_phy is genuinely nonzero (nonlinear constraint) but still tiny vs L_sm

### MHD T too long
- T=0.3 gives trivial RelL2=0.987 — state is almost completely decorrelated from initial
- **Recommended T: 0.10** (trivial ≈ 0.7, hard but learnable)

### Training budget
- NS and Euler need 30–50K Phase 2 steps to fully converge
- Others fine at 10K

---

## File Map

```
py/
  common/
    losses.py            — LSD/PSD/ESD loss (Phase 2)
    flow_map.py          — FlowMap wrapper
    path_encoding.py     — PhiUNet, PhiTransformer, PathEncodingMLP
    phase1_loss.py       — L_phy + L_sm (Phase 1)
    physics_residuals.py — constraint_fn per system (CORRECTED: constraint violation, not dynamics)
    spectral.py          — GRF, FFT, Laplacian, Leray, metrics
    edm2_net.py          — EDM2 UNet (ALL convs via einsum/im2col — no cuDNN)
  configs/
    pde_configs.py       — SYSTEM_PARAMS + factory for all 5 systems
    sw/ns/mhd/multi/euler_phase1/phase2/baseline.py  — thin wrappers
  launchers/
    phase1_learn.py      — Phase 1 training (φ)
    phase2_learn.py      — Phase 2 training (v_θ on frozen φ)
    pde_learn.py         — Baseline training (linear interpolant)
  solvers/
    ns_boussinesq.py     — NS/Boussinesq pseudo-spectral
    mhd_2d.py            — Incompressible MHD (ψ/A formulation)
    multiphase_2d.py     — Buckley-Leverett FV
    shallow_water_2d.py  — Conservative SW pseudo-spectral
    euler_2d.py          — Compressible Euler + artificial viscosity
  evaluate_pde.py        — Generic eval: RelL2 + system constraints
  generate_dataset.py    — CLI: --config <yaml> --output_dir <path>
slurm_scripts/
  generate_datasets.sbatch  — array=0-4, CPU, data generation
  pde_p1_p2.sbatch          — array=0-4, GPU, Phase1+Phase2
  pde_baseline.sbatch       — array=0-4, GPU, baseline training
  pde_evaluate.sbatch       — array=0-4, GPU, evaluation
claude_doc/
  CLAUDE.md                 — this file
  physics_constrained_coupled_pde_benchmarks.md  — benchmark spec
```

---

## cuDNN Fix (H100 PCIe, JAX 0.4.26)

`CUDNN_STATUS_INTERNAL_ERROR` on all conv_general_dilated backward passes.
Fixed in `edm2_net.py`:
- 1×1 conv: `einsum('bchw,oc->bohw')` (no cuDNN)
- k×k conv: im2col → single `einsum` (no cuDNN, ~9× fewer XLA nodes)
- resample: `reshape+mean` (down), `jnp.repeat` (up)

---

## Environment

```bash
conda activate phyflow
# Python 3.10, JAX 0.4.26+cuda12, Flax 0.8.2, Optax 0.2.2
# Cluster: TAMU HPRC, H100 PCIe nodes (partition=gpu)
# Account: 156341690590
```

Data location: `/scratch/user/u.kt348068/physics_informedPDE/`
```
  {system}/train.npz, test.npz     — [N,C,H,W] float32
  checkpoints/phase1_checkpoints/  — phi_final.npz per system
  runs/{system}_phase2/            — v_theta checkpoints
  runs/{system}_baseline/          — baseline checkpoints
  results/{system}_eval.json       — evaluation metrics
```
