# Physics-Constrained Coupled PDE Benchmarks for State-to-State Prediction

This note is designed to be dropped into a coding agent such as Claude Code. The goal is to build a **state-to-state** benchmark suite:

\[
x_0 \mapsto \hat{x}_T,
\]

where the model predicts only the final state \(\hat{x}_T\), but evaluation includes both prediction accuracy and **final-state physical constraint violation**.

## Recommended four benchmark systems

| ID | System | State \(x_t\) | Main final-state constraint | Difficulty |
|---|---|---|---|---|
| A | Incompressible Navier--Stokes / Boussinesq | scalar \(c\), velocity \(u=(u_x,u_y)\) | \(\nabla\cdot u_T=0\) | Medium |
| B | Magnetohydrodynamics, MHD | density \(\rho\), velocity \(u\), magnetic field \(B\) | \(\nabla\cdot B_T=0\), \(\rho_T\ge0\) | Hard |
| C | Multiphase / two-phase flow | saturation/volume fractions \(S_i\), pressure \(P\), optionally velocity \(u_i\) | \(0\le S_i\le1\), \(\sum_iS_i=1\), mass balance | Medium-hard |
| D | 2D shallow-water equations | water height \(h\), momentum \(m=(hu,hv)\) | \(h_T\ge0\), water-mass balance | Medium |

The suite is intentionally not only "PDE residual". It targets hard or semi-hard physical admissibility constraints that can be computed from the final prediction.

---

## General data format

Use HDF5 or NPZ.

Recommended tensor layout:

```text
data/
  navier_stokes_2d/
    train.npz  # x0: [N,C,H,W], xT: [N,C,H,W], optional traj: [N,T,C,H,W]
    test.npz
    meta.json
  mhd_2d/
  multiphase_2d/
  shallow_water_2d/
```

State-to-state dataset fields:

```python
{
    "x0": np.ndarray,     # [N, C, H, W]
    "xT": np.ndarray,     # [N, C, H, W]
    "t0": float,
    "tT": float,
    "dt": float,
    "channel_names": list[str],
    "params": dict,
}
```

Optional trajectory fields:

```python
"traj": np.ndarray  # [N, Nt, C, H, W]
```

Even if your model only evaluates \(x_T\), storing the trajectory is useful for solver debugging and future residual metrics.

---

# A. Incompressible Navier--Stokes / Boussinesq

## PDE

Use a scalar-coupled incompressible flow:

\[
\partial_t c + u\cdot\nabla c = \kappa \Delta c,
\]

\[
\partial_t u + (u\cdot\nabla)u
=
-\nabla p + \nu \Delta u + c b,
\]

\[
\nabla\cdot u=0.
\]

Here \(c\) is a transported scalar or buoyancy field, \(u=(u_x,u_y)\), \(p\) is pressure, \(\nu\) is viscosity, \(\kappa\) is scalar diffusivity, and \(b=(0,b_y)\) is the buoyancy direction.

A simpler vorticity form can also be used:

\[
\partial_t \omega + u\cdot\nabla\omega = \nu\Delta\omega + f,
\quad
u=\nabla^\perp\psi,\quad
\Delta\psi=-\omega.
\]

But if the goal is a final-state divergence constraint, store velocity channels explicitly.

## Recommended state

```text
channels = ["c", "u_x", "u_y"]
x_t = [c_t, u_x_t, u_y_t]
```

## Boundary condition

Periodic boundary conditions on \([0,1]^2\).

## Final-state constraints

### 1. Divergence-free velocity

\[
H_{\mathrm{div}}(\hat u_T)=\nabla\cdot \hat u_T=0.
\]

Metric:

\[
\mathrm{DivErr}_u
=
\frac{\|\partial_x\hat u_x+\partial_y\hat u_y\|_2}
{\|\nabla \hat u_T\|_2+\epsilon}.
\]

### 2. Scalar mass conservation, if periodic and no source

\[
\int_\Omega c_T(x)\,dx=\int_\Omega c_0(x)\,dx.
\]

Metric:

\[
\mathrm{MassErr}_c
=
\frac{
\left|\int_\Omega \hat c_T\,dx-\int_\Omega c_0\,dx\right|
}
{
\left|\int_\Omega c_0\,dx\right|+\epsilon
}.
\]

## Simulation suggestion

Use pseudo-spectral time stepping:

1. Sample smooth scalar \(c_0\) and divergence-free \(u_0\).
2. Project velocity to divergence-free space using the Leray projection:
   \[
   \hat u(k) \leftarrow \left(I-\frac{kk^\top}{|k|^2}\right)\hat u(k).
   \]
3. At every time step:
   - advect/diffuse scalar,
   - advect/diffuse velocity plus buoyancy,
   - project velocity to divergence-free space.

## Metrics

```python
rel_l2_c = ||c_pred - c_true||_2 / ||c_true||_2
rel_l2_u = ||u_pred - u_true||_2 / ||u_true||_2
div_err_u = ||div(u_pred)||_2 / (||grad(u_pred)||_2 + eps)
mass_err_c = abs(sum(c_pred) - sum(c0)) / (abs(sum(c0)) + eps)
```

## Sources

- PDEArena / CoupledFlow-style Navier--Stokes stores scalar \(c\) and velocity \((v_x,v_y)\), with incompressibility \(\nabla\cdot v=0\).
- Project and Generate: Divergence-Free Neural Operators for Incompressible Flows enforces incompressibility as a hard intrinsic constraint through Leray projection and divergence-free Gaussian reference measures.

---

# B. Magnetohydrodynamics, MHD

## PDE option 1: incompressible MHD

Use the form:

\[
\partial_t u+u\cdot\nabla u
=
-\nabla \left(p+\frac{|B|^2}{2}\right)/\rho_0
+B\cdot\nabla B
+\nu\Delta u,
\]

\[
\partial_t B+u\cdot\nabla B
=
B\cdot\nabla u+\eta\Delta B,
\]

\[
\nabla\cdot u=0,\qquad \nabla\cdot B=0.
\]

This is a clean 2D benchmark because both velocity and magnetic field have divergence constraints.

## PDE option 2: compressible MHD

For a harder benchmark:

\[
\partial_t \rho+\nabla\cdot(\rho u)=0,
\]

\[
\partial_t(\rho u)+
\nabla\cdot
\left[
\rho uu+
\left(p+\frac{|B|^2}{2}\right)I
-
BB
\right]
=0,
\]

\[
\partial_t B-\nabla\times(u\times B)=0,
\]

\[
\nabla\cdot B=0.
\]

Use an equation of state such as \(p=c_s^2\rho\) or an energy equation.

## Recommended state

Start with incompressible MHD:

```text
channels = ["u_x", "u_y", "B_x", "B_y"]
```

For compressible MHD:

```text
channels = ["rho", "u_x", "u_y", "B_x", "B_y"]
```

## Boundary condition

Periodic boundary conditions.

## Final-state constraints

### 1. Magnetic solenoidal constraint

\[
H_B(\hat B_T)=\nabla\cdot \hat B_T=0.
\]

Metric:

\[
\mathrm{DivErr}_B
=
\frac{
\|\partial_x\hat B_x+\partial_y\hat B_y\|_2
}
{
\|\nabla \hat B_T\|_2+\epsilon
}.
\]

### 2. Velocity incompressibility, if using incompressible MHD

\[
\mathrm{DivErr}_u
=
\frac{
\|\partial_x\hat u_x+\partial_y\hat u_y\|_2
}
{
\|\nabla \hat u_T\|_2+\epsilon
}.
\]

### 3. Density positivity, if using compressible MHD

\[
\rho_T(x)\ge0.
\]

Metric:

\[
\mathrm{NegDensity}
=
\frac{1}{|\Omega|}
\sum_x
\max(0,-\hat\rho_T(x)).
\]

Invalid fraction:

\[
\mathrm{InvalidFrac}_\rho
=
\frac{|\{x:\hat\rho_T(x)<0\}|}{|\Omega|}.
\]

### 4. Mass conservation, if compressible and periodic

\[
\int_\Omega \rho_T\,dx=\int_\Omega \rho_0\,dx.
\]

## Simulation suggestion

For 2D incompressible MHD, a practical route is pseudo-spectral:

1. Initialize stream function \(\psi\) and magnetic potential \(A\).
2. Set
   \[
   u=\nabla^\perp \psi,\qquad B=\nabla^\perp A.
   \]
   This guarantees \(\nabla\cdot u=0\) and \(\nabla\cdot B=0\).
3. Evolve \(u,B\) or evolve vorticity/current formulation.
4. Periodically project \(u\) and \(B\) to divergence-free space for numerical stability.

## Metrics

```python
rel_l2_u = ||u_pred - u_true||_2 / ||u_true||_2
rel_l2_B = ||B_pred - B_true||_2 / ||B_true||_2
div_err_B = ||div(B_pred)||_2 / (||grad(B_pred)||_2 + eps)
div_err_u = ||div(u_pred)||_2 / (||grad(u_pred)||_2 + eps)
neg_density = mean(relu(-rho_pred))  # compressible only
```

## Sources

- NVIDIA PhysicsNeMo MHD PINO example gives incompressible MHD equations with two evolution equations and two constraints, \(\nabla\cdot u=0\), \(\nabla\cdot B=0\), and notes that evolving magnetic vector potential \(A\) ensures \(\nabla\cdot B=0\).
- The Well MHD dataset provides density, velocity, and magnetic field on uniform Cartesian grids with periodic boundary conditions.
- Physics-constrained Orszag--Tang MHD work studies hard constraints such as absence of magnetic monopoles and nonnegative density.

---

# C. Multiphase / two-phase flow

## PDE

For two-phase porous-media flow, use water/oil components \(\alpha,\beta\):

\[
\partial_t M^\alpha
=
-\nabla\cdot F_a^\alpha+q^\alpha,
\]

\[
\partial_t M^\beta
=
-\nabla\cdot F_a^\beta+q^\beta.
\]

Darcy law for phase velocity:

\[
u_p
=
-k(\nabla P_p-\rho_p g)\frac{k_{rp}}{\mu_p}.
\]

A simpler benchmark can use pressure \(P\) and water saturation \(S\):

\[
\phi\partial_t S+\nabla\cdot f_w(S)u_t=q_w,
\]

\[
u_t=-\lambda(S)k\nabla P,
\]

with \(S_o=1-S_w\).

## Recommended state

For a simple two-phase benchmark:

```text
channels = ["P", "S_w"]
```

For explicit two-phase simplex constraint:

```text
channels = ["P", "S_w", "S_o"]
```

## Boundary condition

Use either:
- periodic boundaries for a toy conservation benchmark, or
- injection/production sources with known \(q_w,q_o\).

## Final-state constraints

### 1. Saturation bounds

\[
0\le S_i(x,T)\le 1.
\]

Metric:

\[
\mathrm{BoundErr}_S
=
\frac{1}{K|\Omega|}
\sum_{i=1}^{K}\sum_{x\in\Omega}
\left[
\max(0,-\hat S_i(x))+\max(0,\hat S_i(x)-1)
\right].
\]

Invalid fraction:

\[
\mathrm{InvalidFrac}_S
=
\frac{
|\{(x,i): \hat S_i(x)<0 \ \mathrm{or}\ \hat S_i(x)>1\}|
}
{K|\Omega|}.
\]

### 2. Volume-fraction / saturation simplex

\[
\sum_{i=1}^K S_i(x,T)=1.
\]

Metric:

\[
\mathrm{SimplexErr}
=
\frac{1}{|\Omega|}
\sum_x
\left|\sum_i\hat S_i(x)-1\right|.
\]

For two phases:

\[
\mathrm{SimplexErr}
=
\frac{1}{|\Omega|}
\sum_x
|\hat S_w(x)+\hat S_o(x)-1|.
\]

### 3. Component mass balance

If source/flux data are known:

\[
M_i(T)
=
M_i(0)+\int_0^T Q_i(t)\,dt
-\int_0^T\int_{\partial\Omega}F_i\cdot n\,dSdt.
\]

Final-state metric:

\[
\mathrm{MassErr}_{i}
=
\frac{
\left|
\int_\Omega \phi \hat S_i(x,T)\,dx
-
M_i^{\mathrm{phys}}(T)
\right|
}
{
|M_i^{\mathrm{phys}}(T)|+\epsilon
}.
\]

If no flux/source metadata are stored, use only bounds and simplex error as true physics constraints.

## Simulation suggestion

Start with a simple differentiable finite-volume Buckley--Leverett style two-phase solver:

1. Sample permeability \(k(x)\) from a smooth random field.
2. Sample initial water saturation \(S_w^0\in[0,1]\).
3. Solve pressure equation for \(P\).
4. Compute flux \(u_t=-\lambda(S)k\nabla P\).
5. Update saturation with conservative finite volume.
6. Enforce \(S_w\in[0,1]\) in the numerical solver.

## Metrics

```python
rel_l2_P = ||P_pred - P_true||_2 / ||P_true||_2
rel_l2_S = ||S_pred - S_true||_2 / ||S_true||_2
bound_err_S = mean(relu(-S_pred) + relu(S_pred - 1))
simplex_err = mean(abs(Sw_pred + So_pred - 1))  # if both channels exist
mass_err_S = abs(sum(phi*S_pred) - expected_mass_T) / (abs(expected_mass_T) + eps)
```

## Sources

- The CoupledFlow appendix describes two-phase oil/water multiphase flow with component-wise mass conservation and Darcy phase velocity.
- PCNN for multiphase flows enforces mass conservation, volume-fraction unity, consistency of reduction, and boundedness of order parameters through a correction algorithm.

---

# D. 2D shallow-water equations

## PDE

Use the conservative shallow-water system with height \(\eta\) and momentum \((\eta u,\eta v)\):

\[
\partial_t \eta+\partial_x(\eta u)+\partial_y(\eta v)=0,
\]

\[
\partial_t(\eta u)
+
\partial_x\left(\eta u^2+\frac{1}{2}g\eta^2\right)
+
\partial_y(\eta u v)
=
\nu(u_{xx}+u_{yy}),
\]

\[
\partial_t(\eta v)
+
\partial_x(\eta u v)
+
\partial_y\left(\eta v^2+\frac{1}{2}g\eta^2\right)
=
\nu(v_{xx}+v_{yy}).
\]

A topography version can add bottom \(b(x,y)\) and well-balanced terms.

## Recommended state

```text
channels = ["eta", "m_x", "m_y"]
m_x = eta * u
m_y = eta * v
```

Storing momentum instead of velocity avoids division by small height.

## Boundary condition

Periodic boundary conditions for the simplest benchmark.

## Final-state constraints

### 1. Water-height positivity

\[
\eta(x,T)\ge0.
\]

Metric:

\[
\mathrm{NegHeight}
=
\frac{1}{|\Omega|}
\sum_x
\max(0,-\hat\eta_T(x)).
\]

Invalid fraction:

\[
\mathrm{InvalidFrac}_\eta
=
\frac{
|\{x:\hat\eta_T(x)<0\}|
}
{|\Omega|}.
\]

### 2. Water-mass conservation

For periodic/no-source domains:

\[
\int_\Omega \eta_T(x)\,dx=\int_\Omega \eta_0(x)\,dx.
\]

Metric:

\[
\mathrm{MassErr}_\eta
=
\frac{
\left|\int_\Omega \hat\eta_T\,dx-\int_\Omega \eta_0\,dx\right|
}
{
\left|\int_\Omega \eta_0\,dx\right|+\epsilon
}.
\]

### 3. Optional lake-at-rest well-balanced error

Only use this if data include lake-at-rest or topography equilibrium cases.

\[
u=0,\qquad \eta+b=\mathrm{constant}.
\]

Metric:

\[
\mathrm{WBErr}
=
\|\hat u_T\|_2+\mathrm{Var}(\hat\eta_T+b).
\]

## Simulation suggestion

Start with periodic finite-volume or pseudo-spectral smooth shallow water:

1. Sample smooth positive height:
   \[
   \eta_0=1+0.1\,\mathrm{GRF}(x,y),
   \]
   clipped to \(\eta_0>0.1\).
2. Set initial velocity zero or sample smooth small velocity.
3. Use a stable finite-volume method with CFL control.
4. Store \(\eta, m_x, m_y\) at \(t=0\) and \(t=T\).

## Metrics

```python
rel_l2_eta = ||eta_pred - eta_true||_2 / ||eta_true||_2
rel_l2_m = ||m_pred - m_true||_2 / ||m_true||_2
neg_height = mean(relu(-eta_pred))
invalid_frac_eta = mean(eta_pred < 0)
mass_err_eta = abs(sum(eta_pred) - sum(eta0)) / (abs(sum(eta0)) + eps)
```

## Sources

- PDEBench includes the 2D shallow-water equation dataset and stores arrays in HDF5 with dimensions `[batch, time, x1, ..., xd, variables]`.
- NVIDIA PhysicsNeMo provides a PINO example for nonlinear shallow-water equations with three coupled equations and equation-residual physics constraints.
- Classical shallow-water numerics emphasizes conservation and positivity of water height.

---

# Common final-state evaluation

For each dataset, evaluate both accuracy and physical admissibility.

## Prediction metrics

Use channel-wise relative L2:

\[
\mathrm{RelL2}_c=
\frac{\|\hat x_T^{(c)}-x_T^{(c)}\|_2}
{\|x_T^{(c)}\|_2+\epsilon}.
\]

Average across channels:

\[
\mathrm{RelL2}_{avg}=
\frac{1}{C}\sum_{c=1}^{C}\mathrm{RelL2}_c.
\]

Also report global relative L2:

\[
\mathrm{RelL2}_{global}=
\frac{\|\hat x_T-x_T\|_2}{\|x_T\|_2+\epsilon}.
\]

## Constraint metrics

Recommended main metric per dataset:

| System | Main constraint metric |
|---|---|
| Navier--Stokes | \(\mathrm{DivErr}_u\) |
| MHD | \(\mathrm{DivErr}_B\), optionally \(\mathrm{DivErr}_u\) |
| Multiphase | \(\mathrm{BoundErr}_S+\mathrm{SimplexErr}\) |
| Shallow water | \(\mathrm{NegHeight}+\mathrm{MassErr}_\eta\) |

## Report table template

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{l|cccc}
\toprule
Method & Rel. $L^2_{\rm avg}\downarrow$ & Main constraint $\downarrow$ & Invalid frac. $\downarrow$ & Time $\downarrow$\\
\midrule
FNO & -- & -- & -- & -- \\
PINO & -- & -- & -- & -- \\
CViT & -- & -- & -- & -- \\
MeanFlow / CoupledFlow & -- & -- & -- & -- \\
Ours & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\caption{Final-state prediction and physical admissibility.}
\end{table}
```

---

# Implementation scaffold for Claude Code

Ask Claude Code to create this repo:

```text
physics_constrained_pde_bench/
  README.md
  pyproject.toml
  configs/
    navier_stokes_2d.yaml
    mhd_2d.yaml
    multiphase_2d.yaml
    shallow_water_2d.yaml
  src/
    common/
      grids.py
      spectral.py
      finite_difference.py
      metrics.py
      io.py
      random_fields.py
    solvers/
      navier_stokes_2d.py
      mhd_2d.py
      multiphase_2d.py
      shallow_water_2d.py
    generate_dataset.py
    evaluate_final_state.py
  tests/
    test_constraints.py
    test_shapes.py
```

## Required common utilities

### `src/common/spectral.py`

Implement:

```python
fft2, ifft2
spectral_grad_x(field, Lx)
spectral_grad_y(field, Ly)
laplacian(field, Lx, Ly)
divergence_2d(vx, vy, Lx, Ly)
leray_project(vx, vy, Lx, Ly)
curl_potential_to_vector(A, Lx, Ly)  # B = (d_y A, -d_x A)
```

### `src/common/metrics.py`

Implement:

```python
relative_l2(pred, true, eps=1e-12)
channelwise_relative_l2(pred, true, channel_names)
divergence_error(vx, vy, Lx, Ly)
mass_error(pred_scalar_T, scalar_0)
negative_violation(field)
bounds_error(field, lo=0.0, hi=1.0)
simplex_error(fields, axis=0)
invalid_fraction_bounds(field, lo=0.0, hi=1.0)
```

### `src/generate_dataset.py`

Command-line interface:

```bash
python -m src.generate_dataset --config configs/navier_stokes_2d.yaml
python -m src.generate_dataset --config configs/mhd_2d.yaml
python -m src.generate_dataset --config configs/multiphase_2d.yaml
python -m src.generate_dataset --config configs/shallow_water_2d.yaml
```

### `src/evaluate_final_state.py`

Inputs:

```bash
python -m src.evaluate_final_state \
  --dataset data/navier_stokes_2d/test.npz \
  --pred predictions/navier_stokes_2d_ours.npz \
  --system navier_stokes_2d
```

Output:

```json
{
  "rel_l2_global": 0.0,
  "rel_l2_channelwise": {...},
  "main_constraint": 0.0,
  "invalid_fraction": 0.0,
  "mass_error": 0.0
}
```

---

# Starter YAML configs

## `configs/navier_stokes_2d.yaml`

```yaml
system: navier_stokes_2d
N_train: 1024
N_test: 256
grid: [64, 64]
domain: [1.0, 1.0]
dt: 0.001
T: 0.1
nu: 0.001
kappa: 0.0005
buoyancy: [0.0, 1.0]
bc: periodic
channels: ["c", "u_x", "u_y"]
store_trajectory: false
seed: 0
```

## `configs/mhd_2d.yaml`

```yaml
system: mhd_2d_incompressible
N_train: 1024
N_test: 256
grid: [64, 64]
domain: [1.0, 1.0]
dt: 0.0005
T: 0.05
nu: 0.001
eta: 0.001
bc: periodic
channels: ["u_x", "u_y", "B_x", "B_y"]
initialization: stream_function_and_vector_potential
store_trajectory: false
seed: 1
```

## `configs/multiphase_2d.yaml`

```yaml
system: multiphase_2d_toy
N_train: 1024
N_test: 256
grid: [64, 64]
domain: [1.0, 1.0]
dt: 0.001
T: 0.1
porosity: 1.0
bc: periodic
channels: ["P", "S_w", "S_o"]
store_trajectory: false
seed: 2
```

## `configs/shallow_water_2d.yaml`

```yaml
system: shallow_water_2d
N_train: 1024
N_test: 256
grid: [64, 64]
domain: [1.0, 1.0]
dt: 0.0005
T: 0.05
g: 1.0
nu: 0.002
bc: periodic
channels: ["eta", "m_x", "m_y"]
store_trajectory: false
seed: 3
```

---

# Minimum tests before trusting the generated datasets

## Navier--Stokes

```python
assert divergence_error(u_x_T, u_y_T) < 1e-5
```

## MHD

```python
assert divergence_error(B_x_T, B_y_T) < 1e-5
assert divergence_error(u_x_T, u_y_T) < 1e-5  # incompressible MHD
```

## Multiphase

```python
assert np.all(S_w >= -1e-6)
assert np.all(S_w <= 1 + 1e-6)
assert np.max(np.abs(S_w + S_o - 1)) < 1e-5
```

## Shallow water

```python
assert np.all(eta_T >= -1e-6)
assert mass_error(eta_T, eta_0) < 1e-4
```

---

# Reference list to include in the paper

1. PDEArena / CoupledFlow-style Navier--Stokes for scalar + velocity incompressible state-to-state prediction.
2. Li et al., Project and Generate: Divergence-Free Neural Operators for Incompressible Flows, 2026.
3. NVIDIA PhysicsNeMo MHD PINO example for incompressible MHD equations and constraints.
4. The Well MHD dataset documentation for large-scale density/velocity/magnetic-field simulation data.
5. Zheng et al., PCNN: A Physics-Constrained Neural Network for Multiphase Flows, 2021/2022.
6. Takamoto et al., PDEBench, including 2D shallow-water equation data.
7. NVIDIA PhysicsNeMo nonlinear shallow-water PINO example for physics-informed equation residual training.
