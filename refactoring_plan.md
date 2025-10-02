# Refactoring Plan for Public Release

## Executive Summary

This document outlines a comprehensive refactoring plan to prepare the Lagrangian Map Matching codebase for public release. The goal is to remove experimental features not used in final paper experiments, simplify the codebase, and improve code quality while maintaining full reproducibility of paper results.

## Analysis of Config Files

Analyzed all config files in `py/configs/`:
- `cifar10.py`: 4 experiments (LSD, PSD uniform, PSD shortcut, ESD full convex)
- `celeba64.py`: 3 experiments → Will update to 4 (LSD, PSD uniform, PSD shortcut, ESD full convex)
- `afhq64.py`: 3 experiments → Will update to 4 (LSD, PSD uniform, PSD shortcut, ESD full convex)
- `checker.py`: 4 experiments (LSD, PSD uniform, PSD shortcut, ESD)

**Final Config Structure** (4 experiments each):
1. **LSD** - Lagrangian Self-Distillation with convex stopgrad
2. **PSD uniform** - Progressive Self-Distillation with uniform u sampling
3. **PSD shortcut** - Progressive Self-Distillation with shortcut (u=0.5*(s+t))
4. **ESD full convex** - Eulerian Self-Distillation with full stopgrad

### Features ACTUALLY Used in Final Experiments

**Loss Types (KEEP):**
- `psd` (Progressive Self-Distillation) - **RENAMED from pfmm**
- `lsd` (Lagrangian Self-Distillation)
- `esd` (Eulerian Self-Distillation)

**PSD Types (KEEP):**
- `uniform` - uniform interpolation between s and t
- `midpoint` - midpoint interpolation (u = 0.5*(s+t))
- `None` - for LSD/ESD which don't use PSD

**Stopgrad Types (KEEP):**
- `convex` - stops gradients on teacher evaluations (USED IN ALL FINAL CONFIGS)
- `none` - full gradients (only for special ablations)
- `full` - for ESD variant

**Time Sampling Strategies (KEEP):**
- `uniform_upper_triangle` - samples from upper triangle (s <= t) (ONLY SAMPLING STRATEGY NEEDED)
- ~~`uniform_full_plane`~~ - **REMOVED** - the one config using this will be updated to use uniform_upper_triangle

**Core Features (KEEP):**
- Flow map training (`train_velocity=False` in all configs)
- Single weight functions (`use_dual_weight_functions=False` in all configs)
- Time parameterization: `s_dt` (used in all configs)
- EMA factors: `[0.999, 0.9999]` (standard across all)
- No EMA teacher (`use_ema_teacher=False` in all configs)

---

## Features to Remove

### 1. Velocity Field Training
**Rationale**: All final experiments use `train_velocity=False` - training flow maps, not velocity fields.

**Config Parameters to Remove:**
- `config.training.train_velocity` - always False in final configs
- `config.network.is_velocity` - derived from train_velocity
- `config.network.load_from_velocity` - for loading velocity checkpoints

**Code to Remove:**
- `py/common/velocity.py` - entire file (168 lines)
- Velocity initialization in `state_utils.py:setup_training_state()` (lines 259-263)
- `load_velocity_checkpoint()` in `state_utils.py` (lines 115-153)
- `convert_velocity_to_flow()` and `convert_flow_to_velocity()` in `state_utils.py` (lines 59-112)
- Velocity-related imports and conditional logic throughout

**Files to Modify:**
- `py/common/state_utils.py`: Remove velocity conversion functions, velocity initialization
- `py/launchers/learn.py`: Remove velocity-specific compilation logic (line 84)
- `py/common/flow_map.py`: No changes needed (already flow-map centric)
- Config files: Remove `train_velocity`, `is_velocity`, `load_from_velocity` parameters

### 2. Unused Stopgrad Types
**Rationale**: Only `convex`, `none`, and `full` are used in final experiments. All other variants are experimental.

**Stopgrad Types to REMOVE:**
- `lsd_mean_flow` (lines 260-281 in losses.py)
- `lmd` (lines 303-312 in losses.py, 161-166 in losses.py for PFMM, 267-275 in loss_args.py)
- `interpolant` (lines 391-403 in losses.py for ESD)
- `inverse` (lines 221-234 in losses.py for FMM)
- Mean flow stopgrad types: `standard`, `self_distill_full`, `self_distill_jvp`, `self_distill_outer` (lines 473-516 in losses.py)

**Code Locations:**
- `py/common/losses.py`:
  - `lsd_term()`: Remove `lsd_mean_flow` and `lmd` branches (lines 260-281, 303-312)
  - `esd_term()`: Remove `interpolant` branch (lines 391-403)
  - `fmm_term()`: Remove `inverse` branch (lines 221-234)
  - `pfmm_term()`: Remove `lmd` branch (lines 161-166)
  - `mean_flow()`: Remove all stopgrad variants, keep only if needed (entire function lines 457-518)

### 3. PSD LMD Type and Associated Parameters
**Rationale**: `psd_type="lmd"` (formerly pfmm_type) is not used in any final config. Only `uniform` and `shortcut` are used.

**Parameters to REMOVE:**
- `config.training.hmin` - only used for LMD (always 0.0 in final configs)
- `config.training.hmax` - only used for LMD (always 0.0 in final configs)
- `psd_type="lmd"` support (formerly pfmm_type)

**Code Locations:**
- `py/common/loss_args.py`: Remove LMD sampling logic (lines 267-275)
- `py/common/losses.py`: Remove LMD branch from `psd_term()` (renamed from pfmm_term)
- Config files: Remove `hmin` and `hmax` parameters

### 4. Calc Both Diagonals
**Rationale**: `calc_both_diagonals=False` in all final configs - this feature is never used.

**Config Parameters to REMOVE:**
- `config.training.calc_both_diagonals` - always False

**Code to Remove:**
- `py/common/losses.py`:
  - `self_distill()`: Remove conditional for computing second diagonal (lines 557-570)
  - `setup_loss()`: Remove `calc_both_diagonals` parameter passing (line 680)

### 5. Rescale LSD
**Rationale**: `rescale_lsd=False` in all final configs - this feature is never used.

**Config Parameters to REMOVE:**
- `config.training.rescale_lsd` - always False
- `config.training.min_step` - only used when rescaling (not present in final configs)

**Code to Remove:**
- `py/common/losses.py`:
  - `lsd_term()`: Remove rescaling conditional (lines 332-335, simplify to just `error = b_eval - dt_Xst`)
  - `setup_loss()`: Remove `rescale_lsd` parameter passing (lines 683, 743)
- `py/common/loss_args.py`: Remove rescaling check (lines 381-383)

### 6. Unused Time Sampling Strategies
**Rationale**: Only `uniform_upper_triangle` is needed. User confirmed to remove ALL other strategies.

**Sampling Strategies to REMOVE:**
- `uniform_full_plane` - **REMOVE** (update the one CIFAR10 config using it)
- `uniform_full_plane_shared` (lines 171-189 in loss_args.py)
- `uniform_upper_triangle_shared` (lines 191-201 in loss_args.py)
- `align_your_flow` (lines 203-237 in loss_args.py)
- `align_your_flow_shared` (lines 203-237 in loss_args.py)

**SIMPLIFICATION:** Since there's only one sampling strategy, remove the `sampling_strategy` parameter entirely and hardcode `uniform_upper_triangle` behavior.

**Config Parameters to REMOVE:**
- `config.training.tau_mean` - only for AYF (set to 0.0 in configs but never used)
- `config.training.tau_std` - only for AYF (set to 1.0 in configs but never used)

**Code to Remove:**
- `py/common/loss_args.py`: Remove branches for shared and AYF sampling (lines 171-237)
- Remove `_compute_ayf_d()` helper (lines 51-56)
- Config files: Remove `tau_mean` and `tau_std` parameters

### 7. Unused Loss Types
**Rationale**: Only `pfmm`, `lsd`, and `esd` are used. `mean_flow` and `fmm` are experimental.

**Loss Types to REMOVE:**
- `mean_flow` - not used in any final config
- `fmm` - not used in any final config

**Code to Remove:**
- `py/common/losses.py`:
  - `mean_flow()` function (lines 457-518) - ENTIRE FUNCTION
  - `fmm_term()` function (lines 181-237) - ENTIRE FUNCTION
  - Remove from `self_distill()` (lines 621-633, 634-646)
  - Remove from `setup_loss()` offdiagonal handler (lines 761-785)

### 8. Load From NVIDIA
**Rationale**: `load_from_nvidia=False` in all final configs - not used for paper results.

**Config Parameters to REMOVE:**
- `config.network.load_from_nvidia` - always False

**Code to Remove:**
- `py/common/state_utils.py`: `load_nvidia_checkpoint()` function (lines 156-195)
- `py/common/state_utils.py`: Remove NVIDIA loading branch in `setup_training_state()` (lines 279-282)

### 9. Dual Weight Functions
**Rationale**: `use_dual_weight_functions=False` in all final configs - single weight function is always used.

**Config Parameters to REMOVE:**
- `config.network.use_dual_weight_functions` - always False

**Code to Remove:**
- `py/common/edm2_net.py`: Remove dual weight function support
  - Remove `calc_weight_diagonal()` and `calc_weight_offdiagonal()` methods
  - Simplify to single `calc_weight()` function
- `py/common/network_utils.py`: Remove dual weight wrapper methods (lines 70-77, 185-194)
- `py/common/losses.py`: Remove all `use_dual_weight_functions` conditionals
  - `diagonal_term()` (lines 72-76, simplify to always use `calc_weight(t, t)`)
  - `pfmm_term()` (lines 173-177)
  - `lsd_term()` (lines 326-329)
  - `esd_term()` (lines 449-452)
  - Remove parameter throughout `setup_loss()` (lines 659, 685, 701, 728, 744, 759)

### 10. Annealing Parameters
**Rationale**: All final configs use `interp_anneal=0`, `interpolant_steps=0`, `annealing_steps=0` - no annealing.

**Config Parameters to REMOVE:**
- `config.training.interp_anneal` - always 0
- `config.training.interpolant_steps` - always 0
- `config.training.annealing_steps` - always 0 (but keep the annealing schedule infrastructure for delta constraint)

**Code to SIMPLIFY (not remove):**
- `py/launchers/learn.py`: `setup_annealing_schedule()` - simplify to always return `constant_schedule(tmax - tmin)`
- `py/common/state_utils.py`: `use_velocity_loss()` - simplify since interp_anneal is always 0

**NOTE**: Keep the delta constraint in loss_args.py as it's used for time sampling even without annealing.

### 11. EMA Teacher
**Rationale**: `use_ema_teacher=False` in all final configs - teacher is always the current params.

**Config Parameters to REMOVE:**
- `config.training.use_ema_teacher` - always False

**Code to SIMPLIFY:**
- `py/common/loss_args.py`: Remove EMA teacher selection (lines 389-392, simplify to always use `train_state.params`)

---

## Additional Simplifications

### 1. Unused Config Parameters
Remove config parameters that are set but never used:
- Network loading parameters when `load_path=""` (load_ema_fac, etc.)
- FID velocity parameters when not training velocity (`fid_n_steps_velocity=None`)

### 2. Code Quality Improvements
- Remove verbose AI-generated comments
- Consolidate duplicated logic
- Simplify overly nested conditionals
- Remove debug print statements
- Apply black and isort formatting

### 3. Documentation Updates
- Update `py/CLAUDE.md` to reflect simplified codebase
- Remove references to removed features
- Simplify architecture descriptions

---

## Files Requiring Modification

### Major Changes (Significant Refactoring)

1. **`py/common/losses.py`** (852 lines)
   - Remove `mean_flow()` function
   - Remove `fmm_term()` function
   - Simplify `lsd_term()` - remove unused stopgrad branches and rescaling
   - Simplify `esd_term()` - remove unused stopgrad branches
   - Simplify `pfmm_term()` - remove LMD support
   - Remove dual weight conditionals throughout
   - Remove calc_both_diagonals support
   - Estimated reduction: ~250 lines

2. **`py/common/loss_args.py`** (427 lines)
   - Remove shared sampling strategies
   - Remove AYF sampling strategies
   - Remove LMD sampling logic
   - Remove rescaling check
   - Simplify EMA teacher selection
   - Remove `_compute_ayf_d()` helper
   - Estimated reduction: ~100 lines

3. **`py/common/state_utils.py`** (322 lines)
   - Remove velocity conversion functions
   - Remove `load_velocity_checkpoint()`
   - Remove `load_nvidia_checkpoint()`
   - Remove velocity initialization
   - Simplify `use_velocity_loss()`
   - Estimated reduction: ~150 lines

4. **`py/common/velocity.py`** (168 lines)
   - **REMOVE ENTIRE FILE**

5. **`py/common/edm2_net.py`** (to be analyzed)
   - Remove dual weight function support
   - Simplify weight calculation to single function

6. **`py/common/network_utils.py`** (285 lines)
   - Remove dual weight wrapper methods
   - Minor simplifications

### Moderate Changes

7. **`py/launchers/learn.py`** (300+ lines)
   - Simplify `setup_annealing_schedule()` for no-annealing case
   - Remove velocity-specific compilation logic
   - Minor cleanup

8. **`py/configs/*.py`** (All config files)
   - Remove unused parameters from all configs
   - Clean up comments

### Minor Changes

9. **`py/common/flow_map.py`** (199 lines)
   - Remove dual weight wrapper methods (if any)
   - Otherwise minimal changes

10. **`py/common/interpolant.py`** (to be checked)
    - Likely no changes needed

11. **`py/common/logging.py`** (to be checked)
    - Remove velocity-related logging if any

12. **`py/launchers/sample_*.py`** (Sampling scripts)
    - Remove velocity sampling support
    - Simplify checkpoint loading

---

## Estimated Impact

### Code Reduction
- **Before**: ~5000+ lines of Python code
- **After**: ~4000 lines (estimated 20-25% reduction)
- **Removed**: ~1000+ lines of experimental/unused code

### Files Removed
- `py/common/velocity.py` (complete removal)

### Files Significantly Simplified
- `py/common/losses.py`: -250 lines
- `py/common/loss_args.py`: -100 lines
- `py/common/state_utils.py`: -150 lines
- `py/common/edm2_net.py`: -50 lines (estimated)

---

## Implementation Strategy

### Phase 1: Analysis and Validation
1. ✅ Analyze all config files (DONE)
2. ✅ Map features to code locations (DONE)
3. ✅ Create comprehensive removal plan (DONE)
4. ⏳ Review plan with user for approval

### Phase 2: Systematic Removal (After User Approval)

**Stage 1: Rename PFMM → PSD**
1. Rename all "pfmm" references to "psd" in code
2. Rename `pfmm_term()` → `psd_term()` in losses.py
3. Update config parameter names: `pfmm_type` → `psd_type`
4. Update all config files to use "psd" instead of "pfmm"
5. Update comments and docstrings

**Stage 2: Remove Entire Features**
1. Delete `py/common/velocity.py`
2. Remove `mean_flow()` and `fmm_term()` from losses.py
3. Remove NVIDIA checkpoint loading
4. Remove dual weight function infrastructure
5. Remove velocity-related functions from state_utils.py

**Stage 3: Simplify Stopgrad Logic**
1. Simplify `lsd_term()` - remove unused branches
2. Simplify `esd_term()` - remove unused branches
3. Simplify `psd_term()` (formerly pfmm_term) - remove LMD
4. Update all call sites

**Stage 4: Simplify Sampling**
1. Remove ALL sampling strategies except uniform_upper_triangle
2. Remove sampling_strategy parameter entirely (hardcode uniform_upper_triangle)
3. Update the one CIFAR10 config using uniform_full_plane
4. Remove related config parameters (tau_mean, tau_std, etc.)

**Stage 5: Clean Up Configs**
1. Remove all unused parameters from config files
2. Add ESD "full convex" experiment to cifar10.py, celeba64.py, afhq64.py
3. Ensure all final configs use updated naming (psd, not pfmm)
4. Simplify config comments

**Stage 6: Final Cleanup**
1. Remove unused imports
2. Apply black formatting
3. Apply isort
4. Update CLAUDE.md documentation
5. Update all comments mentioning removed features

### Phase 3: Testing and Verification
1. Run one config from each dataset to ensure nothing broke
2. Verify training starts and loss computes correctly
3. Check that checkpoint saving/loading works
4. Verify FID computation still works

---

## Risk Mitigation

### Testing Strategy
- Test one representative config from each dataset after refactoring
- Verify training loop starts without errors
- Check that loss values are reasonable
- Ensure checkpoint I/O works

### Rollback Plan
- User has git version control
- Can revert commits if issues arise
- Keep original configs in a backup branch

### Validation Criteria
- All final configs must still parse correctly
- Training must start without errors
- Loss computation must produce sensible values
- FID computation must work
- Checkpoint loading must work

---

## Post-Refactoring Codebase Structure

### Core Components (Simplified)
```
py/
├── common/
│   ├── datasets.py          # Dataset loading
│   ├── edm2_net.py          # EDM2 UNet architecture (simplified)
│   ├── flow_map.py          # Flow map wrapper
│   ├── interpolant.py       # Interpolant definitions
│   ├── losses.py            # Loss functions (simplified)
│   ├── loss_args.py         # Loss argument construction (simplified)
│   ├── network_utils.py     # Network setup (simplified)
│   ├── state_utils.py       # State management (simplified)
│   ├── updates.py           # Optimizer updates
│   ├── fid_utils.py         # FID computation
│   ├── logging.py           # Logging utilities
│   └── dist_utils.py        # Distribution utilities
├── configs/
│   ├── cifar10.py           # CIFAR-10 experiments (cleaned)
│   ├── celeba64.py          # CelebA experiments (cleaned)
│   ├── afhq64.py            # AFHQ experiments (cleaned)
│   └── checker.py           # Checker experiments (cleaned)
└── launchers/
    ├── learn.py             # Main training script (simplified)
    ├── sample_model.py      # Sampling script
    ├── calc_fid.py          # FID calculation
    └── sample_and_calc_fid.py  # Combined sampling + FID
```

### Removed Components
- `py/common/velocity.py` ❌
- Velocity-related functions in state_utils.py ❌
- Mean flow and FMM loss functions ❌
- Dual weight function infrastructure ❌
- NVIDIA checkpoint loading ❌
- Unused stopgrad strategies ❌
- Shared and AYF sampling strategies ❌

---

## Benefits of Refactoring

1. **Clarity**: Easier to understand what the code actually does
2. **Maintainability**: Less code to maintain and debug
3. **Reproducibility**: Only contains features needed for paper results
4. **Usability**: Simpler for others to extend and build upon
5. **Documentation**: Clearer correspondence between code and paper
6. **Performance**: Slightly faster compilation (fewer branches)

---

## Next Steps

1. **Review this plan** with the user
2. **Get approval** before making changes
3. **Implement systematically** following the phased approach
4. **Test thoroughly** at each stage
5. **Update documentation** to reflect changes
6. **Create README.md** for public release (separate task)

---

## User Decisions (Confirmed)

1. ✅ **Remove velocity.py completely** - not used in final experiments
2. ✅ **Remove FMM and mean_flow losses** - experimental only
3. ✅ **Do NOT preserve experimental features** - strip to minimal reproducible set
4. ✅ **Rename PFMM → PSD** (Progressive Self-Distillation) to match paper terminology
5. ✅ **Remove uniform_full_plane** entirely - update the one CIFAR10 config to use uniform_upper_triangle
6. ✅ **Add "full convex" ESD configs** to each image dataset (CIFAR-10, CelebA-64, AFHQ-64)

## Additional Changes from User Feedback

### Rename PFMM → PSD
**Rationale**: Match paper terminology (Progressive Self-Distillation)

**Changes Required:**
- Rename config parameter: `config.training.loss_type = "pfmm"` → `"psd"`
- Rename config parameter: `config.training.pfmm_type` → `config.training.psd_type`
- Update all references in code:
  - `py/common/losses.py`: Rename `pfmm_term()` → `psd_term()`
  - `py/common/losses.py`: Update all "pfmm" string checks to "psd"
  - `py/common/loss_args.py`: Update loss type checks
  - All config files: Update loss_type from "pfmm" to "psd"
  - All comments and docstrings mentioning PFMM
- Update CLAUDE.md documentation

### Remove uniform_full_plane Sampling
**Rationale**: Simplify to single sampling strategy

**Changes Required:**
- Remove `uniform_full_plane` branch from `py/common/loss_args.py`
- Update the one CIFAR10 LSD config that uses it to use `uniform_upper_triangle` instead
- Keep only `uniform_upper_triangle` as the single sampling strategy
- Remove the sampling_strategy parameter entirely since there's only one option

### Add Full Convex ESD Configs
**Rationale**: Include all paper experiments

**New Configs to Create:**
- `cifar10.py`: Add experiment with `loss_type="esd"`, `stopgrad_type="full"`
- `celeba64.py`: Add experiment with `loss_type="esd"`, `stopgrad_type="full"`
- `afhq64.py`: Add experiment with `loss_type="esd"`, `stopgrad_type="full"`
- Keep `checker.py` ESD config as-is (already exists)
