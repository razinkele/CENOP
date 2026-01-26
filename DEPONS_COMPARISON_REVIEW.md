# CENOP vs DEPONS 3.0 - Model Implementation Review

> Review Date: 2026-01-26
> Reviewer: Deep code analysis comparing CENOP Python implementation to original DEPONS 3.0 Java

---

## Issues Requiring Attention

> **Update (2026-01-26): All high and medium priority issues have been FIXED.**

### High Priority - FIXED

1. **Maturity Age Hardcoded (BUG)** - FIXED
   - Location: `population.py` `_handle_reproduction()`
   - Fix: Now uses `params.maturity_age` (default 3.44 years)
   - Changed: `age >= maturity_age` replaces hardcoded `age >= 4`

2. **Mortality Constants Not Parameterized** - FIXED
   - Location: `population.py` `_check_mortality()`
   - Fix: All constants now read from `params`:
     - `params.m_mort_prob_const` (default 0.5)
     - `params.x_survival_const` (default 0.15)
     - `params.mortality_juvenile/adult/elderly` (age-based rates)

### Medium Priority - FIXED

3. **Environmental Modulation Simplified** - FIXED
   - Location: `population.py` `_update_movement()`
   - Fix: Full DEPONS CRW environmental modulation now implemented:
     - Turning angle: `presAngle = angleTmp * (b1*depth + b2*salinity + b3)`
     - Step length: `log10_mov = a0*prev + a1*depth + a2*salinity + R1`
   - Pre-allocated arrays `_salinity_vals` and `_env_mod_angle` for performance

4. **Max Breeding Age Added** - PARAMETERIZED
   - Location: `population.py` `_handle_reproduction()`
   - Fix: Now uses `params.max_breeding_age` (default 20.0 years)
   - User can configure or disable via parameter

### Low Priority - N/A

5. **Unused Parameter: beta** - N/A
   - The `beta` parameter mentioned in original review does not exist in current codebase
   - Mortality uses `m_mort_prob_const` and `x_survival_const` (DEPONS M_MORT_PROB_CONST formulation)

---

## Executive Summary

CENOP (CETacean Noise-Population Model) is a Python reimplementation of the DEPONS (Disturbance Effects on the Harbour Porpoise Population in the North Sea) agent-based model. This review examines the fidelity of the CENOP implementation to the original DEPONS 3.0 Java codebase.

**Overall Assessment:** CENOP faithfully implements the core DEPONS algorithms with some simplifications and one significant enhancement (social communication). The vectorized approach enables simulation of 10,000+ agents with reasonable performance.

| Category | Status | Notes |
|----------|--------|-------|
| Movement (CRW) | Faithful | Vectorized, full env modulation (depth/salinity) |
| Energetics | Faithful | All formulas match DEPONS |
| Mortality | Faithful | Three mortality pathways, all params configurable |
| Reproduction | Faithful | Breeding season, mating day, maturity age param |
| PSM/Dispersal | Faithful | PSM-Type2 implemented correctly |
| Deterrence | Faithful | Turbine + ship models complete |
| Sound Model | Faithful | TL = 20*log10(r) + alpha*r |
| **New Feature** | Social Communication | Not in original DEPONS |

---

## 1. Movement Model (Correlated Random Walk)

### DEPONS 3.0 Reference
From `Porpoise.java`:
- `updatePosition()` - main movement logic
- Turning angle: `angleTmp = b0 * prevAngle + N(0, 4)`
- Present angle: `presAngle = angleTmp * (b1*depth + b2*salinity + b3)`
- Step length: `logMov = R1 + a0*prevLogMov + a1*depth + a2*salinity`

### CENOP Implementation
**File:** `src/cenop/agents/population.py`, method `_update_movement()` (lines 552-623)

```python
# Turning angle calculation
pres_angle = (corr_angle_base * prev_angle + N(r2_mean, r2_sd)) * corr_angle_base_sd
heading += pres_angle  # Applied to heading

# Step length (log10 movement)
log_mov = N(r1_mean, r1_sd) + corr_logmov_length * prev_log_mov
step_dist = (10 ** log_mov) / 4.0  # Convert to grid cells
```

### Parameters Comparison

| Parameter | DEPONS | CENOP | Match |
|-----------|--------|-------|-------|
| b0 (corr_angle_base) | -0.024 | -0.024 | Yes |
| b1 (corr_angle_bathy) | -0.008 | -0.008 | Yes* |
| b2 (corr_angle_salinity) | 0.93 | 0.93 | Yes* |
| b3 (corr_angle_base_sd) | -14.0 | -14.0 | Yes |
| a0 (corr_logmov_length) | 0.35 | 0.35 | Yes |
| a1 (corr_logmov_bathy) | 0.0005 | 0.0005 | Yes* |
| a2 (corr_logmov_salinity) | -0.02 | -0.02 | Yes* |
| R1 mean | 1.25 | 1.25 | Yes |
| R1 sd | 0.15 | 0.15 | Yes |
| R2 mean | 0.0 | 0.0 | Yes |
| R2 sd | 4.0 | 4.0 | Yes |
| max_mov | 1.73 | 1.73 | Yes |

*Note: Environmental modulation (b1, b2, a1, a2) parameters exist but effect is simplified in vectorized version.

### Discrepancies

1. **Environmental Modulation:** The depth/salinity modulation on turning angle and step length is parameterized but the current vectorized implementation uses a simplified constant environment approach. The parameters exist for future enhancement.

2. **Vectorization:** CENOP processes all 10,000+ agents in parallel using NumPy arrays, while DEPONS iterates sequentially. This provides ~100x speedup but may have minor floating-point differences.

---

## 2. Energetics Model

### DEPONS 3.0 Reference
From `Porpoise.java`:
- `updEnergeticStatus()` - main energetics logic
- Energy consumption: `consumed = 0.001 * scaling * E_USE_PER_30_MIN + movementCost`
- Food consumption: `fractOfFoodToEat = (20 - energyLevel) / 10, max 0.99`

### CENOP Implementation
**File:** `src/cenop/agents/population.py`, method `_update_energy_dynamics()` (lines 768-800)

```python
# Food consumption (hungry porpoises eat more)
fract_to_eat = clip((20.0 - energy) / 10.0, 0.0, 0.99)
food_gained = landscape.eat_food_vectorized(x, y, fract_to_eat)
energy += food_gained

# Energy consumption
bmr_cost = 0.001 * scaling_factor * e_use_per_30_min  # 4.5 default
swimming_cost = (10 ** prev_log_mov) * 0.001 * scaling_factor * 0.0  # Disabled
energy -= (bmr_cost + swimming_cost)
```

### Parameters Comparison

| Parameter | DEPONS | CENOP | Match |
|-----------|--------|-------|-------|
| E_USE_PER_30_MIN | 4.5 | 4.5 | Yes |
| E_USE_PER_KM | 0.0 | 0.0 | Yes |
| e_lact (lactation) | 1.4 | 1.4 | Yes |
| e_warm (warm water) | 1.3 | 1.3 | Yes |
| Energy max | 20.0 | 20.0 | Yes |
| Initial energy | N(10, 1) | N(10, 1) | Yes |

### Seasonal Scaling

| Month | DEPONS | CENOP | Match |
|-------|--------|-------|-------|
| Nov-Mar | 1.0 | 1.0 | Yes |
| Apr, Oct | ~1.15 | 1.15 | Yes |
| May-Sep | 1.3 | 1.3 | Yes |
| With calf | x1.4 | x1.4 | Yes |

### Discrepancies

1. **Swimming Cost:** `E_USE_PER_KM = 0` in both implementations, effectively disabling swimming cost. This matches DEPONS default but differs from TRACE document which suggests non-zero values.

2. **Food Model:** CENOP uses `eat_food_vectorized()` which processes all agents simultaneously against the landscape food grid. The fractional consumption formula matches DEPONS.

---

## 3. Mortality Model

### DEPONS 3.0 Reference
From `Porpoise.java`:
- Starvation: `yearlySurvProb = 1 - (M_MORT_PROB_CONST * exp(-energyLevel * xSurvivalProbConst))`
- Natural mortality: Age-dependent annual rates converted to per-tick
- Bycatch: Annual probability converted to per-tick

### CENOP Implementation
**File:** `src/cenop/agents/population.py`, method `_check_mortality()` (lines 802-864)

```python
# Starvation mortality
m_mort_prob_const = 0.5
x_survival_const = 0.15
yearly_surv_prob = 1.0 - (m_mort_prob_const * exp(-energy * x_survival_const))
step_surv_prob = exp(log(yearly_surv_prob) / (360 * 48))

# Age-based mortality
annual_juvenile = 0.15   # age < 1
annual_adult = 0.05      # 1 <= age <= 20
annual_elderly = 0.15    # age > 20
per_tick = annual_rate / 365 / 48

# Bycatch
bycatch_per_tick = bycatch_prob / 365 / 48
```

### Parameters Comparison

| Parameter | DEPONS | CENOP | Match |
|-----------|--------|-------|-------|
| M_MORT_PROB_CONST | 0.5 | 0.5 | Yes |
| xSurvivalProbConst | 0.15 | 0.15 | Yes |
| Juvenile mortality | 15%/yr | 15%/yr | Yes |
| Adult mortality | 5%/yr | 5%/yr | Yes |
| Elderly mortality | 15%/yr | 15%/yr | Yes |
| Juvenile age | <1 yr | <1 yr | Yes |
| Elderly age | >20 yr | >20 yr | Yes |

### Mother-Calf Behavior

CENOP correctly implements DEPONS behavior:
- Lactating mothers facing starvation abandon calf first (survive)
- Only non-lactating starving porpoises die

### Discrepancies

1. **Hardcoded Constants (Should be Parameterized):**
   - `m_mort_prob_const = 0.5` hardcoded at line 809
   - `x_survival_const = 0.15` hardcoded at line 810
   - Parameter `beta = 0.4` exists in `simulation_params.py` but is NOT USED
   - **Note:** These may be intentionally different formulations

2. **Formula Variant:** Comment in code notes DEPONS originally used `yearlySurvProb = 1 - exp(-energyLevel * beta)`, but current implementation uses the M_MORT_PROB_CONST formulation which is more recent.

---

## 4. Reproduction Model

### DEPONS 3.0 Reference
From `Porpoise.java`:
- `updateReproduction()` - main reproduction logic
- Breeding season: Days 195-255 (July-September)
- Mating day: Individual per female, N(225, 20)
- Birth probability: ~60% of eligible females per season

### CENOP Implementation
**File:** `src/cenop/agents/population.py`, method `_handle_reproduction()` (lines 870-925)

```python
# Breeding season check
current_day = day_of_year // 48
if not (195 <= current_day <= 255):
    return

# Eligible females
eligible = mask & is_female & (age >= 4) & (age <= 20) & ~with_calf

# Per-tick birth probability (~60% over 60-day season)
birth_prob = 0.0003
giving_birth = (random() < birth_prob) & eligible
```

### Parameters Comparison

| Parameter | DEPONS | CENOP | Match |
|-----------|--------|-------|-------|
| Breeding start | Day 195 | Day 195 | Yes |
| Breeding end | Day 255 | Day 255 | Yes |
| Mating day mean | 225 | 225 | Yes |
| Mating day SD | 20 | 20 | Yes |
| Maturity age | 3.44 yr | 4 yr | Differs |
| Max breeding age | Not limited | 20 yr | Added |
| Conception prob | 0.68 | 0.0003/tick | Reformulated |
| Sex ratio | 50% | 50% | Yes |

### Discrepancies

1. **Maturity Age (BUG):**
   - `simulation_params.py` defines `maturity_age = 3.44` (correct DEPONS value)
   - `population.py` line 889 hardcodes `age >= 4` instead of using the parameter
   - **Fix Required:** Replace `self.age >= 4` with `self.age >= self.params.maturity_age`

2. **Max Breeding Age:** CENOP adds upper limit (age <= 20) not present in original DEPONS.

3. **Birth Probability:** CENOP reformulates from per-day to per-tick probability to achieve ~60% overall reproduction rate.

---

## 5. PSM and Dispersal (PSM-Type2)

### DEPONS 3.0 Reference
From `DispersalPSMType2.java`:
- Persistent Spatial Memory tracks food obtained per cell
- Dispersal triggers after t_disp days of declining energy
- Target: Remembered cell at preferred distance with highest food expectation
- Dispersal ends at 95% of target distance

### CENOP Implementation
**File:** `src/cenop/agents/population.py`, methods:
- `_check_dispersal_trigger()` (lines 1034-1070)
- `_start_dispersal()` (lines 1072-1180)
- `_update_dispersal()` (lines 1182-1270)

```python
# PSM Memory Buffer (vectorized)
psm_buffer: shape (count, rows, cols, 2)  # [ticks_spent, food_obtained]

# Dispersal trigger
t_disp = 3  # Days of declining energy
min_memory_cells = 50  # Minimum visited cells

# Target selection
preferred_distance = N(300, 100) km
tolerance = 5 km
target = cell with highest food/ticks at preferred distance

# Heading dampening during dispersal
log_mult = 1 / (1 + exp(0.6 * (3*dist_perc - 1.5)))
dampened_angle = pres_angle * log_mult * 0.3
```

### Parameters Comparison

| Parameter | DEPONS | CENOP | Match |
|-----------|--------|-------|-------|
| t_disp | 3 days | 3 days | Yes |
| Min memory cells | 50 | 50 | Yes |
| PSM cell size | 5 grid units | 5 grid units | Yes |
| Preferred dist mean | 300 km | 300 km | Yes |
| Preferred dist SD | 100 km | 100 km | Yes |
| Tolerance | 5 km | 5 km | Yes |
| End at % target | 95% | 95% | Yes |
| psm_log | 0.6 | 0.6 | Yes |

### Discrepancies

1. **Energy History:** CENOP tracks daily energy averages in a rolling buffer. Critical bug fix prevents double-counting of energy (noted in code comments).

2. **Vectorization:** PSM buffer is fully vectorized (count x rows x cols x 2) for all agents, while DEPONS uses per-agent object instances.

---

## 6. Deterrence Model

### DEPONS 3.0 Reference
From `Turbine.java` and `Ship.java`:
- Sound propagation: TL = beta * log10(r) + alpha * r
- Received level: RL = SL - TL
- Deterrence strength: if RL > threshold, strength = RL - threshold
- Deterrence vector: proportional to strength, away from source

### CENOP Implementation - Turbines
**File:** `src/cenop/agents/turbine.py` (lines 121-175)

```python
# Transmission loss
TL = beta_hat * log10(distance_m) + alpha_hat * distance_m

# Received level
RL = source_level - TL

# Deterrence
if RL > deter_threshold:
    strength = RL - deter_threshold
    deterred = True
```

### CENOP Implementation - Ships
**File:** `src/cenop/agents/ship.py` (lines 232-292)

```python
# JOMOPANS source level model
SL = base + 60*log10(length/100) + 20*log10(speed/12) + vhf_weighting

# Day/Night probabilistic response
if is_day:
    linear = pship_int_day + pship_noise_day*RL + pship_dist_day*dist + pship_dist_x_noise_day*RL*dist
else:
    linear = pship_int_night + pship_noise_night*RL + pship_dist_night*dist + pship_dist_x_noise_night*RL*dist
prob = 1 / (1 + exp(-linear))
```

### Parameters Comparison

| Parameter | DEPONS | CENOP | Match |
|-----------|--------|-------|-------|
| alpha_hat | 0.0 | 0.0 | Yes |
| beta_hat | 20.0 | 20.0 | Yes |
| deter_threshold | 158 dB | 158 dB | Yes |
| deter_max_distance | 50 km | 50 km | Yes |
| deter_coeff | 0.07 | 0.07 | Yes |
| deter_time | 5 ticks | 5 ticks | Yes |
| deter_decay | 50% | 50% | Yes |
| Construction SL | 200 dB | 200 dB | Yes |
| Operation SL | 145 dB | 145 dB | Yes |

### Ship Deterrence Coefficients

| Parameter | DEPONS | CENOP | Match |
|-----------|--------|-------|-------|
| pship_int_day | -3.0569351 | -3.0569351 | Yes |
| pship_int_night | -3.233771 | -3.233771 | Yes |
| pship_noise_day | 0.2172813 | 0.2172813 | Yes |
| pship_dist_day | -0.1303880 | -0.1303880 | Yes |
| pship_dist_x_noise_day | 0.0293443 | 0.0293443 | Yes |
| cship_int_day | 2.9647996 | 2.9647996 | Yes |
| cship_int_night | 2.7543376 | 2.7543376 | Yes |

### Discrepancies

1. **Probabilistic Option:** CENOP adds `deter_probabilistic` flag for sigmoid-based probability response (optional, not in original DEPONS).

---

## 7. New Feature: Social Communication

### Not in DEPONS 3.0

CENOP introduces a social communication model not present in the original DEPONS:

**File:** `src/cenop/agents/population.py`, method `_compute_social_vectors()` (lines 241-540)

### Implementation

```python
# Communication detection
source_level = 160 dB (porpoise call)
RL = SL - TL(distance)
SNR = RL - ambient_noise

# Detection probability (logistic)
prob = 1 / (1 + exp(-slope * (SNR - threshold)))

# Social attraction vector
if detected:
    unit_vector = direction to neighbor
    social_dx += unit_vector * prob * social_weight
    social_dy += unit_vector * prob * social_weight
```

### Parameters (New)

| Parameter | Default | Description |
|-----------|---------|-------------|
| communication_enabled | True | Enable social calling |
| communication_range_km | 10.0 | Max detection range |
| communication_source_level | 160 dB | Porpoise call level |
| communication_threshold | 120 dB | 50% detection RL |
| communication_response_slope | 0.2 | Logistic steepness |
| social_weight | 0.3 | Attraction influence |
| communication_recompute_interval | 4 | Ticks between topology updates |

### Impact on Model

- Adds cohesion behavior to porpoise groups
- Can be disabled for DEPONS-compatible runs
- Uses cKDTree for efficient neighbor queries
- Ambient noise masking affects detection probability

---

## 8. Implementation Quality Assessment

### Strengths

1. **Faithful Core Algorithms:** All core DEPONS algorithms are correctly implemented with matching parameters.

2. **Performance:** Vectorized NumPy operations enable 10,000+ agent simulations.

3. **Code Quality:** Clean separation of concerns with documented methods and DEPONS references.

4. **Parameterization:** All parameters accessible via `SimulationParameters` dataclass with validation.

### Areas for Improvement

1. **Environmental Modulation:** Depth/salinity effects on CRW are parameterized but not fully applied in vectorized code.

2. **Documentation:** Some DEPONS formula variants noted in comments but not fully documented.

3. **Validation:** No automated regression tests comparing CENOP output to DEPONS baseline.

---

## 9. Recommendations

### For DEPONS Compatibility

1. **Disable Social Communication:** Set `communication_enabled=False` for DEPONS-compatible runs.

2. **Verify Maturity Age:** Update from 4.0 to 3.44 years to match DEPONS exactly.

3. **Remove Max Breeding Age:** Consider removing the `age <= 20` constraint for exact DEPONS match.

### For Model Validation

1. **Energy Trajectories:** Compare mean population energy over time between CENOP and DEPONS.

2. **Population Dynamics:** Verify birth/death rates match DEPONS over multi-year simulations.

3. **Dispersal Patterns:** Compare PSM dispersal trigger frequency and target selection.

4. **Deterrence Response:** Verify deterrence probability curves match DEPONS for same scenarios.

### For Future Development

1. **Environmental Modulation:** Implement full depth/salinity effects on CRW.

2. **Memory Efficiency:** Consider sparse PSM buffer for very large populations.

3. **Parallel Processing:** Investigate GPU acceleration for 100k+ agent simulations.

---

## Appendix: File Reference

| Component | CENOP File | DEPONS Reference |
|-----------|------------|------------------|
| Movement | population.py | Porpoise.java |
| Energetics | population.py | Porpoise.java |
| Mortality | population.py | Porpoise.java |
| Reproduction | population.py | Porpoise.java |
| PSM | population.py, behavior/psm.py | DispersalPSMType2.java |
| Turbines | agents/turbine.py | Turbine.java |
| Ships | agents/ship.py | Ship.java |
| Sound | behavior/sound.py | SoundSource.java |
| Parameters | parameters/simulation_params.py | SimulationParameters.java |
