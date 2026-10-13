---

# Reproducibility Protocol for Area-to-Flow Neural Surrogates

**Purpose.** This file records the dataset partitioning, random-number control, preprocessing, neural-network hyperparameters, training procedures, ensemble construction, out-of-distribution (OOD) generation, and model-selection rules used for the area-to-flow surrogate experiments.

---

## 1. Task definition


The learning task is local waveform regression:

- **Input:** cross-sectional arterial area waveform, \(A(x,t)\)
- **Target:** colocated volumetric flow waveform, \(q(x,t)\)
- **Samples per waveform:** 100 uniformly spaced time points over one normalized cardiac cycle, \(t/T \in [0,1)\)
- **Input tensor shape:** `B x 1 x 100`
- **Output tensor shape:**
  - deterministic model: `B x 1 x 100`
  - heteroscedastic model: two output channels representing predictive mean and log standard deviation

All reported neural-network results use synthetic virtual subjects generated from the ADAN56 one-dimensional arterial model family.

---

## 2. Dataset composition and partitions

### 2.1 Full synthetic cohort


- Number of virtual subjects: **100**
- Number of waveform locations per subject: **462**
- Total waveform pairs: **46,200**

Each waveform pair consists of an area waveform and its colocated simulated flow waveform.

### 2.2 Primary subject-level partition


| Partition | Subjects | Waveform pairs |
|---|---:|---:|
| Training | 70 | 32,340 |
| Validation | 10 | 4,620 |
| Test | 20 | 9,240 |
| **Total** | **100** | **46,200** |

Partitioning is performed at the **virtual-subject level**. All arterial locations belonging to one virtual subject must remain in the same partition. No locations from a validation or test subject may occur in the training set.

- Partition seed: **42**
- Validation and test subjects remain fixed for the main experiments.
- The **test set is reserved for final evaluation only** and must not be used for hyperparameter selection, early stopping, threshold setting, or recalibration.

### 2.3 Exact partition file


The released archive contains the exact subject identifiers in a machine-readable file. The deterministic seed-42 split is listed below.

```json
{
  "partition_seed": 42,
  "train_subject_ids": [
    "VP043",
    "VP042",
    "VP092",
    "VP010",
    "VP066",
    "VP051",
    "VP002",
    "VP071",
    "VP016",
    "VP079",
    "VP074",
    "VP011",
    "VP056",
    "VP057",
    "VP073",
    "VP046",
    "VP049",
    "VP093",
    "VP077",
    "VP038",
    "VP031",
    "VP022",
    "VP033",
    "VP097",
    "VP081",
    "VP050",
    "VP084",
    "VP027",
    "VP088",
    "VP034",
    "VP009",
    "VP048",
    "VP060",
    "VP064",
    "VP075",
    "VP045",
    "VP099",
    "VP053",
    "VP086",
    "VP013",
    "VP037",
    "VP024",
    "VP040",
    "VP041",
    "VP019",
    "VP067",
    "VP062",
    "VP061",
    "VP008",
    "VP035",
    "VP100",
    "VP047",
    "VP003",
    "VP052",
    "VP017",
    "VP039",
    "VP059",
    "VP069",
    "VP023",
    "VP063",
    "VP025",
    "VP006",
    "VP007",
    "VP068",
    "VP083",
    "VP020",
    "VP080",
    "VP044",
    "VP091",
    "VP021"
  ],
  "validation_subject_ids": [
    "VP001",
    "VP096",
    "VP058",
    "VP094",
    "VP054",
    "VP090",
    "VP026",
    "VP072",
    "VP085",
    "VP078"
  ],
  "test_subject_ids": [
    "VP065",
    "VP030",
    "VP028",
    "VP089",
    "VP098",
    "VP005",
    "VP055",
    "VP076",
    "VP012",
    "VP070",
    "VP087",
    "VP014",
    "VP018",
    "VP029",
    "VP032",
    "VP036",
    "VP095",
    "VP004",
    "VP015",
    "VP082"
  ]
}
```


---

## 3. Training-size convergence experiment

Training-size experiments use:

\[
N_{\mathrm{train}} \in \{20,40,55,70\}.
\]

The same:

- 10 validation subjects, and
- 20 test subjects

are retained across all training-size experiments.

For each training size:

- **5 independent runs** are performed.
- For \(N_{\mathrm{train}}<70\), runs use:
  - independent training-subject subsets, and
  - independent model-initialization/stochastic-training seeds.
- For \(N_{\mathrm{train}}=70\), the full 70-subject training partition is used in every run; only initialization, data ordering, and stochastic-training seeds vary.
- Training normalization statistics are refitted independently for each reduced training subset.
- The same held-out test subjects are used for all comparisons.

Reported results are the **mean ± standard deviation across five runs**.

The **5%** criterion is taken from the manuscript's training-size analysis. For a reported metric \(M\), the change from 55 to 70 training subjects is calculated as:

\[
\Delta_{55\rightarrow70}(M)
=
100\frac{|M_{55}-M_{70}|}{|M_{70}|}.
\]

The **5% relative-change criterion applies only to primary RMSE and Gaussian NLL** and indicates an **approaching**, not established, performance plateau. Coverage and sharpness are evaluated descriptively for convergence toward nominal calibration and interval stabilization; they are not required to satisfy the 5% relative-change threshold. The manuscript reports changes below 5% for RMSE and NLL, while MAE changes remain approximately 6%. This training-size criterion is separate from the manuscript's forward-model cohort-convergence criteria.

### 3.1 Deterministic seed schedule for new reproduction runs


This schedule applies **only to new reproduction runs**. It does not specify historical replicate seeds and must not be used to relabel previously reported runs.

The following schedule is specified for new reproduction runs when historical replicate seeds are unavailable. These values must not be interpreted as the historical seeds used to generate previously reported results:

```text
subset_seed(N, repeat) = 100000 + 100*N + repeat
model_seed(N, repeat)  = 200000 + 100*N + repeat
loader_seed(N, repeat) = 300000 + 100*N + repeat
```

where:

- `N` is one of `20, 40, 55, 70`
- `repeat` is one of `1, 2, 3, 4, 5`

Example:

| Ntrain | Repeat | Subset seed | Model seed | Loader seed |
|---:|---:|---:|---:|---:|
| 20 | 1 | 102001 | 202001 | 302001 |
| 20 | 2 | 102002 | 202002 | 302002 |
| 40 | 1 | 104001 | 204001 | 304001 |
| 55 | 1 | 105501 | 205501 | 305501 |
| 70 | 1 | not applicable | 207001 | 307001 |

For `Ntrain = 70`, no subset draw is required because the complete primary training partition is used.

The complete reproduction-only seed table is:

| Ntrain | Repeat | Subset seed | Model seed | Loader seed |
|---:|---:|---:|---:|---:|
| 20 | 1 | 102001 | 202001 | 302001 |
| 20 | 2 | 102002 | 202002 | 302002 |
| 20 | 3 | 102003 | 202003 | 302003 |
| 20 | 4 | 102004 | 202004 | 302004 |
| 20 | 5 | 102005 | 202005 | 302005 |
| 40 | 1 | 104001 | 204001 | 304001 |
| 40 | 2 | 104002 | 204002 | 304002 |
| 40 | 3 | 104003 | 204003 | 304003 |
| 40 | 4 | 104004 | 204004 | 304004 |
| 40 | 5 | 104005 | 204005 | 304005 |
| 55 | 1 | 105501 | 205501 | 305501 |
| 55 | 2 | 105502 | 205502 | 305502 |
| 55 | 3 | 105503 | 205503 | 305503 |
| 55 | 4 | 105504 | 205504 | 305504 |
| 55 | 5 | 105505 | 205505 | 305505 |
| 70 | 1 | N/A | 207001 | 307001 |
| 70 | 2 | N/A | 207002 | 307002 |
| 70 | 3 | N/A | 207003 | 307003 |
| 70 | 4 | N/A | 207004 | 307004 |
| 70 | 5 | N/A | 207005 | 307005 |

If historical run seeds exist in checkpoint metadata, logs, or filenames, **use those historical seeds instead of this proposed schedule**.

---

## 4. Random-number control

### 4.1 Primary experiment seeds


| Purpose | Seed |
|---|---:|
| Virtual-cohort sampling | **2024** |
| Primary train/validation/test partition | **42** |
| Principal single-model training | **12345** |
| Data-loader shuffling | **54321** |
| OOD cohort generation | **2025** |
| Ensemble member 1 | **1001** |
| Ensemble member 2 | **1002** |
| Ensemble member 3 | **1003** |
| Ensemble member 4 | **1004** |
| Ensemble member 5 | **1005** |

### 4.2 Libraries to seed

Seed, where applicable:

- Python built-in random generator
- NumPy
- PyTorch CPU generator
- PyTorch CUDA generator(s)
- DataLoader generator / sampler

Deterministic PyTorch/CUDA execution should be enabled where supported.

### 4.3 Reference implementation

The adopted random-number-control reference implementation is:

```python
import os
import random
import warnings
import numpy as np
import torch

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Use deterministic kernels where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as exc:
        warnings.warn(
            f"Deterministic execution could not be enabled: {exc}",
            RuntimeWarning,
        )
```

The adopted reference implementation reports failure to enable deterministic algorithms. Set `PYTHONHASHSEED` before launching Python if reproducible Python hash randomization is required; assigning it inside the function does not reset the current interpreter's hash seed. Enabling deterministic algorithms can also raise errors later when an unsupported operation is executed.

**Important:** deterministic execution can depend on device, CUDA/cuDNN versions, and individual operations. The software environment and hardware therefore are archived together with the seeds.

---

## 5. Preprocessing


### 5.1 Physical-unit conversion

Before neural-network training:

- area is converted from \(\mathrm{m^2}\) to \(\mathrm{cm^2}\)
- flow is converted from \(\mathrm{m^3\,s^{-1}}\) to \(\mathrm{cm^3\,s^{-1}}\)

### 5.2 Temporal representation

Each waveform is periodically resampled to:

- **100 uniform points**
- normalized cardiac-cycle coordinate \(t/T \in [0,1)\)

No additional filtering or smoothing is applied.

### 5.3 Standardization

Input and target waveforms are standardized using **training-set statistics only**:

\[
\widetilde{A}
=
\frac{A-\mu_{A,\mathrm{train}}}
{\sigma_{A,\mathrm{train}}},
\]

\[
\widetilde{q}
=
\frac{q-\mu_{q,\mathrm{train}}}
{\sigma_{q,\mathrm{train}}}.
\]

Validation, test, ensemble, and OOD data use the normalization statistics fitted from the relevant training partition.

Predictions are transformed back to physical units before reporting physical-error metrics.

The archive retains, at minimum:

```text
normalization/
    area_mean.npy
    area_std.npy
    flow_mean.npy
    flow_std.npy
```

or an equivalent single JSON/NPZ file.

For reduced training-size experiments, the normalization statistics must be stored separately for each run because they are refitted to each training subset.

---

## 6. Neural-network architecture

### 6.1 Shared one-dimensional convolutional backbone

Input:

```text
1 channel x 100 time samples
```

Initial layer:

- Conv1D: `1 -> 32` channels
- kernel size: **7**
- stride: **1**
- dilation: **1**
- dimension-preserving zero padding
- BatchNorm1D
- ReLU

Eight subsequent convolutional blocks use channel widths:

```text
[32, 32, 64, 64, 64, 64, 32, 32]
```

with:

- alternating kernel sizes **5** and **3**
- stride **1**
- dilation **1**
- dimension-preserving zero padding
- BatchNorm1D
- ReLU

Output layer:

- final **1 x 1 Conv1D**

Outputs:

- deterministic model: one mean-flow channel
- heteroscedastic model:
  - predictive mean \(\mu_\theta(t)\)
  - log standard deviation \(\ell_\theta(t)=\log \sigma_\theta(t)\)

### 6.2 Residual connection

A fixed residual connection is applied **only to the predictive mean** for both deterministic and heteroscedastic models:

\[
\widehat{\mu}_\theta(t)
=
f_{\mu,\theta}(\widetilde{A})(t)
+0.10\,\widetilde{A}(t).
\]

The heteroscedastic log-standard-deviation channel has no residual term:

\[
\widehat{\ell}_\theta(t)=f_{\ell,\theta}(\widetilde{A})(t).
\]

Residual coefficient:

```text
0.10
```

### 6.3 Parameter initialization

- convolution weights: **Kaiming-normal**
- convolution biases: **0**
- BatchNorm scale / gamma: **1**
- BatchNorm offset / beta: **0**
- heteroscedastic log-standard-deviation output bias: **-1.0**

---

## 7. Optimizer and common training hyperparameters


Optimizer:

```text
Adam
```

Adam parameters:

| Hyperparameter | Value |
|---|---:|
| beta1 | 0.9 |
| beta2 | 0.999 |
| epsilon | 1e-8 |
| weight decay | 1e-5 |

Other shared settings:

| Hyperparameter | Value |
|---|---:|
| Training batch size | **256** |
| Evaluation batch size | **512** |
| Gradient-norm clipping | **1.0** |
| Minimum learning rate | **1e-6** |
| LR reduction factor | **0.5** |
| LR scheduler patience | **15 epochs** |
| Early-stopping patience | **30 epochs** |

Training and evaluation DataLoaders use **`num_workers=0`**; no separate worker-seeding procedure is required. Evaluation loaders are not shuffled.

A single PyTorch generator controls training-data shuffling and is initialized **once with seed `54321`** for the reported experiments. For new training-size reproduction runs only, Section 3.1 specifies run-specific loader seeds instead. It is not reset between epochs. Its RNG state advances across epochs, yielding deterministic, epoch-dependent shuffling.

---

## 8. Deterministic model training


### 8.1 Objective

The deterministic network minimizes mean squared error in standardized target space:

\[
\mathcal{L}_{\mathrm{MSE}}
=
\frac{1}{B N_t}
\sum_{b=1}^{B}
\sum_{j=1}^{N_t}
\left(
\widetilde q_{b,j}
-
\widehat\mu_{b,j}
\right)^2.
\]

### 8.2 Training schedule

| Setting | Value |
|---|---:|
| Initial learning rate | **1e-3** |
| Maximum epochs | **300** |
| Scheduler monitor | validation MSE |
| Reduce LR after | 15 epochs without validation improvement |
| LR factor | 0.5 |
| Minimum LR | 1e-6 |
| Early stopping | 30 epochs without validation-MSE improvement |
| Best checkpoint | minimum subject-averaged validation MSE |

The best validation checkpoint, not necessarily the last epoch, is used for final test evaluation.

---

## 9. Heteroscedastic model training


Training has two phases.

### 9.1 Phase 1: mean-only warm-up

- Duration: **50 epochs**
- Objective: MSE on the predictive mean
- Initial learning rate: **1e-3**
- The log-standard-deviation output is excluded from the warm-up loss.

### 9.2 Phase 2: joint probabilistic optimization

- Maximum duration: **250 epochs**
- Objective: Gaussian negative log likelihood
- Initial learning rate: **3e-4**
- Backbone, mean head, and log-standard-deviation output are jointly optimized.

The Gaussian NLL is:

\[
\mathcal{L}_{\mathrm{NLL}}
=
\frac{1}{B N_t}
\sum_{b=1}^{B}
\sum_{j=1}^{N_t}
\left[
\ell_{b,j}
+
\frac{1}{2}
\exp(-2\ell_{b,j})
\left(
\widetilde q_{b,j}
-
\widehat\mu_{b,j}
\right)^2
\right].
\]

Terms independent of the network parameters are omitted.

### 9.3 Predictive-standard-deviation constraints

Before exponentiation:

```text
log_sigma = clamp(log_sigma, -3, 1)
```

Thus:

\[
\widehat{\sigma}
=
\exp\left[\operatorname{clamp}(\ell,-3,1)\right].
\]

During loss calculation, a numerical lower bound of:

```text
1e-6
```

is imposed on the predictive standard deviation. This floor is a defensive numerical safeguard: under the stated clamp, the standardized predictive standard deviation is at least `exp(-3)` (approximately `0.0498`), so the `1e-6` floor is not expected to become active.

### 9.4 Stopping and checkpoint selection

| Setting | Value |
|---|---:|
| Warm-up epochs | 50 |
| Joint NLL epochs | up to 250 |
| Joint-training LR | 3e-4 |
| Scheduler monitor | validation NLL |
| LR scheduler patience | 15 epochs |
| Early-stopping patience | 30 epochs |
| Best checkpoint | minimum subject-averaged validation Gaussian NLL |

---

## 10. Subject-averaged model-selection criterion

Validation losses are first computed **within each virtual subject** and then averaged across validation subjects.

This avoids giving greater weight to a subject merely because it contributes more arterial locations.

For deterministic models:

```text
selection metric = subject-averaged validation MSE
```

For heteroscedastic models:

```text
selection metric = subject-averaged validation Gaussian NLL
```

The test set is not examined during model or hyperparameter selection.

### 10.1 Checkpoint rule

Use:

```text
best_epoch = argmin(validation_metric)
```

and restore the corresponding checkpoint before final evaluation.

### 10.2 Tie handling

Use the following deterministic tie rule:

```text
retain the earliest checkpoint achieving the minimum metric
```

---

## 11. Deep ensemble



The uncertainty benchmark consists of:

```text
M = 5 heteroscedastic networks
```

Ensemble seeds:

```text
[1001, 1002, 1003, 1004, 1005]
```

All members use the same:

- architecture
- preprocessing
- 70/10/20 subject-level partition
- normalization procedure
- optimizer
- learning-rate schedule
- heteroscedastic loss
- stopping criteria

Members differ by random initialization and stochastic training.

Ensemble predictive mean:

\[
\bar{\mu}
=
\frac{1}{M}\sum_{m=1}^{M}\mu_m.
\]

Aleatoric variance:

\[
\sigma_{\mathrm{alea}}^2
=
\frac{1}{M}\sum_{m=1}^{M}\sigma_m^2.
\]

Epistemic variance:

\[
\sigma_{\mathrm{epis}}^2
=
\frac{1}{M}
\sum_{m=1}^{M}
(\mu_m-\bar{\mu})^2.
\]

Total variance:

\[
\sigma_{\mathrm{tot}}^2
=
\sigma_{\mathrm{alea}}^2
+
\sigma_{\mathrm{epis}}^2.
\]

Every ensemble member has a separate checkpoint and metadata file recording:

- seed
- best epoch
- best validation NLL
- final learning rate
- normalization-file identifier
- Git commit hash
- software environment

---

## 12. Out-of-distribution stress tests


OOD seed:

```text
2025
```

Number of additional OOD subjects:

```text
30
```

Five scenarios are generated, with **6 subjects per scenario**:

1. altered calibre
2. increased stiffness
3. increased terminal resistance
4. reduced terminal compliance
5. combined perturbation

Relative to nominal ADAN56:

| Parameter | OOD range |
|---|---:|
| Radius scale | `[0.85, 0.90)` or `[1.10, 1.15)` |
| Young's modulus scale | `[1.40, 1.60)` |
| Terminal resistance scale | `[1.35, 1.50)` |
| Terminal compliance scale | `[0.50, 0.65)` |

Within each scenario, values are generated by **uniform random sampling with seed 2025**. In each single-parameter scenario, only the designated parameter class is changed; all other parameters remain at nominal ADAN56 values. Radius values are sampled only outside the training interval.

For bit-for-bit OOD generation, the adopted NumPy implementation is:

```python
rng = np.random.Generator(np.random.PCG64(2025))
```

Subjects are generated sequentially in the fixed scenario order:

1. altered calibre
2. increased stiffness
3. increased terminal resistance
4. reduced terminal compliance
5. combined perturbation

Each scenario contains six subjects, and random draws are consumed in subject order. Radius sampling uses a Bernoulli-then-uniform procedure:

```python
side = rng.binomial(1, 0.5)  # 0 = lower interval, 1 = upper interval
if side == 0:
    s_rad = rng.uniform(0.85, 0.90)
else:
    s_rad = rng.uniform(1.10, 1.15)
```

Thus the operational NumPy intervals are `[0.85, 0.90)` and `[1.10, 1.15)`: `Generator.uniform(low, high)` includes `low` and excludes `high` under the NumPy 1.26.4 convention. No endpoint post-processing or clipping is applied.

The single-parameter scenarios consume one parameter draw per subject after any required radius-side draw. In the combined scenario, radius uses the **same Bernoulli-then-uniform procedure**, followed by independent draws in the fixed order:

```python
side = rng.binomial(1, 0.5)
s_rad = rng.uniform(0.85, 0.90) if side == 0 else rng.uniform(1.10, 1.15)
s_E = rng.uniform(1.40, 1.60)
s_res = rng.uniform(1.35, 1.50)
s_comp = rng.uniform(0.50, 0.65)
```

No random numbers are discarded between subjects or scenarios. This ordering is part of the reproducibility specification; changing it while retaining seed 2025 will produce a different OOD cohort.

Geometry and stiffness perturbations are applied coherently across the arterial network. Terminal resistance or compliance scaling is applied uniformly to all terminal outlets in the corresponding scenario. In the combined scenario, radius, Young's modulus, terminal resistance, and terminal compliance factors are sampled independently within their respective OOD ranges and applied simultaneously.

OOD data:

- are never used for model fitting,
- are never used for checkpoint selection,
- are never used for recalibration,
- use the unchanged training normalization,
- are evaluated with the same five trained ensemble members.

The archive retains:

```text
ood/
    seed_2025/
        subject_parameters.*
        scenario_labels.*
        area_waveforms.*
        flow_waveforms.*
        ood_config.*
```

---

## 13. Evaluation metrics

### 13.1 Point-prediction metrics

Let \(q_i(t)\) and \(\widehat q_i(t)\) denote the simulated and predicted flow waveforms for artery \(i\) of a virtual subject. For probabilistic models, \(\widehat q_i\) is the predictive mean. Physical-error metrics use flow in \(\mathrm{cm^3\,s^{-1}}\).

Within each waveform:

\[
\mathrm{MAE}_i=\operatorname{mean}_t|\widehat q_i(t)-q_i(t)|,
\qquad
\mathrm{RMSE}_i=\sqrt{\operatorname{mean}_t(\widehat q_i(t)-q_i(t))^2}.
\]

The time-sample-wise relative residual uses a minimum reference magnitude of \(1\,\mathrm{cm^3\,s^{-1}}\):

\[
r_{\mathrm{rel},i}(t)=
\frac{\widehat q_i(t)-q_i(t)}
{\max\left(|q_i(t)|,1\,\mathrm{cm^3\,s^{-1}}\right)}.
\]

Relative MAE and relative RMSE, in percent, are:

\[
\mathrm{MAE}_{\mathrm{rel},i}=100\operatorname{mean}_t|r_{\mathrm{rel},i}(t)|,
\qquad
\mathrm{RMSE}_{\mathrm{rel},i}=100\sqrt{\operatorname{mean}_t\left(r_{\mathrm{rel},i}^2(t)\right)}.
\]

Peak-flow errors compare waveform maxima:

\[
E_{\mathrm{peak,abs},i}
=\left|\max_t\widehat q_i(t)-\max_t q_i(t)\right|,
\]

\[
E_{\mathrm{peak,rel},i}
=100\frac{E_{\mathrm{peak,abs},i}}
{\max\left(\left|\max_t q_i(t)\right|,1\,\mathrm{cm^3\,s^{-1}}\right)}.
\]

Reported peak-flow MAE is the arithmetic mean of \(E_{\mathrm{peak,abs},i}\) over the evaluated waveforms, using the aggregation rule below.

Cycle-integrated flow volumes and their errors are:

\[
V_i=\int_0^Tq_i(t)\,dt,
\qquad
\widehat V_i=\int_0^T\widehat q_i(t)\,dt,
\]

\[
E_{V,\mathrm{abs},i}=|\widehat V_i-V_i|,
\qquad
E_{V,\mathrm{rel},i}=100\frac{|\widehat V_i-V_i|}{|V_i|}.
\]

Cycle-integrated flow is interpreted as stroke volume **only at the aortic inlet**; elsewhere it is local cycle-integrated flow volume. The 100 stored waveform samples lie on the periodic normalized grid
\[
\tau_j=\frac{j}{N_t},\qquad j=0,\ldots,N_t-1,\qquad N_t=100,
\]
with physical time \(t_j=T\tau_j\) and \(\Delta t=T/N_t\). For integration, the periodic endpoint is restored explicitly by appending \(q(T)=q(0)\) (and likewise for \(\widehat q\)), after which the composite trapezoidal rule is applied over \([0,T]\). Under NumPy 1.26.4 the reference implementation is:

```python
dt = T / q.size
q_periodic = np.concatenate([q, q[:1]])
qhat_periodic = np.concatenate([qhat, qhat[:1]])
t_periodic = np.arange(q.size + 1) * dt

V = np.trapz(q_periodic, x=t_periodic)
Vhat = np.trapz(qhat_periodic, x=t_periodic)
```

Flow is in \(\mathrm{cm^3\,s^{-1}}\) and physical time is in seconds, so the integrated volume is in \(\mathrm{cm^3}=\mathrm{mL}\). No additional denominator floor is applied to \(E_{V,\mathrm{rel}}\); all retained evaluation waveforms have non-zero reference cycle volume. A zero reference volume, if encountered in a future reproduction or extension, must be reported as undefined for the relative-volume metric rather than silently stabilized, while its absolute-volume error remains reportable.

Metrics are calculated first within each artery of each virtual subject, then averaged across arteries within that subject, and finally across subjects. Each subject receives equal weight, and each artery within a subject receives equal weight, regardless of waveform sample count. Where all subjects contribute the same number of arteries, this agrees with the arithmetic mean over all evaluated waveforms.

### 13.2 Probabilistic metrics

For a single heteroscedastic model, \(\widehat\sigma\) is its predictive standard deviation. For the ensemble, it is the total predictive standard deviation:

\[
\sigma_{\mathrm{tot}}
=\sqrt{\sigma_{\mathrm{alea}}^2+\sigma_{\mathrm{epis}}^2}.
\]

The variance components are defined in Section 11. Standard deviations used for physical-unit uncertainty summaries are transformed back using the target normalization scale.

Gaussian NLL is evaluated in **standardized target space**, using the same convention as the training objective in Section 9.2. For a single heteroscedastic model, the model mean and aleatoric standard deviation are used; for the ensemble, the ensemble predictive mean and total predictive standard deviation are used after expressing both in standardized target units. The additive Gaussian constant \(\tfrac{1}{2}\log(2\pi)\) is omitted, consistent with the training loss, so reported NLL values are directly comparable only to results using this same convention.

For \(k=1,2,3\), empirical coverage is the fraction of samples satisfying:

\[
|q(t)-\widehat\mu(t)|\leq k\widehat\sigma(t).
\]

Nominal Gaussian coverage is \(C_k^{\mathrm{nom}}=2\Phi(k)-1\), approximately 68.27%, 95.45%, and 99.73%. With coverage values expressed as percentages, mean absolute coverage deviation is:

\[
\mathrm{CoverageDeviation}
=\frac{1}{3}\sum_{k=1}^{3}|C_k-C_k^{\mathrm{nom}}|,
\]

reported in **percentage points**. Coverage summaries follow the artery-then-subject aggregation rule. The separate 95% predictive interval is \(\widehat\mu\pm1.96\widehat\sigma\), as specified in the manuscript; it is distinct from the two-standard-deviation interval.

Expected normalization error (ENE) compares empirical absolute standardized-residual quantiles with standard half-normal quantiles:

\[
|z|=\frac{|q-\widehat\mu|}{\widehat\sigma},
\qquad
H^{-1}(p_k)=\Phi^{-1}\left(\frac{1+p_k}{2}\right),
\]

\[
\mathrm{ENE}=\frac{1}{K}\sum_{k=1}^{K}
\left|\widehat Q_{|z|}(p_k)-H^{-1}(p_k)\right|.
\]

The same fixed set of quantile levels is used for all models and evaluation subsets:
\[
K=19,\qquad
p_k\in\{0.05,0.10,0.15,\ldots,0.90,0.95\}.
\]
Standardized residuals for calibration are formed after returning the predictive mean and standard deviation to physical flow units, using
\[
|z|=\frac{|q-\widehat\mu|}{\widehat\sigma+10^{-6}\,\mathrm{cm^3\,s^{-1}}}.
\]
The \(10^{-6}\,\mathrm{cm^3\,s^{-1}}\) term is a numerical safeguard only. Empirical quantiles are computed with NumPy 1.26.4 using

```python
p = np.arange(0.05, 1.00, 0.05)
q_emp = np.quantile(np.abs(z), p, method="linear")
```

so the interpolation method is explicitly **linear**. ENE quantiles are **not** computed separately for each artery and then averaged. For the global ENE, standardized residual samples from the complete evaluation scope are pooled before quantile calculation; for a territory-specific ENE, samples are pooled only within that predefined territory. In whole-subject bootstrap analyses, subjects are resampled first and ENE is recomputed from the pooled residual samples of each bootstrap resample. Lower ENE indicates better agreement between empirical standardized residuals and the Gaussian predictive distribution.

Sharpness is the mean predictive standard deviation:

\[
\mathrm{Sharpness}=\operatorname{mean}(\widehat\sigma).
\]

For all uncertainty summaries, standard deviation is averaged first over waveform samples within each artery, then across arteries within each virtual subject, and finally across subjects. Aleatoric and epistemic summaries average their respective **standard deviations**, rather than taking a square root after averaging variances.

### 13.3 OOD uncertainty inflation

For OOD scenario \(k\), uncertainty inflation is the ratio of mean epistemic standard deviation to its held-out in-distribution baseline:

\[
r_k^{\mathrm{epis}}
=\frac{\overline\sigma_{\mathrm{epis},k}^{\mathrm{OOD}}}
{\overline\sigma_{\mathrm{epis}}^{\mathrm{ID}}},
\qquad
\overline\sigma_{\mathrm{epis}}^{\mathrm{ID}}
=4.10\,\mathrm{cm^3\,s^{-1}}.
\]

Both means use the uncertainty aggregation rule above. This is a dimensionless ratio; values above one indicate greater between-model disagreement. The value `4.10` is the reported baseline; for new reproduction runs, compute the baseline from their corresponding held-out in-distribution predictions.

---

## 14. Software environment


| Component | Version / hardware |
|---|---|
| Python | **3.10.13** |
| PyTorch | **2.1.2** |
| NumPy | **1.26.4** |
| SciPy | **1.11.4** |
| CUDA | **12.1** |
| Operating system | **Ubuntu 22.04 LTS** |
| GPU | **NVIDIA RTX A5000, 24 GB** |


---

## 15. Checkpoint metadata


Each checkpoint is accompanied by a JSON/YAML record using the adopted metadata schema below. Run-dependent fields remain explicit placeholders until populated from the archived run:

```yaml
model_type: heteroscedastic
seed: 12345  # principal single-model run; ensemble members use 1001–1005
partition_seed: 42
loader_seed: 54321

input_points: 100
train_batch_size: 256
eval_batch_size: 512

optimizer: Adam
beta1: 0.9
beta2: 0.999
epsilon: 1.0e-8
weight_decay: 1.0e-5
gradient_clip_norm: 1.0

warmup_epochs: 50
warmup_lr: 1.0e-3
joint_max_epochs: 250
joint_lr: 3.0e-4

lr_scheduler:
  mode: min
  factor: 0.5
  patience_epochs: 15
  min_lr: 1.0e-6

early_stopping:
  metric: subject_averaged_validation_nll
  patience_epochs: 30

log_sigma_clamp:
  min: -3.0
  max: 1.0

sigma_numerical_floor: 1.0e-6

data_record: "zenodo:18539549"
```

---

## 16. The public archive structure

A practical layout is:

```text
repository/
├── README.md
├── REPRODUCIBILITY_PROTOCOL.md
├── environment.yml
├── requirements.txt
├── configs/
│   ├── deterministic.yaml
│   ├── heteroscedastic.yaml
│   ├── ensemble.yaml
│   └── ood.yaml
├── partitions/
│   └── primary_split_seed42.json
├── preprocessing/
│   └── normalization/
├── cohort_generation/
├── simulation/
├── training/
├── evaluation/
├── ood/
├── analysis/
└── checkpoints/
```

The corresponding Zenodo data record contains:

- retained virtual-subject parameter sets
- simulated area and flow waveforms
- exact partition identifiers
- normalization statistics
- inlet waveform/configuration
- OOD parameter sets and scenario labels
- retained model checkpoints
- analysis outputs required to regenerate manuscript tables/figures

---

## 17. Code and data identifiers

Current project identifiers supplied for the revision:

```text
Code:
https://github.com/AhmdG/Blood-Flow-Simulation-1D-Arterial-Network-Sovler

Data:
https://zenodo.org/records/18539549
```

For a final reproducible release, also record:

```
Zenodo record/version: `18539549`
```




### Release rule

Before release, inspect the protocol for unresolved placeholder tokens:

```bash
grep -niE 'TODO|TBD|PLACEHOLDER|FROM_ARCHIVED_RUN' REPRODUCIBILITY_PROTOCOL.md
```

Review each match. The illustrative metadata schema intentionally contains placeholders; these must be replaced with actual run values if the schema is presented as an archived run record. A `grep` exit status of 1 means no matches were found; it does not indicate a command error.

---

## 18. Interpretation


This protocol is intended to make the **synthetic computational study** reproducible. Reproducing the archived simulations and neural-network results does not constitute clinical validation, biological validation, or evidence of clinical utility.


---

