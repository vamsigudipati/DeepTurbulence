# DeepTurbulence — Developer Guide

> Baseline: current `master`-equivalent snapshot of the repository (no version tag or manifest is present). For contributors who need to read the source, extend the models, or reproduce the results of the accompanying paper.

---

## 1. Project positioning & scope

DeepTurbulence is a research code base that assesses whether deep neural networks can predict the *temporal* evolution of a low-order model of near-wall turbulence. It couples a MATLAB implementation of the nine-equation shear-flow model of Moehlis *et al.* (2004) — which produces ground-truth time series of nine modal amplitudes — with Python/Keras scripts that train and evaluate two families of networks (multilayer perceptron, MLP, and long short-term memory, LSTM) on those series. The instantaneous velocity field is reconstructed from the nine amplitudes as a superposition of nine Fourier modes,

$$\tilde{\mathbf{u}}(\mathbf{x}, t) = \sum_{j=1}^{9} a_j(t)\,\mathbf{u}_j(\mathbf{x}),$$

so learning the dynamics of the amplitude vector $a_j(t)$ is equivalent to learning the dynamics of the flow. The repository is the artifact behind "Predictions of turbulent shear flows using deep neural networks" (Srinivasan *et al.*, *Phys. Rev. Fluids* **4**, 054603, 2019).

Headline capabilities:
- Generate turbulent time series of the nine modal amplitudes by integrating the Moehlis ODE system at $\mathrm{Re}=400$ ([Data generator (Moehlis model)/moehlis_data_gen.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_data_gen.m)).
- Train MLP and LSTM regressors that predict the next amplitude state from the previous $p$ states ([Neural networks models/train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py), [Neural networks models/train_lstm_model.py](../../Neural%20networks%20models/train_lstm_model.py)).
- Autoregressively roll out new time series from a seed sequence using a trained model, and reconstruct/visualize the corresponding velocity fields ([Neural networks models/predict_using_lstm.py](../../Neural%20networks%20models/predict_using_lstm.py), [Data generator (Moehlis model)/visualize_fields.m](../../Data%20generator%20%28Moehlis%20model%29/visualize_fields.m)).

## 2. Metadata snapshot

- Package name: none — the repository is a collection of standalone scripts, not an installable package (no `setup.py`, `pyproject.toml`, or `package.json`).
- License: no `LICENSE` file is present in the repository.
- Language runtime: Python (CPython) with Keras for the neural networks; MATLAB for data generation (uses the built-in stiff solver `ode15s`).
- Core dependencies: `numpy`, `scipy.io` (`.mat` I/O), `keras` (`Sequential`, `Dense`, `LSTM`, `load_model`), `matplotlib`, and the Python standard `timeit`. MATLAB scripts require only base MATLAB.
- Version source: none. There is no version constant; provenance is the citation in each file's header docstring.

## 3. Repository layout

```
DeepTurbulence/
├── README.md                         # Project overview + links to paper & data
├── Data generator (Moehlis model)/   # MATLAB: ground-truth data generation
│   ├── moehlis_model_odefun.m        # The 9-ODE right-hand side (Galerkin system)
│   ├── moehlis_model_script.m        # Integrate ONE time series
│   ├── moehlis_data_gen.m            # Integrate MANY series -> moehlis_data_###.mat
│   ├── plot_amplitudes.m             # Plot the 9 amplitude time series
│   └── visualize_fields.m            # Reconstruct & plot 3D velocity fields
├── Neural networks models/           # Python/Keras: train + predict
│   ├── train_mlp_model.py            # Build/train an MLP -> ****.h5 + ****_loss.mat
│   ├── train_lstm_model.py           # Build/train an LSTM -> ****.h5 + ****_loss.mat
│   ├── predict_using_mlp.py          # Autoregressive rollout with a trained MLP
│   ├── predict_using_lstm.py         # Autoregressive rollout with a trained LSTM
│   └── trained_nn_models/            # Pre-trained *.h5 weights (MLP1-5, LSTM1-3)
└── docs/                             # Generated documentation & reference paper
```

## 4. Runtime architecture overview

```
      MATLAB (data generation)              Python / Keras (learning)
 ┌───────────────────────────────┐   ┌────────────────────────────────────┐
 │ moehlis_model_odefun.m        │   │ train_mlp_model.py                  │
 │   9-ODE Galerkin system       │   │ train_lstm_model.py                 │
 │            │                  │   │   build Sequential model            │
 │            ▼                  │   │   window data -> (X, Y)             │
 │ moehlis_data_gen.m  ──────────┼──▶│   model.fit(...)                    │
 │   ode15s integration          │   │            │                        │
 │   -> moehlis_data_###.mat     │   │            ▼                        │
 │      array (nTS, nTP, 9)      │   │   model.save(****.h5)               │
 └───────────────────────────────┘   │      + ****_loss.mat                │
                                      └───────────────┬────────────────────┘
                                                      │  trained_nn_models/*.h5
                                                      ▼
                                      ┌────────────────────────────────────┐
        moehlis_test_data_###.mat ───▶│ predict_using_{mlp,lstm}.py         │
                                      │   load_model(*.h5)                  │
                                      │   autoregressive one-step rollout   │
                                      │   -> series_#.mat (testSeq/predSeq) │
                                      └───────────────┬────────────────────┘
                                                      ▼
                                      visualize_fields.m / plot_amplitudes.m
```

Narrative: the pipeline has a strict left-to-right data-flow. MATLAB integrates the ODE system to produce a 3D array of shape `(nTS, nTP, 9)` (number of time series × time points × nine amplitudes), saved as a `.mat` file. The Python training scripts slide a window of length `seqLen` over each series to form supervised `(X, Y)` pairs, fit a Keras `Sequential` model, and persist it as an HDF5 `.h5` file plus a `_loss.mat` history. The prediction scripts reload the `.h5` model and roll it forward autoregressively — each predicted state is appended to the input window to predict the next — reproducing entire trajectories from only a seed of the first `seqLen` states. There is no shared library between the MATLAB and Python sides; the `.mat` file is the sole interface contract.

## 5. Entry-point scripts (no exported namespace)

There is no package `__init__.py` or export surface; the repository is driven by running scripts directly. The table below maps each script to its role, inputs, and outputs.

| Script | Role | Reads | Writes |
| --- | --- | --- | --- |
| [moehlis_data_gen.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_data_gen.m) | Generate training/test data | — | `moehlis_data_###.mat` |
| [moehlis_model_script.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_model_script.m) | Generate a single series + visualize | — | figures |
| [train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py) | Train an MLP | `moehlis_data_###.mat` | `<name>.h5`, `<name>_loss.mat` |
| [train_lstm_model.py](../../Neural%20networks%20models/train_lstm_model.py) | Train an LSTM | `moehlis_data_###.mat` | `<name>.h5`, `<name>_loss.mat` |
| [predict_using_mlp.py](../../Neural%20networks%20models/predict_using_mlp.py) | Roll out with an MLP | `moehlis_test_data_###.mat`, `<name>.h5` | `series_#.mat` |
| [predict_using_lstm.py](../../Neural%20networks%20models/predict_using_lstm.py) | Roll out with an LSTM | `moehlis_test_data_###.mat`, `<name>.h5` | `series_#.mat` |

## 6. Backend / platform abstraction

The repository has no multi-backend indirection layer of its own. The Python scripts import Keras directly (`from keras.models import Sequential, load_model`; `from keras.layers import Dense, LSTM`), so the numerical backend is whatever Keras is configured to use (TensorFlow in the environment used for the paper). Device placement, precision, and graph execution are delegated entirely to Keras/TensorFlow defaults — the code contains no explicit backend selection, `tf.device`, or precision configuration.

The only cross-platform "abstraction" is the on-disk `.mat` contract produced by MATLAB and consumed in Python through `scipy.io.loadmat`, e.g. in [train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py):

```python
dataStruct = sio.loadmat(dataFilename)
A = dataStruct['data']            # 3D array, shape (nTS, nTP, 9)
```

## 7. Data & domain layer

The data layer is the MATLAB Moehlis model. Through a Galerkin projection of the Navier–Stokes equations onto nine Fourier modes, the amplitudes $a_j(t)$ obey a system of nine coupled ODEs. That right-hand side is implemented verbatim in [moehlis_model_odefun.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_model_odefun.m), which returns the derivative vector `da` (9×1). For example, the first and last equations are:

```matlab
da(1) = B^2/Re - B^2/Re*a(1) ...
    - 1.5^0.5 *B*C /k3 *a(6)*a(8) ...
    + 1.5^0.5 *B*C /k2 *a(2)*a(3);
% ...
da(9) = -9*B^2/Re*a(9) ...
    + 1.5^0.5 *B*C /k2 *a(2)*a(3) ...
    - 1.5^0.5 *B*C /k3 *a(6)*a(8);
```

The wavenumbers derive from the domain geometry and Reynolds number, fixed in [moehlis_data_gen.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_data_gen.m):

```matlab
Re = 400;                 % model Reynolds number
Lx = 4*pi;  Lz = 2*pi;    % domain size
A = 2*pi/Lx;  B = pi/2;  C = 2*pi/Lz;
k1 = sqrt(A^2 + C^2);
k2 = sqrt(B^2 + C^2);
k3 = sqrt(A^2 + B^2 + C^2);
```

Data generation (`moehlis_data_gen.m`) integrates the system with the stiff solver `ode15s` from the initial condition `init = [1 0.07066 -0.07076 0 0 0 0 0 0]`, perturbing the fourth amplitude randomly (`init(4) = 0.1*rand`) to diversify trajectories. The first 99 time units are discarded to remove initial-condition transients, and any series that laminarizes is rejected before storage:

```matlab
% Check for laminarization and add to data matrix only if not
ind = find(abs(a_(:,1)-1) < 0.01, 1);
if isempty(ind)
    data(count,:,:) = a_;
    count = count + 1;
end
```

The domain data container is a single 3D array named `data` of shape `(nTS, nTP, 9)`, saved as `moehlis_data_<nTS>.mat`. The reconstruction of the physical velocity field from the nine amplitudes (used only for visualization/statistics) lives in [visualize_fields.m](../../Data%20generator%20%28Moehlis%20model%29/visualize_fields.m), which hard-codes the analytic form of each Fourier mode $\mathbf{u}_j(\mathbf{x})$ on a grid, e.g. `u1x = 2^0.5*sin(B*yp(j))` for the mean-profile mode.

The held-out `moehlis_test_data_###.mat` file required by both prediction scripts is produced by [moehlis_test_data_gen.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_test_data_gen.m), added alongside `moehlis_data_gen.m`. It runs the identical ODE integration and laminarization filter, differing only in two respects: it seeds the RNG explicitly (`rng(12345)`) so its perturbations never coincide with an unseeded training run of `moehlis_data_gen.m`, and it saves under the `moehlis_test_data_<nTS>.mat` name (with `nTS = 100` by default, matching the `dataFilename` already hard-coded in [predict_using_mlp.py](../../Neural%20networks%20models/predict_using_mlp.py) and [predict_using_lstm.py](../../Neural%20networks%20models/predict_using_lstm.py)).

## 8. Learning core (loss, back-propagation, optimization)

There is no hand-written automatic-differentiation code; gradient computation is delegated to Keras/TensorFlow. Both training scripts pin the objective to the mean squared error via `model.compile(optimizer='adam', loss='mean_squared_error')`. This is exactly the validation loss defined in the paper,

$$L(f(\mathbf{x}); \boldsymbol{\psi}) = \frac{1}{2m} \sum_{j=1}^{m} \lVert \psi^j - f(\chi^j) \rVert^2,$$

where $\psi$ and $f(\chi)$ are the $m$-dimensional true and predicted amplitude vectors ($m = 9$). Weights and biases are updated by back-propagation using **Adam** (adaptive moment estimation), an adaptive-learning-rate variant of stochastic gradient descent chosen in the paper for its suitability to deep networks. The public "API" is therefore just the Keras `compile`/`fit` calls in [train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py) and [train_lstm_model.py](../../Neural%20networks%20models/train_lstm_model.py).

## 9. Networks / models catalog

The two network families are built inline (not as reusable classes) inside the training scripts. Both use `glorot_normal` initialization — the Glorot–Bengio normal scheme the paper reports as the best-performing initializer — and a `tanh` output activation, matching the paper's finding that the hyperbolic tangent gave the best agreement with reference data.

| Model | Path | Construction (default settings in file) |
| --- | --- | --- |
| MLP | [train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py) | `Sequential` of `Dense(90, activation='tanh', kernel_initializer='glorot_normal')` layers; `arch = [seqLen*9] + [90,90,90,90] + [9]`, `seqLen=500` |
| LSTM | [train_lstm_model.py](../../Neural%20networks%20models/train_lstm_model.py) | `LSTM(90, input_shape=(seqLen, 9), kernel_initializer='glorot_normal')` + `Dense(9, activation='tanh')`; `seqLen=10`; supports 1 or 2 stacked LSTM layers |

MLP construction:

```python
model = Sequential()
for i in range(len(hiddenLayers)):
    model.add(Dense(arch[i+1], input_dim=arch[i], activation='tanh',
                    kernel_initializer='glorot_normal'))
model.add(Dense(arch[-1], activation='tanh',
                kernel_initializer='glorot_normal'))
```

LSTM construction (the file supports only 1 or 2 layers via `return_sequences`):

```python
if len(lstmLayers) == 1:
    model.add(LSTM(lstmLayers[0], input_shape=(seqLen, 9),
                   kernel_initializer='glorot_normal', return_sequences=False))
else:
    model.add(LSTM(lstmLayers[0], input_shape=(seqLen, 9),
                   kernel_initializer='glorot_normal', return_sequences=True))
    model.add(LSTM(lstmLayers[1], kernel_initializer='glorot_normal',
                   return_sequences=False))
model.add(Dense(9, activation='tanh'))
```

The key architectural difference reflects the paper's central result: the MLP flattens `seqLen*9 = 4500` inputs into a point-prediction problem, whereas the LSTM consumes an explicit `(seqLen, 9)` sequence and exploits its recurrent connections to model temporal dependencies — which is why the LSTM achieves comparable or better accuracy with an input dimension up to 50× smaller ($p=10$ vs. $p=500$).

Pre-trained weights shipped in [trained_nn_models/](../../Neural%20networks%20models/trained_nn_models):

| File | Architecture | Training sets |
| --- | --- | --- |
| `MLP1_t1000.h5` … `MLP5_t1000.h5` | MLP variants (l, n per paper Table II) | 1000 |
| `LSTM1_t100.h5`, `LSTM1_t1000.h5`, `LSTM1_t10000.h5` | single-layer LSTM | 100 / 1000 / 10000 |
| `LSTM2_t100.h5` | two-layer LSTM | 100 |
| `LSTM3_t100.h5` | single-layer LSTM, longer input | 100 |

## 10. Optimizers & schedulers

- Optimizer: **Adam** only, selected via the string identifier `'adam'` in `model.compile(...)`. No optimizer object is instantiated, so Keras's default Adam learning rate is used.
- Learning-rate schedule: none is configured in code. The paper notes Adam's adaptive learning rate makes a manual decay schedule unnecessary; there is no `LearningRateScheduler` or decay argument in either training script.
- Early stopping: the paper describes early stopping to avoid overfitting, but the code does **not** register a Keras `EarlyStopping` callback. Training runs for a fixed `nbEpochs` (10 in both files) and reserves 20% of samples for validation via `validation_split=0.20`.

## 11. Training main loop

Both training scripts share the same linear structure (illustrated with [train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py); the LSTM version differs only in the input tensor shape).

1. **Configure** — set `seqLen` (the prediction order $p$), the layer widths, `nbEpochs`, the input `dataFilename`, and the output `saveFilename`.
2. **Build or load** — if `train_more_epochs` is `True`, resume with `load_model(saveFilename + '.h5')`; otherwise build a fresh `Sequential` model and `compile` it with Adam + MSE.
3. **Window the data** — load the `(nTS, nTP, 9)` array and slide a window of length `seqLen` to build supervised pairs. For the MLP the window is flattened to `seqLen*9`; for the LSTM it stays `(seqLen, 9)`:

   ```python
   nSamples = A.shape[0]*(A.shape[1]-seqLen)
   X = np.empty([nSamples, seqLen*A.shape[2]])   # LSTM: [nSamples, seqLen, 9]
   Y = np.empty([nSamples, A.shape[2]])
   k = 0
   for i in np.arange(A.shape[0]):
       for j in np.arange(A.shape[1]-seqLen):
           X[k] = A[i, j:j+seqLen].reshape(1,-1) # LSTM: A[i, j:j+seqLen]
           Y[k] = A[i, j+seqLen].reshape(1,-1)
           k = k + 1
   ```

4. **Fit** — `model.fit(X, Y, batch_size=32, epochs=nbEpochs, verbose=1, validation_split=0.20, shuffle=True)`; the returned `score.history` holds `loss` and `val_loss`.
5. **Persist** — `model.save(saveFilename + '.h5')`; the loss/validation-loss histories are written to `saveFilename + '_loss.mat'` via `sio.savemat`. When resuming, the new history is concatenated onto the previously saved one.

Prediction (rollout) in [predict_using_mlp.py](../../Neural%20networks%20models/predict_using_mlp.py) / [predict_using_lstm.py](../../Neural%20networks%20models/predict_using_lstm.py): after `load_model(...)`, the first `seqLen` states of a test series seed the prediction, then each new state is appended and fed back to predict the next — a closed-loop autoregression:

```python
predSeq = testSeq[:, :seqLen]                       # LSTM seed
for i in np.arange(testSeq.shape[1] - seqLen):
    nextState = model.predict(predSeq[:, i:i+seqLen], verbose=0)
    predSeq = np.concatenate((predSeq, [nextState]), axis=1)
```

The reference and predicted series are saved together as `series_#.mat` (`testSeq`, `predSeq`) for downstream statistics/visualization.

## 12. Callback / hook system

No Keras callbacks are registered. The only lifecycle hook used is the `History` object returned by `model.fit`, from which `score.history['loss']` and `score.history['val_loss']` are extracted and persisted:

```python
score = model.fit(X, Y, batch_size=32, epochs=nbEpochs,
                  verbose=1, validation_split=0.20, shuffle=True)
lossHistory = score.history['loss']
valLossHistory = score.history['val_loss']
```

| Hook | Trigger | Use case |
| --- | --- | --- |
| `History` (implicit) | end of `model.fit` | capture `loss` / `val_loss` to write `_loss.mat` |

Early stopping is described in the paper (Fig. 2) as the mechanism used to avoid overfitting, but no `EarlyStopping` callback is registered in either training script — `nbEpochs` always runs to completion. A contributor adding it would import `EarlyStopping` from `keras.callbacks` and pass it through `model.fit`'s `callbacks` argument, monitoring `val_loss`:

```python
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
score = model.fit(X, Y, batch_size=32, epochs=nbEpochs, verbose=1,
                  validation_split=0.20, shuffle=True, callbacks=[early_stop])
```

This is intentionally left as an extension point rather than a default in `train_mlp_model.py`/`train_lstm_model.py` (see also `tutorial.md` Chapter 10, extension 1), since those scripts reproduce the exact configuration used to produce the paper's published results.

## 13. Loss & metric registries

- Loss: a single string identifier `'mean_squared_error'`, passed to `model.compile` in both training scripts. This is the objective $L$ in §8.
- Metric during training: the validation loss `val_loss`, produced automatically by `validation_split=0.20` and saved to `<name>_loss.mat`.
- Evaluation metrics (post-processing, in the paper, not in the repo code): the amplitude relative error
  $$\varepsilon_1 = \frac{1}{(N_s - p)\,a_{1,\text{lam}}} \sum_{j=p+1}^{N_s} \lvert a_{1,\text{tra}}^j - a_{1,\text{pred}}^j \rvert,$$
  and the mean-flow relative error $E_{\bar u}$ (paper Eq. 4). These are computed from the saved `series_#.mat` files.

Both metrics are now computed by [compute_turbulence_statistics.py](../../Neural%20networks%20models/compute_turbulence_statistics.py), added alongside the prediction scripts. It loads a batch of `series_#.mat` files (produced by `predict_using_mlp.py`/`predict_using_lstm.py`) and reports $\varepsilon_1$ directly from the `a_1` columns of `testSeq`/`predSeq`, and $E_{\bar u}$ by reconstructing the mean streamwise profile analytically from amplitudes $a_1$ and $a_9$ alone — the only two Fourier modes without an $x$ or $z$ dependence, so every other mode integrates to zero over the periodic directions (see `visualize_fields.m`). It deliberately does **not** compute $E_{u'^2}$, the Reynolds shear stress, skewness, or flatness, since those require reconstructing the full 3D velocity field from all nine modes and evaluating cross-mode integrals — a larger undertaking left as a further extension.

```bash
python compute_turbulence_statistics.py LSTM1_t100_ps 10
# Series evaluated: 10
# eps_1  (mean over series) : <value> %
# E_ubar (ensemble-averaged): <value> %
```

## 14. Parallel / distributed training

No parallel or distributed-training code is present. Training is single-process `model.fit`, and any GPU acceleration is whatever the underlying Keras/TensorFlow install provides automatically; there is no `MirroredStrategy`, multi-GPU, or MPI logic in the repository. The paper does note that DNN training benefits from GPUs in general, but the checked-in scripts make no explicit device or parallelism choice.

## 15. Contribution SOPs & debugging

### 15.1 Dev environment

```bash
# Python side (training + prediction)
pip install numpy scipy matplotlib keras tensorflow

# MATLAB side (data generation): base MATLAB with ode15s is sufficient.
```

### 15.2 Adding a new network architecture (template)

1. Copy [train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py) or [train_lstm_model.py](../../Neural%20networks%20models/train_lstm_model.py) and edit the architecture arrays (`hiddenLayers` / `lstmLayers`) and `seqLen` (the prediction order $p$). Keep the output layer at `Dense(9, ...)` — nine amplitudes.
2. Keep the input/output windowing loop consistent with the model's expected shape: `seqLen*9` (flat) for a dense front-end, or `(seqLen, 9)` for a recurrent one.
3. Set a unique `saveFilename` so the `.h5` and `_loss.mat` outputs do not overwrite existing trained models, then run the script.

### 15.3 Common error → root cause

| Symptom | Check |
| --- | --- |
| `KeyError: 'data'` from `sio.loadmat` | the `.mat` file must contain a variable named `data` (see `moehlis_data_gen.m`'s `save(... , 'data')`) |
| Shape mismatch in `model.fit` / `model.predict` | `seqLen` in the training and prediction scripts must match the `seqLen` baked into the saved `.h5` model |
| `FileNotFoundError` when saving `series_#.mat` | the output folder (e.g. `MLP3_t100_ps/`, `LSTM1_t100_ps/`) must be created before running the prediction script |
| Predictions collapse to the laminar state | seed series must be turbulent; `moehlis_data_gen.m` already rejects laminarized series (`abs(a_(:,1)-1) < 0.01`) |

---

## References

- Main paper: P. A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa, "Predictions of turbulent shear flows using deep neural networks", *Phys. Rev. Fluids* **4**, 054603 (2019). DOI: [10.1103/PhysRevFluids.4.054603](https://doi.org/10.1103/PhysRevFluids.4.054603); preprint [arXiv:1905.03634](https://arxiv.org/abs/1905.03634).
- Underlying model: J. Moehlis, H. Faisst, B. Eckhardt, "A low-dimensional model for turbulent shear flows", *New J. Phys.* **6**, 56 (2004). DOI: [10.1088/1367-2630/6/1/056](https://doi.org/10.1088/1367-2630/6/1/056).
- Local reference copy: [docs/references/Predictions of turbulent shear flows using deep neural networks/](../references/).
