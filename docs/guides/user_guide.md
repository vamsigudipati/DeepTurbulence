# DeepTurbulence — User Guide

> For users who want to generate turbulence data and train/run the MLP or LSTM predictors, without reading the full [developer guide](./developer_guide.md).

---

## 1. What DeepTurbulence can do

- Generate ground-truth turbulent time series for the nine-mode Moehlis shear-flow model via MATLAB ([Data generator (Moehlis model)/moehlis_data_gen.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_data_gen.m)).
- Train a multilayer perceptron (MLP) or a long short-term memory (LSTM) network in Python/Keras to predict the next state of the nine amplitudes from a window of previous states ([Neural networks models/train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py), [Neural networks models/train_lstm_model.py](../../Neural%20networks%20models/train_lstm_model.py)).
- Use a trained model (either one you trained or one of the pre-trained `.h5` files under [trained_nn_models/](../../Neural%20networks%20models/trained_nn_models)) to autoregressively predict a full new time series from a short seed, and save it for comparison against the reference ([Neural networks models/predict_using_mlp.py](../../Neural%20networks%20models/predict_using_mlp.py), [Neural networks models/predict_using_lstm.py](../../Neural%20networks%20models/predict_using_lstm.py)).
- Visualize the nine amplitude time series or reconstruct the 3D velocity field from them ([Data generator (Moehlis model)/plot_amplitudes.m](../../Data%20generator%20%28Moehlis%20model%29/plot_amplitudes.m), [Data generator (Moehlis model)/visualize_fields.m](../../Data%20generator%20%28Moehlis%20model%29/visualize_fields.m)).

## 2. Installation

DeepTurbulence is not a packaged library — there is no `setup.py`, `pyproject.toml`, or `requirements.txt` in the repository. You run the MATLAB and Python scripts directly from a local clone.

| Component | Requirement |
| --- | --- |
| Data generation | MATLAB (base install; uses only the built-in `ode15s` stiff ODE solver — no toolboxes required) |
| Training / prediction | Python 3 with `numpy`, `scipy`, `matplotlib`, `keras`, and a Keras backend (`tensorflow`) |

Install the Python dependencies with pip:

```bash
pip install numpy scipy matplotlib keras tensorflow
```

The repository does not pin exact package versions, and the scripts use the standalone `keras` import style (`from keras.models import Sequential, load_model`) rather than `tensorflow.keras`. That style matches the multi-backend `keras` PyPI package (2.2.x-2.3.x) from before the Keras/TensorFlow 2 unification, paired with a TensorFlow 1.x backend (e.g. `tensorflow==1.12`-`1.15`) — the safest bet if you hit API incompatibilities on a fresh install, matching the 2018-2019 era of the accompanying paper. If you prefer a current environment instead, replace the two `keras` imports in each script with `from tensorflow.keras.models import ...` / `from tensorflow.keras.layers import ...` and install only `tensorflow` (2.x ships `tf.keras`, which is API-compatible with these scripts).

No further build step is required — clone the repository and run the scripts from within their own folders, since each script references data/model files by bare filename (e.g. `"moehlis_data_100.mat"`), not by absolute path.

## 3. Choosing a backend / runtime

Not applicable. There is no backend-selection mechanism in this repository — the Python scripts import Keras directly and rely on whichever backend (TensorFlow) is installed. The MATLAB side runs on any local MATLAB installation; there is no notion of a distributed or cloud runtime.

## 4. Global configuration

There is no global configuration object. Each script exposes its own settings as plain module-level variables in a `Settings` block near the top of the file, which you edit directly before running. Example, from [train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py):

```python
# Prediction order (p), length of the sequence used for prediction of the next state.
seqLen = 500

# The MLP architecture in an array (each entry = units in that hidden layer)
hiddenLayers = [90, 90, 90, 90]

# True to resume training an existing <saveFilename>.h5, False to train from scratch
train_more_epochs = False

# Number of epochs for training
nbEpochs = 10

# Input data file (must contain a 'data' array of shape (nTS, nTP, 9))
dataFilename = "moehlis_data_100.mat"

# Output model/loss filename stem
saveFilename = "MLP3_t100"
```

Key knobs (same names apply to [train_lstm_model.py](../../Neural%20networks%20models/train_lstm_model.py), where `hiddenLayers` becomes `lstmLayers` and supports only 1 or 2 entries):
- `seqLen` — the prediction order $p$: how many previous states are fed in to predict the next one. Large for the MLP (500 in the shipped default), small for the LSTM (10).
- `dataFilename` / `saveFilename` — the input `.mat` data file and the output `.h5`/`.mat` name stem; change `saveFilename` before every new run to avoid overwriting existing trained models.

## 5. Core-object map

There are no user-facing classes; the "objects" a user interacts with are files passed between MATLAB and Python:

```
 moehlis_data_gen.m  --> moehlis_data_<nTS>.mat  (data['data']: (nTS, nTP, 9))
                                  |
                                  v
        train_{mlp,lstm}_model.py  --> <name>.h5  +  <name>_loss.mat
                                  |
                                  v
 moehlis_test_data_<nTS>.mat + <name>.h5
                                  |
                                  v
        predict_using_{mlp,lstm}.py  --> series_#.mat (testSeq, predSeq)
                                  |
                                  v
        plot_amplitudes.m / visualize_fields.m
```

## 6. Geometry / domain definition

There is no geometry API. The physical domain (a channel of size $L_x = 4\pi$, $L_z = 2\pi$) and the model Reynolds number $\mathrm{Re}=400$ are hard-coded constants at the top of [moehlis_data_gen.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_data_gen.m) and [moehlis_model_script.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_model_script.m):

```matlab
Re = 400;
Lx = 4*pi;  Lz = 2*pi;
A = 2*pi/Lx;  B = pi/2;  C = 2*pi/Lz;
```

To change the domain or Reynolds number, edit these constants directly in the MATLAB script and regenerate the data; there is no separate configuration file.

## 7. Boundary / initial conditions

The nine-mode ODE system takes a single initial-condition vector, set the same way in both [moehlis_data_gen.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_data_gen.m) and [moehlis_model_script.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_model_script.m):

```matlab
init = [1 0.07066 -0.07076 0 0 0 0 0 0];
init(4) = 0.1*rand;   % moehlis_data_gen.m perturbs a_4 randomly per series
% init(4) = 1e-4;     % moehlis_model_script.m uses a fixed small perturbation
```

`moehlis_data_gen.m` also discards a generated series if it laminarizes (`abs(a_(:,1)-1) < 0.01`) instead of turning turbulent, so every saved series is guaranteed turbulent.

## 8. Data objects

| File pattern | Produced by | Shape / fields | Used by |
| --- | --- | --- | --- |
| `moehlis_data_<nTS>.mat` | [moehlis_data_gen.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_data_gen.m) | variable `data`, shape `(nTS, nTP, 9)` | `train_mlp_model.py`, `train_lstm_model.py` (training set) |
| `moehlis_test_data_<nTS>.mat` | [moehlis_test_data_gen.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_test_data_gen.m) | variable `data`, shape `(nTS, nTP, 9)` | `predict_using_mlp.py`, `predict_using_lstm.py` (seed + reference) |
| `<name>.h5` | `train_mlp_model.py` / `train_lstm_model.py` | Keras `Sequential` model (weights + architecture) | `predict_using_*.py` via `load_model` |
| `<name>_loss.mat` | `train_mlp_model.py` / `train_lstm_model.py` | `lossHistory`, `valLossHistory` arrays | inspecting training convergence |
| `series_#.mat` | `predict_using_mlp.py` / `predict_using_lstm.py` | `testSeq`, `predSeq`, each `(nTP, 9)` | `plot_amplitudes.m`, `visualize_fields.m`, `compute_turbulence_statistics.py` |

[moehlis_test_data_gen.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_test_data_gen.m) runs the same ODE integration and laminarization filter as `moehlis_data_gen.m`, differing only in an explicit RNG seed (so it never overlaps with a training run) and the `moehlis_test_data_<nTS>.mat` output name (`nTS = 100` by default, matching the `dataFilename` already hard-coded in the prediction scripts).

## 9. PDE residual / constraint authoring

Not applicable. The networks are trained with plain supervised regression (mean squared error between predicted and true next-state amplitudes); there is no physics-informed residual, automatic-differentiation constraint, or PDE loss term authored in this repository. The physics only enters through the MATLAB-generated training data itself.

## 10. Networks

Both networks are built inline in the training scripts, not exposed as reusable classes. MLP, from [train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py):

```python
model = Sequential()
for i in range(len(hiddenLayers)):
    model.add(Dense(arch[i+1], input_dim=arch[i], activation='tanh',
                    kernel_initializer='glorot_normal'))
model.add(Dense(arch[-1], activation='tanh', kernel_initializer='glorot_normal'))
model.compile(optimizer='adam', loss='mean_squared_error')
```

LSTM, from [train_lstm_model.py](../../Neural%20networks%20models/train_lstm_model.py):

```python
model = Sequential()
model.add(LSTM(lstmLayers[0], input_shape=(seqLen, 9),
               kernel_initializer='glorot_normal', return_sequences=False))
model.add(Dense(9, activation='tanh'))
model.compile(optimizer='adam', loss='mean_squared_error')
```

Built-ins: dense MLP (`Dense` stack) and single- or two-layer LSTM (`LSTM` + `Dense`). Pre-trained variants of both are shipped in [trained_nn_models/](../../Neural%20networks%20models/trained_nn_models) (`MLP1`–`MLP5`, `LSTM1`–`LSTM3`).

## 11. Training

To train a model, edit the settings block described in §4, then run the script from within the [Neural networks models/](../../Neural%20networks%20models) folder (so the relative `dataFilename` resolves):

```bash
cd "Neural networks models"
python train_mlp_model.py     # or: python train_lstm_model.py
```

Internally the script:

```python
dataStruct = sio.loadmat(dataFilename)      # load moehlis_data_###.mat
A = dataStruct['data']                      # (nTS, nTP, 9)
# ... slide a window of length seqLen over A to build (X, Y) ...
score = model.fit(X, Y, batch_size=32, epochs=nbEpochs,
                  verbose=1, validation_split=0.20, shuffle=True)
model.save(saveFilename + '.h5')
sio.savemat(saveFilename + '_loss',
    {'lossHistory': lossHistory, 'valLossHistory': valLossHistory})
```

Key options: `seqLen` (prediction order $p$), `hiddenLayers`/`lstmLayers` (network width/depth), `nbEpochs`, `train_more_epochs` (resume training from `<saveFilename>.h5` instead of starting fresh), `dataFilename`, `saveFilename`. There is no command-line argument parsing — all options are set by editing the script.

## 12. Inverse problems

Not applicable. This repository only performs forward time-series prediction (given past amplitudes, predict future ones); it does not solve inverse/parameter-identification problems.

## 13. Operator learning

Not applicable. There is no operator-learning (e.g. DeepONet-style) architecture in this repository; both networks map a fixed-length window of past states to the next state.

## 14. Uncertainty quantification & multifidelity

Not applicable. Training and prediction are deterministic single-model runs; no ensembling, Bayesian, or multifidelity mechanism is implemented.

## 15. Persistence

Trained models are saved and reloaded as standard Keras HDF5 files:

```python
# Save (end of train_{mlp,lstm}_model.py)
model.save(saveFilename + '.h5')

# Reload for further training
if train_more_epochs:
    model = load_model(saveFilename + '.h5')

# Reload for prediction (predict_using_{mlp,lstm}.py)
model = load_model('MLP3_t100.h5')
```

Loss history is stored separately as `<name>_loss.mat` (`lossHistory`, `valLossHistory` arrays); when `train_more_epochs` is `True`, the new epoch history is concatenated onto the previously saved one rather than overwriting it.

## 16. Visualization & post-processing

After running a prediction script, use the MATLAB helpers to inspect results:

- [plot_amplitudes.m](../../Data%20generator%20%28Moehlis%20model%29/plot_amplitudes.m) — plots the nine amplitude time series `a_1`…`a_9` from a `(nTP, 9)` array (e.g. `testSeq` or `predSeq` loaded from a `series_#.mat` file).
- [visualize_fields.m](../../Data%20generator%20%28Moehlis%20model%29/visualize_fields.m) — reconstructs and visualizes the 3D velocity field on a grid from the nine amplitudes at a chosen time index (`start_t`).

```matlab
load('series_1.mat')          % loads testSeq, predSeq
plot_amplitudes(predSeq)
visualize_fields(predSeq)
```

Both `predict_using_mlp.py` and `predict_using_lstm.py` `import matplotlib.pyplot as plt` but never call a plotting function — all current plotting is done from MATLAB on the saved `series_#.mat` files, as shown above. For a quantitative (rather than visual) check, use [compute_turbulence_statistics.py](../../Neural%20networks%20models/compute_turbulence_statistics.py) to compute the paper's relative-error metrics directly from a folder of `series_#.mat` files:

```bash
python compute_turbulence_statistics.py LSTM1_t100_ps 10
```

## 17. Parallel training

Not applicable. Training runs as a single-process `model.fit` call; there is no multi-GPU, multi-node, or distributed-strategy code in the repository.

## 18. Troubleshooting

| Symptom | Suggestion |
| --- | --- |
| `KeyError: 'data'` when loading a `.mat` file | The file must contain a variable literally named `data` (produced by `save(..., 'data')` in `moehlis_data_gen.m`); check you passed the right file. |
| Shape mismatch in `model.fit` / `model.predict` | `seqLen` used at prediction time must match the `seqLen` the loaded `.h5` model was trained with. |
| `FileNotFoundError` when a prediction script saves `series_#.mat` | Create the output folder first (e.g. `MLP3_t100_ps/`, `LSTM1_t100_ps/`) — the scripts do not create it automatically. |
| Training from scratch overwrites an old run | Set a new `saveFilename` before running; `train_more_epochs = False` always builds a fresh model and will overwrite an existing `<saveFilename>.h5`. |
| `import keras` fails | Install a Keras/TensorFlow version pair compatible with the standalone `keras.models` / `keras.layers` import style used in these scripts (see §2 TODO). |

## 19. Minimal runnable templates

### 19.1 Generate data and train an MLP

```bash
# 1. In MATLAB, from Data generator (Moehlis model)/:
%   edit nTS, nTP in moehlis_data_gen.m if desired, then run it
%   -> moehlis_data_10.mat

# 2. Copy/point train_mlp_model.py's dataFilename at that file, then:
cd "Neural networks models"
python train_mlp_model.py
# -> MLP3_t100.h5, MLP3_t100_loss.mat (names per saveFilename in the script)
```

### 19.2 Predict a new series with a trained LSTM

```bash
# Requires moehlis_test_data_100.mat and LSTM1_t100.h5 in the working directory
# (a pre-trained copy of LSTM1_t100.h5 ships in trained_nn_models/)
mkdir -p LSTM1_t100_ps
cd "Neural networks models"
python predict_using_lstm.py
# -> LSTM1_t100_ps/series_1.mat ... series_10.mat (testSeq, predSeq)
```

## 20. Further reading

- Developer guide: [developer_guide.md](./developer_guide.md) — architecture, training-loop internals, and contribution SOPs.
- Repository topology: [topology.md](./topology.md)
- Main paper: P. A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa, "Predictions of turbulent shear flows using deep neural networks", *Phys. Rev. Fluids* **4**, 054603 (2019). DOI: [10.1103/PhysRevFluids.4.054603](https://doi.org/10.1103/PhysRevFluids.4.054603); local copy in [docs/references/](../references/).
