# DeepTurbulence — From Beginner to Expert Tutorial

> Ten chapters, progressively deeper. Every chapter is anchored in at least one real script from the repository — no invented APIs.

## Table of contents

- Chapter 1 — Environment setup & your first time series
- Chapter 2 — Understanding the nine-mode turbulence model
- Chapter 3 — Generating a full training dataset
- Chapter 4 — Preparing windowed data for supervised learning
- Chapter 5 — Training your first MLP predictor
- Chapter 6 — Training an LSTM predictor
- Chapter 7 — Predicting new turbulent time series
- Chapter 8 — Visualizing predictions and reconstructed flow fields
- Chapter 9 — Improving accuracy: architectures, epochs, and turbulence statistics
- Chapter 10 — Mastery: extending DeepTurbulence

---

## Chapter 1 · Environment setup & your first time series

Install the two toolchains this repository needs: MATLAB (for data generation, no toolboxes beyond the built-in `ode15s`) and Python with Keras (for training/prediction).

```bash
pip install numpy scipy matplotlib keras tensorflow
```

Your first runnable script is [Data generator (Moehlis model)/moehlis_model_script.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_model_script.m), which integrates the nine-mode ODE system for a single trajectory and visualizes it:

```matlab
Re = 400;
Lx = 4*pi;  Lz = 2*pi;
init = [1 0.07066 -0.07076 0 0 0 0 0 0];
init(4) = 1e-4;
[t, a_] = ode15s(@(t,a) moehlis_model_odefun(t,a,Re), 0:1:4099, init);
a_ = a_(end-3999:end, :);   % drop the first 100 time units (initial-condition transient)
visualize_fields(a_)
```

Run it directly in MATLAB (`Current Folder` set to [Data generator (Moehlis model)/](../../Data%20generator%20%28Moehlis%20model%29)):

```matlab
moehlis_model_script
```

Expected result: a plotted 3D reconstruction of the turbulent velocity field at time index `start_t = 404`, produced from the nine amplitudes `a_` by [visualize_fields.m](../../Data%20generator%20%28Moehlis%20model%29/visualize_fields.m).

**Exercise** — Edit `moehlis_model_script.m` to call `plot_amplitudes(a_)` instead of `visualize_fields(a_)` and observe the nine amplitude time series `a_1`…`a_9` directly.

---

## Chapter 2 · Understanding the nine-mode turbulence model

The physics behind every dataset in this repository is the nine-equation Galerkin model of Moehlis *et al.* (2004): the instantaneous velocity field is a superposition of nine fixed Fourier modes $\mathbf{u}_j(\mathbf{x})$ weighted by time-dependent amplitudes $a_j(t)$,

$$\tilde{\mathbf{u}}(\mathbf{x}, t) = \sum_{j=1}^{9} a_j(t)\,\mathbf{u}_j(\mathbf{x}).$$

The amplitudes evolve according to nine coupled ODEs implemented in [moehlis_model_odefun.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_model_odefun.m). Its first equation shows the pattern every other mode follows — quadratic coupling terms scaled by the domain wavenumbers `k1`, `k2`, `k3`:

```matlab
da(1) = B^2/Re - B^2/Re*a(1) ...
    - 1.5^0.5 *B*C /k3 *a(6)*a(8) ...
    + 1.5^0.5 *B*C /k2 *a(2)*a(3);
```

The wavenumbers and the model Reynolds number are set once in the generator scripts:

```matlab
Re = 400;
Lx = 4*pi;  Lz = 2*pi;
A = 2*pi/Lx;  B = pi/2;  C = 2*pi/Lz;
k1 = sqrt(A^2 + C^2);  k2 = sqrt(B^2 + C^2);  k3 = sqrt(A^2 + B^2 + C^2);
```

Everything downstream — the training data, the trained networks, the predictions — is ultimately a statement about this ODE system: neural networks are being trained to mimic `moehlis_model_odefun.m` without ever seeing its equations, using only the amplitude sequences it produces.

**Exercise** — Read `moehlis_model_odefun.m` end to end and identify which of the nine equations has the fewest coupling terms (i.e., depends on the fewest other amplitudes).

---

## Chapter 3 · Generating a full training dataset

A single time series (Chapter 1) is not enough to train a network. [moehlis_data_gen.m](../../Data%20generator%20%28Moehlis%20model%29/moehlis_data_gen.m) integrates many independent series and stores them together:

```matlab
nTS = 10;     % number of time series
nTP = 4000;   % number of time points per series
data = zeros(nTS, nTP, 9);

count = 1;
while count <= nTS
    init(4) = 0.1*rand;                         % randomize the perturbation each run
    [t, a_] = ode15s(@(t,a) moehlis_model_odefun(t,a,Re), 0:1:nTP+99, init);
    a_ = a_(end-nTP+1:end, :);

    ind = find(abs(a_(:,1)-1) < 0.01, 1);        % reject laminarized series
    if isempty(ind)
        data(count,:,:) = a_;
        count = count + 1;
    end
end
save(['./moehlis_data_' num2str(nTS) '.mat'], 'data')
```

Two details matter for later chapters. First, the perturbation `init(4) = 0.1*rand` is re-drawn for every series, which is what makes the `nTS` trajectories statistically independent. Second, the laminarization check discards any series whose first amplitude returns to the laminar value $a_1 \approx 1$ — every saved series is guaranteed to stay turbulent for its full 4000 time units. The output, `moehlis_data_<nTS>.mat`, holds a single 3D array `data` of shape `(nTS, nTP, 9)` — this is the exact file the Python training scripts expect (Chapters 5–6).

Run it in MATLAB from [Data generator (Moehlis model)/](../../Data%20generator%20%28Moehlis%20model%29):

```matlab
moehlis_data_gen
```

**Exercise** — Set `nTS = 100` and regenerate the data. Run the script a second time with a different output filename (e.g. rename the saved `.mat` file) to produce a held-out set you can later use as `moehlis_test_data_100.mat` in Chapter 7.

---

## Chapter 4 · Preparing windowed data for supervised learning

Both `moehlis_data_<nTS>.mat` files hold raw trajectories, but the networks are trained as **one-step predictors**: given the previous `seqLen` states, predict the next one. This windowing happens identically at the top of [train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py) and [train_lstm_model.py](../../Neural%20networks%20models/train_lstm_model.py):

```python
dataStruct = sio.loadmat(dataFilename)
A = dataStruct['data']                       # (nTS, nTP, 9)

nSamples = A.shape[0]*(A.shape[1]-seqLen)
X = np.empty([nSamples, seqLen*A.shape[2]])  # LSTM keeps the 3D shape: [nSamples, seqLen, 9]
Y = np.empty([nSamples, A.shape[2]])

k = 0
for i in np.arange(A.shape[0]):
    for j in np.arange(A.shape[1]-seqLen):
        X[k] = A[i, j:j+seqLen].reshape(1,-1)   # LSTM: A[i, j:j+seqLen] (no flattening)
        Y[k] = A[i, j+seqLen].reshape(1,-1)
        k = k + 1
```

`seqLen` is the prediction order $p$ from the paper: the MLP defaults to $p=500$ (a large input window flattened to `500*9 = 4500` features), while the LSTM defaults to $p=10$ and keeps the window as an explicit `(seqLen, 9)` sequence — this difference is why the two architectures are configured so differently in the next two chapters.

**Exercise** — For a dataset with `nTS=10` and `nTP=4000`, compute `nSamples` by hand for `seqLen=500` (MLP) and `seqLen=10` (LSTM), and explain why a smaller `seqLen` yields more training samples from the same raw data.

---

## Chapter 5 · Training your first MLP predictor

With `moehlis_data_10.mat` (or the larger set from Chapter 3's exercise) in hand, open [train_mlp_model.py](../../Neural%20networks%20models/train_mlp_model.py) and check the settings block:

```python
seqLen = 500
hiddenLayers = [90, 90, 90, 90]     # four hidden layers of 90 units each
arch = [seqLen*9] + hiddenLayers + [9]
nbEpochs = 10
dataFilename = "moehlis_data_100.mat"
saveFilename = "MLP3_t100"
```

The model itself is a plain Keras `Sequential` stack of `tanh` layers with Glorot-normal initialization — the initializer and activation the paper found to perform best — trained with Adam against mean squared error:

```python
model = Sequential()
for i in range(len(hiddenLayers)):
    model.add(Dense(arch[i+1], input_dim=arch[i], activation='tanh',
                    kernel_initializer='glorot_normal'))
model.add(Dense(arch[-1], activation='tanh', kernel_initializer='glorot_normal'))
model.compile(optimizer='adam', loss='mean_squared_error')

score = model.fit(X, Y, batch_size=32, epochs=nbEpochs,
                  verbose=1, validation_split=0.20, shuffle=True)
model.save(saveFilename + '.h5')
```

Run it from [Neural networks models/](../../Neural%20networks%20models):

```bash
cd "Neural networks models"
python train_mlp_model.py
```

Expected result: `MLP3_t100.h5` (the trained model) and `MLP3_t100_loss.mat` (the `lossHistory`/`valLossHistory` arrays) appear in the current directory, and the console prints the per-epoch training and validation loss.

**Exercise** — Reduce `hiddenLayers` to `[90, 90, 90]` (three layers, matching the paper's `MLP2` architecture) and compare the final validation loss against the four-layer run.

---

## Chapter 6 · Training an LSTM predictor

The MLP treats each window as a flat vector, discarding the ordering of the `seqLen` states. [train_lstm_model.py](../../Neural%20networks%20models/train_lstm_model.py) instead feeds the window through a recurrent layer that processes it step by step, so it can exploit the temporal structure directly:

```python
seqLen = 10
lstmLayers = [90]     # one LSTM layer of 90 units; supports 1 or 2 entries
nbEpochs = 10
dataFilename = "moehlis_data_100.mat"
saveFilename = "LSTM1_t100"

model = Sequential()
model.add(LSTM(lstmLayers[0], input_shape=(seqLen, 9),
               kernel_initializer='glorot_normal', return_sequences=False))
model.add(Dense(9, activation='tanh'))
model.compile(optimizer='adam', loss='mean_squared_error')
```

Run it the same way as the MLP:

```bash
cd "Neural networks models"
python train_lstm_model.py
```

Expected result: `LSTM1_t100.h5` and `LSTM1_t100_loss.mat`. Because the recurrent connections make the temporal dependency explicit, the paper reports the LSTM reaching comparable or better accuracy than the best MLP while using an input window 50 times shorter ($p=10$ vs. $p=500$) — you can verify this yourself in Chapter 9 by comparing loss values.

**Exercise** — Change `lstmLayers` to `[90, 90]` (a second stacked LSTM layer, matching the paper's `LSTM2` architecture) and retrain; note that the code path in `train_lstm_model.py` automatically switches `return_sequences=True` on the first layer when `len(lstmLayers) == 2`.

---

## Chapter 7 · Predicting new turbulent time series

A trained model only produces one state at a time. To generate a full new trajectory, [predict_using_mlp.py](../../Neural%20networks%20models/predict_using_mlp.py) and [predict_using_lstm.py](../../Neural%20networks%20models/predict_using_lstm.py) seed the input with the first `seqLen` states of a held-out test series, then close the loop: each prediction is appended and fed back in to produce the next one.

```python
model = load_model('LSTM1_t100.h5')
dataStruct = sio.loadmat('moehlis_test_data_100.mat')
A = dataStruct['data']

testSeq = A[seqNo:seqNo+1]
predSeq = testSeq[:, :seqLen]                       # seed = first seqLen ground-truth states

for i in np.arange(testSeq.shape[1] - seqLen):
    nextState = model.predict(predSeq[:, i:i+seqLen], verbose=0)
    predSeq = np.concatenate((predSeq, [nextState]), axis=1)

sio.savemat(saveFilename + '%d' % (seqNo + 1), {'testSeq': testSeq, 'predSeq': predSeq})
```

Before running, create the output folder the script writes into and make sure `moehlis_test_data_100.mat` (a held-out dataset, see Chapter 3's exercise) sits alongside the model file:

```bash
mkdir -p LSTM1_t100_ps
cd "Neural networks models"
python predict_using_lstm.py
```

Expected result: `LSTM1_t100_ps/series_1.mat` through `series_10.mat`, each holding a `testSeq` (ground truth) and `predSeq` (autoregressive prediction) pair of shape `(nTP, 9)`.

**Exercise** — Run `predict_using_mlp.py` against `MLP3_t100.h5` on the same test data and compare its `predSeq` to the LSTM's for the same `seqNo`.

---

## Chapter 8 · Visualizing predictions and reconstructed flow fields

The `series_#.mat` files from Chapter 7 are plain amplitude arrays; the MATLAB helpers turn them back into something visual. [plot_amplitudes.m](../../Data%20generator%20%28Moehlis%20model%29/plot_amplitudes.m) plots all nine amplitudes side by side, and [visualize_fields.m](../../Data%20generator%20%28Moehlis%20model%29/visualize_fields.m) reconstructs the 3D velocity field from them using the same closed-form Fourier modes referenced in Chapter 2:

```matlab
load('LSTM1_t100_ps/series_1.mat')   % loads testSeq, predSeq
plot_amplitudes(predSeq)             % nine amplitude time series
visualize_fields(predSeq)            % 3D reconstructed velocity field at a chosen time
plot_amplitudes(testSeq)             % compare against the ground truth
```

Because `predSeq` and `testSeq` share the same `(nTP, 9)` shape, you can plot them on the same axes to see visually how closely the network's autoregressive rollout tracks the true trajectory — this is a qualitative complement to the quantitative error metrics in Chapter 9.

**Exercise** — Plot `predSeq(:,1)` and `testSeq(:,1)` (amplitude $a_1$, the mean-profile mode) on the same figure using MATLAB's `hold on`, and note how far the curves diverge by the end of the 4000 time units.

---

## Chapter 9 · Improving accuracy: architectures, epochs, and turbulence statistics

The paper quantifies prediction quality with the relative error of amplitude $a_1$ over $N_s$ samples,

$$\varepsilon_1 = \frac{1}{(N_s - p)\,a_{1,\text{lam}}} \sum_{j=p+1}^{N_s} \lvert a_{1,\text{tra}}^j - a_{1,\text{pred}}^j \rvert,$$

and analogous errors $E_{\bar u}$, $E_{\overline{u^2}}$ for the reconstructed mean-flow and fluctuation statistics. None of these metrics are computed by a checked-in script (see `developer_guide.md` §13), but you can reproduce their spirit directly from the `.mat` outputs you already have:

- **More epochs or resumed training** — set `train_more_epochs = True` in either training script to reload `<saveFilename>.h5` and continue fitting; the new loss history is concatenated onto the old one in `<saveFilename>_loss.mat`.
- **More training data** — the paper shows LSTM1's errors dropping from 2.36%/14.73% at 100 series to 0.45%/2.49% at 10 000 series (Table III); regenerate `moehlis_data_gen.m` with a larger `nTS` (Chapter 3) and retrain.
- **Architecture width/depth** — compare the `hiddenLayers`/`lstmLayers` variants you trained in Chapters 5–6; the paper's Table II and III give the reference errors for `MLP1`–`MLP5` and `LSTM1`–`LSTM3` so you can sanity-check your own runs against pre-trained weights in [trained_nn_models/](../../Neural%20networks%20models/trained_nn_models).

**Exercise** — Load `<name>_loss.mat` in MATLAB (`load('MLP3_t100_loss.mat')`) and plot `lossHistory` against `valLossHistory` across epochs; identify the epoch (if any) where validation loss starts increasing while training loss keeps decreasing — the overfitting pattern described in the paper's Fig. 2.

---

## Chapter 10 · Mastery: extending DeepTurbulence

Pick one of the following extension points, grounded in the gaps identified while writing this documentation, and implement a small patch:

1. **Add early stopping** — the paper describes early stopping (Fig. 2), but neither training script registers a Keras `EarlyStopping` callback; add one via the `callbacks=[...]` argument of `model.fit` in [train_lstm_model.py](../../Neural%20networks%20models/train_lstm_model.py).
2. **Generate the missing test set** — write a small variant of `moehlis_data_gen.m` that saves its output as `moehlis_test_data_<nTS>.mat`, closing the gap noted in Chapter 3/7.
3. **Compute the paper's error metrics** — write a Python script that loads a batch of `series_#.mat` files and computes $\varepsilon_1$ (Chapter 9) directly, instead of only visualizing the curves.
4. **Try a new architecture** — extend `train_lstm_model.py`'s two-layer branch to support three stacked LSTM layers, following the existing `return_sequences` pattern.
5. **Automate the whole pipeline** — script the sequence generate data → train → predict → visualize as a single command instead of four manual steps, without changing any of the existing files' behavior.

**Exercise** — Implement extension 1 (early stopping) and rerun Chapter 5's MLP training; confirm in `MLP3_t100_loss.mat` that training now stops before all 10 configured epochs complete if validation loss stops improving.

---

## After this tutorial

- Deep source reading: [developer_guide.md](./developer_guide.md).
- Task-oriented how-tos: [user_guide.md](./user_guide.md).
- Structural map: [topology.md](./topology.md).
- Canonical citation: P. A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa, "Predictions of turbulent shear flows using deep neural networks", *Phys. Rev. Fluids* **4**, 054603 (2019); local copy in [docs/references/](../references/).
