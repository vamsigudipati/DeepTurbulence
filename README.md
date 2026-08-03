# DeepTurbulence

Predicting the temporal dynamics of turbulent shear flows with deep neural networks, using a low-order model of near-wall turbulence as ground truth.

## 1. Project summary

DeepTurbulence couples a MATLAB implementation of a low-dimensional turbulence model with Python/Keras neural networks that learn to predict its time evolution. Concretely, the repository:

- Integrates the **nine-equation shear-flow model** of Moehlis *et al.* to generate turbulent time series of nine modal amplitudes ([Data generator (Moehlis model)/](Data%20generator%20%28Moehlis%20model%29)).
- Trains a **multilayer perceptron (MLP)** or a **long short-term memory (LSTM)** network to predict the next amplitude state from a window of previous states ([Neural networks models/](Neural%20networks%20models)).
- Uses a trained network to **autoregressively roll out** a full new turbulent trajectory from a short seed sequence, and reconstructs/visualizes the corresponding 3D velocity field.

## 2. Scientific context

The flow model and the neural-network results are described in P. A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa, ["Predictions of turbulent shear flows using deep neural networks"](https://link.aps.org/doi/10.1103/PhysRevFluids.4.054603), *Phys. Rev. Fluids* **4**, 054603 (2019) (also on [arXiv](https://arxiv.org/abs/1905.03634)), building on the nine-mode model of [Moehlis *et al.*](https://iopscience.iop.org/article/10.1088/1367-2630/6/1/056/meta), *New J. Phys.* **6**, 56 (2004).

The core idea is a Galerkin projection of the Navier–Stokes equations onto nine fixed Fourier modes $\mathbf{u}_j(\mathbf{x})$. The instantaneous velocity field is recovered as a weighted superposition of these modes, with time-dependent amplitudes $a_j(t)$ carrying all the dynamics:

$$\tilde{\mathbf{u}}(\mathbf{x}, t) = \sum_{j=1}^{9} a_j(t)\,\mathbf{u}_j(\mathbf{x}).$$

The amplitudes evolve according to a coupled system of nine ODEs (implemented in [moehlis_model_odefun.m](Data%20generator%20%28Moehlis%20model%29/moehlis_model_odefun.m)) at a model Reynolds number $\mathrm{Re}=400$. Learning to predict $a_j(t)$ is therefore equivalent to learning the dynamics of the flow itself — the task the MLP and LSTM networks in this repository are trained to perform, using the mean-squared-error loss

$$L(f(\mathbf{x}); \boldsymbol{\psi}) = \frac{1}{2m} \sum_{j=1}^{m} \lVert \psi^j - f(\chi^j) \rVert^2.$$

A full local copy of the paper (converted to Markdown) is available under [docs/references/](docs/references).

## 3. Quick start

Generate a training dataset in MATLAB, then train and predict in Python:

```matlab
% 1. In MATLAB, from Data generator (Moehlis model)/
moehlis_data_gen        % -> moehlis_data_10.mat  (data: (nTS, nTP, 9))
```

```bash
# 2. Train an MLP or LSTM predictor
cd "Neural networks models"
python train_mlp_model.py     # or: python train_lstm_model.py

# 3. Predict a new time series with the trained model
mkdir -p MLP3_t100_ps
python predict_using_mlp.py   # or: python predict_using_lstm.py
```

Python dependencies: `numpy`, `scipy`, `matplotlib`, `keras`, `tensorflow` (`pip install numpy scipy matplotlib keras tensorflow`). See the [User Guide](docs/guides/user_guide.md) for full installation and configuration details.

## 4. Documentation hub

Coordinated documentation for this repository lives under [docs/guides/](docs/guides), produced with the topology-first mining workflow:

- [Topology Map](docs/guides/topology.md) — Visual structural & module map.
- [User Guide](docs/guides/user_guide.md) — Setup, configuration, & execution.
- [Developer Guide](docs/guides/developer_guide.md) — Deep architecture, math, & code details.
- [Tutorial](docs/guides/tutorial.md) — Step-by-step 10-chapter learning path.

Start with the [Topology Map](docs/guides/topology.md) for a structural overview, then follow the [Tutorial](docs/guides/tutorial.md) for a hands-on walkthrough from data generation to visualization, or jump straight to the [User Guide](docs/guides/user_guide.md) / [Developer Guide](docs/guides/developer_guide.md) for task-specific reference.

## 5. Repository layout

| Folder | Contents |
| --- | --- |
| [Data generator (Moehlis model)/](Data%20generator%20%28Moehlis%20model%29) | MATLAB scripts to generate and visualize training/test time series from the nine-equation model. |
| [Neural networks models/](Neural%20networks%20models) | Python/Keras scripts to train MLP/LSTM predictors and run autoregressive predictions; pre-trained weights in `trained_nn_models/`. |
| [docs/guides/](docs/guides) | Generated documentation (topology, developer guide, user guide, tutorial). |
| [docs/references/](docs/references) | Local Markdown copy of the reference paper. |
