"""
compute_turbulence_statistics.py
---------------------------------
Computes the two headline turbulence-statistics error metrics reported in
Tables I-III of:
    "Predictions of turbulent shear flows using deep neural networks"
    P.A. Srinivasan, L. Guastoni, H. Azizpour, P. Schlatter, R. Vinuesa
    Physical Review Fluids (2019)

Requires:
    A folder of series_#.mat files produced by predict_using_mlp.py or
    predict_using_lstm.py, each containing:
        testSeq  - (nTP, 9) ground-truth amplitude time series (from the
                   nine-equation Moehlis model)
        predSeq  - (nTP, 9) network-predicted amplitude time series

Computes:
    eps1   - relative error of amplitude a_1 (paper Eq. 3)
    E_ubar - relative error of the mean streamwise velocity profile (paper
             Eq. 4). Averaging over the periodic x and z directions leaves
             only modes 1 and 9 (the only two modes with no x, z dependence
             -- see Data generator (Moehlis model)/visualize_fields.m,
             u1x and u9x), so the mean profile is recovered analytically
             from a_1 and a_9 alone, without reconstructing the full 3D
             velocity field.

Does NOT compute:
    E_{u'^2}, Reynolds shear stress, skewness, or flatness. Those
    statistics need the full 3D velocity field (all nine spatial modes)
    and cross-mode integrals over the periodic directions, which are not
    implemented here; see visualize_fields.m for the mode definitions if
    extending this script.

Usage:
    python compute_turbulence_statistics.py <series_dir> <seqLen>

    <series_dir> - folder of series_#.mat files (e.g. "MLP3_t100_ps" or
                    "LSTM1_t100_ps")
    <seqLen>     - the prediction order p used to generate that folder
                    (must match the seqLen in the corresponding
                    predict_using_{mlp,lstm}.py run)
"""
import glob
import os
import sys

import numpy as np
import scipy.io as sio

# Domain constant shared with Data generator (Moehlis model)/moehlis_data_gen.m
B = np.pi / 2

# Value of a_1 for laminar flow (paper, text following Eq. 3)
A1_LAM = 1.0

# Number of grid points used to numerically integrate over y in [-1, 1]
NY = 201


def mean_profile(a1, a9, y):
    """Mean streamwise velocity profile ubar(y, t) from amplitudes a1, a9
    (paper Eq. 1), analytically averaged over the periodic x and z
    directions. Every other mode has an x- or z-dependence that integrates
    to zero over a full period (see u2x, u6x-u8x in visualize_fields.m)."""
    u1 = np.sqrt(2) * np.sin(B * y)
    u9 = np.sqrt(2) * np.sin(3 * B * y)
    return a1[:, None] * u1[None, :] + a9[:, None] * u9[None, :]


def relative_amplitude_error(a1_true, a1_pred, seqLen):
    """eps_1, paper Eq. 3. Only the points after the seed window (index
    seqLen onward) are actual network predictions."""
    Ns = a1_true.shape[0]
    num = np.sum(np.abs(a1_true[seqLen:] - a1_pred[seqLen:]))
    return num / ((Ns - seqLen) * A1_LAM)


def mean_flow_error(testSeqs, predSeqs):
    """E_ubar, paper Eq. 4, ensemble- and time-averaged over all series."""
    y = np.linspace(-1, 1, NY)

    ubar_mod = np.mean(
        [mean_profile(s[:, 0], s[:, 8], y).mean(axis=0) for s in testSeqs],
        axis=0,
    )
    ubar_pred = np.mean(
        [mean_profile(s[:, 0], s[:, 8], y).mean(axis=0) for s in predSeqs],
        axis=0,
    )

    numerator = np.trapz(np.abs(ubar_mod - ubar_pred), y)
    denominator = 2 * np.max(np.abs(ubar_mod))
    return numerator / denominator


def main(seriesDir, seqLen):
    files = sorted(glob.glob(os.path.join(seriesDir, "series_*.mat")))
    if not files:
        sys.exit("No series_#.mat files found in %s" % seriesDir)

    testSeqs, predSeqs, eps1_list = [], [], []
    for f in files:
        matData = sio.loadmat(f)
        testSeq, predSeq = matData["testSeq"], matData["predSeq"]
        testSeqs.append(testSeq)
        predSeqs.append(predSeq)
        eps1_list.append(
            relative_amplitude_error(testSeq[:, 0], predSeq[:, 0], seqLen)
        )

    print("Series evaluated: %d" % len(files))
    print("eps_1  (mean over series) : %.4f %%" % (100 * np.mean(eps1_list)))
    print("E_ubar (ensemble-averaged): %.4f %%" % (100 * mean_flow_error(testSeqs, predSeqs)))


if __name__ == "__main__":
    seriesDir = sys.argv[1] if len(sys.argv) > 1 else "./LSTM1_t100_ps"
    seqLen = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(seriesDir, seqLen)
