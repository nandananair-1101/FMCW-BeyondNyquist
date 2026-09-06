# FMCW Beyond-Nyquist Analysis 

This repository contains a collection of  MATLAB scripts designed for Frequency Modulated Continuous Wave (FMCW) signal processing analysis. Implementing and analyzing methods for beyond Nyquist target detection.

## Repository Files Overview

- **`RadarAnalysis_RCS_CFAR.m`**  
  Provides Radar Analysis within Nyquist limits,covering system diagnostics, 1D/2D FFT profiles, multi-target resolution matrices, Radar Cross Section (RCS) scaling, target variations with noise addition (same range/different Doppler and same Doppler/different range), and Cell-Averaging CFAR detection.

- **`BeyondNyquist_matrix_spectrogram.m`**  
  Demonstrates FMCW range aliasing proofs (comparing within-Nyquist vs. beyond-Nyquist targets),  IFF likelihood surface estimation, matrix-based analysis, and multi-target range detection by analyzing the spectrogram gap to detect the delay.

- **`multiTarget_fft_iff.m`**  
  Resolves multiple targets beyond Nyquist limits by combining 2D Fast Fourier Transform (FFT) rough binning with iterative IFF refinement and target subtraction.

- **`multiTarget_mf_iff.m`**  
  Combines Matched Filtering (MF) coarse range profiling with IFF fine-range likelihood optimization for multi-target FMCW scenarios.

- **`iff_noise_monteCarlo.m`**  
  Executes parallelized Monte Carlo simulations across varying Signal-to-Noise Ratios  to evaluate and benchmark the Root Mean Square Error (RMSE) of range and velocity estimations against standard 2D FFT techniques.


## Requirements

- **MATLAB** 
- **Phased Array System Toolbox** (Required for `phased.FMCWWaveform` and `delayseq`)
- **Parallel Computing Toolbox**  (Required for `parfor`)
