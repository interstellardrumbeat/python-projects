"""
/)  /)  ~ ┏━━━━━━━━┓
( •-• )  ~     HOW TO USE
/づづ ~ ┗━━━━━━━━┛

Check README for detailed documentation.

This script needs an info file, whose path and name must be given at the end of this script, in the "main" call (info_file_path and info_file_name variable).
The info file MUST have a specific structure. Check the example on GitLab.

GLOSSARY:
ACS: Absorption Cross Section
Sigma: Wavenumber
"""

import sys
import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import os
import struct
import matplotlib.pyplot as plt
import re

# Small functions to parse the info file.
# Each info file line starts with an identifier name (eg. Input folder: C:\Users\rest of the path...)
def read_string(instring):
    parts = instring.split(":", 1)[1].strip()
    return parts

def read_array(instring):
    data_part = instring.split(':', 1)[1].strip()
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', data_part)
    return np.array(nums, dtype=float)

# Polynomial calculation function
def fit_pTpoly_for_sigma(obs_ACS, T_input, p_input, add_weights, noise_arr, T_degree, p_degree):
    """
    Solve for coefficients "a" of:
    f(T, p) = ∑{i=0..T_degree} ∑{j=0..p_degree} a{i,j} * T^i * log(p+5)^j

    n_ACS: number of input spectra, not defined here but later (main function)
    obs_ACS: observed Absorption Cross Section, baseline-corrected and scaled, of all the spectra (array of length n_ACS )
    T_input: Temperatures of all the spectra (array of length n_ACS)
    p_input: total pressures of all the spectra (array of length n_ACS)
    add_weights: per-spectrum multiplicative additional weights from add_weight_per_spectrum[region] (array of length n_ACS)
    noise_arr: noise of each spectrum (array of length n_ACS)

    popt: coefficients of the polynomial fitted from the input spectra; the degrees are decided by the user in the info file (flattened length (T_degree+1)*(p_degree+1))
    pcov: covariance of the coefficients, diagonal values are parameter variances
    calc: predicted ACS values
    """
    n = len(obs_ACS)
    nT = T_degree + 1
    nP = p_degree + 1
    npar = nT * nP

    # Function for curve_fit
    def acs_polynomial_function(Tp, *coeffs):
        T, p = Tp
        logp = np.log(p + 5.0)
        ACS_temp = np.zeros_like(T)
        poly_index = 0
        for i in range(nT):
            for j in range(nP):
                ACS_temp += coeffs[poly_index] * (T ** i) * (logp ** j)
                poly_index += 1
        return ACS_temp

    # Initial guess for coefficients, if a better guess is available, insert here
    par_init = np.zeros(npar)

    weights = np.sqrt(add_weights / (noise_arr ** 2))

    popt, pcov = curve_fit(acs_polynomial_function, (T_input, p_input), obs_ACS, p0=par_init, sigma=1/weights, absolute_sigma=True, maxfev=2000)

    calc = acs_polynomial_function((T_input, p_input), *popt)

    return popt, pcov, calc

# Main loop/pipeline
def run_from_info(info_file_path, info_file_name, region=0, plot=True):
    """
    info_file_path: directory where info file is stored, given at the end of the script
    info_file_name: name of the info file (single filename with format, not full path), given at the end of the script
    region: section of the spectra to fit, if multiple defined in info file (default 0)
    output_path: path (including filename) where to save results with .npz format; given at the end of the script
    """

    # Read info file
    info_full = os.path.join(info_file_path, info_file_name)
    with open(info_full, 'r') as f:
        lines = f.readlines()

    ln = 0
    # Read common section of the info file (first 5 lines):
    filepath = read_string(lines[ln]); ln += 1
    outpath = read_string(lines[ln]); ln += 1
    polyfitranges = read_array(lines[ln]); ln += 1

    all_ranges_min = polyfitranges[0::2]
    all_ranges_max = polyfitranges[1::2]
    p_poly_degree = int(read_array(lines[ln])[0]); ln += 1
    T_poly_degree = int(read_array(lines[ln])[0]); ln += 1

   # Define sigma ranges based on selected region
    sigma_min_fit = all_ranges_min[region]
    sigma_max_fit = all_ranges_max[region]

    # Read repeating sections of the info file, one section/block is a different spectrum.
    ACS_entries = []
    while ln < len(lines):
        line = lines[ln].strip()

        # Check if this is an ACS file line (i.e. the name of the input spectrum), otherwise skips it.
        # Each block has 7 data lines.
        if line.startswith("ACS file:"):
            ACS_filename = read_string(line)
            noise = read_array(lines[ln+1])[0]
            add_weight_per_spectrum = read_array(lines[ln+2])
            p_tot = read_array(lines[ln+3])[0]
            T = read_array(lines[ln+4])[0]
            baseline_poly_degree = read_array(lines[ln+5]).astype(int)
            scale_ACS = read_array(lines[ln+6])

            ACS_entries.append({
                'ACS_filename': ACS_filename,
                'noise': noise,
                'p_tot': p_tot,
                'T': T,
                'add_weight': add_weight_per_spectrum,
                'baseline_poly_degree': baseline_poly_degree,
                'scale_ACS': scale_ACS
            })
            ln += 7
        else:
            ln += 1

    n_ACS = len(ACS_entries)
    if n_ACS == 0:
        raise RuntimeError('No ACS entries read from info file.')
    print(f"Info file {info_file_name} read. ACS entries: {n_ACS}")

    # Read each spectrum and store its ACS and sigma
    sigma_axis = None
    for acs in ACS_entries:
        input_spectrum = os.path.join(filepath, acs['ACS_filename'])
        data = np.loadtxt(input_spectrum)
        x = data[:, 0].astype(float)    # Sigma, i.e. wavenumber
        y = data[:, 1].astype(float)    # Absorption Cross Section (or absorption coefficient or any other spectroscopically relevant value)

        mask = (x + 1e-6 >= sigma_min_fit) & (x - 1e-6 <= sigma_max_fit)    #1d-6 added in case of values falling on the given sigma limits
        x_sel = x[mask]
        y_sel = y[mask]
        if sigma_axis is None:
            sigma_axis = x_sel.copy()
            n_spectral_points = len(x_sel)
        else:
            if len(x_sel) != n_spectral_points or not np.allclose(x_sel, sigma_axis):
                raise RuntimeError(f"x axis mismatch for file {acs['ACS_filename']}")
        acs['sigma'] = x_sel
        acs['ACS'] = y_sel

        print(f"ACS file {acs['ACS_filename']} read.")

    print(f"Reading {n_ACS} ACS files completed.")
    sigma = sigma_axis.copy()
    n_sigma = len(sigma)

    # Prepare arrays for fitting
    baseline_poly_degree_all = np.array([entry['baseline_poly_degree'][region] for entry in ACS_entries], dtype=int)
    scale_ACS_all = np.array([entry['scale_ACS'][region] for entry in ACS_entries])
    add_weight_all_arr = np.array([entry['add_weight'][region] for entry in ACS_entries])
    noise_arr = np.array([entry['noise'] for entry in ACS_entries])
    p_tot_arr = np.array([entry['p_tot'] for entry in ACS_entries])
    T_arr = np.array([entry['T'] for entry in ACS_entries])

    # Check for which spectra fit the baseline, scaling and which include in the fit
    fit_baseline_index = np.where(baseline_poly_degree_all != -1)[0]
    fit_scaling_index = np.where(scale_ACS_all == 1)[0]
    obs_in_fit_index = np.where((baseline_poly_degree_all != -1) | (scale_ACS_all == 1))[0]

    # Prepare initial parameters (pars) vector
    pars_list = []
    baseline_blocks = []
    # Add parameter and store only spectra which baseline and scaling will be fitted (based on flag in info file)
    for i, entry in enumerate(ACS_entries):
        deg = entry['baseline_poly_degree'][region]
        if deg != -1:
            baseline_blocks.append((i, deg))
            pars_list.extend([0.0] * (deg + 1))
    for i in fit_scaling_index:
        pars_list.append(1.0)

    pars0 = np.array(pars_list)
    print("Initial pars length:", len(pars0))

    # Setup relation between blocks (both scaling and baseline) and pars
    baseline_param_slices = []
    pos = 0
    for (i, deg) in baseline_blocks:
        baseline_param_slices.append((i, pos, pos + deg + 1))
        pos += deg + 1
    scaling_param_slices = []
    for i in fit_scaling_index:
        scaling_param_slices.append((i, pos))
        pos += 1

    ACS_data_matrix = np.vstack([entry['ACS'] for entry in ACS_entries])    # Shape (n_ACS, n_sigma)

    # Function for optimization of parameters in least_squares (used later)
    def residuals_for_pars(pars):

        ACS_basecorr_scaled = ACS_data_matrix.copy()
        baselines_matrix = np.zeros_like(ACS_basecorr_scaled)
        scalings = np.ones(n_ACS)

        # Apply baseline correction and scaling and then run polynomial fit
        for (i, start, end) in baseline_param_slices:
            coeffs = pars[start:end]
            baseline = poly.polyval(sigma - sigma[0], coeffs)   # Reminder: poynomial.polyval uses increasing-order coeffs, np.polyval decreasing (and arguments inverted)
            ACS_basecorr_scaled[i, :] -= baseline
            baselines_matrix[i, :] = baseline

        for (i, pos) in scaling_param_slices:
            scal = pars[pos]
            ACS_basecorr_scaled[i, :] *= scal
            scalings[i] = scal

        model_ACS = np.zeros_like(ACS_basecorr_scaled)
        for j in range(n_sigma):
            obs = ACS_basecorr_scaled[:, j]
            poly_coeff, coeff_covariance, predicted_ACS = fit_pTpoly_for_sigma(
                obs_ACS=obs,
                T_input=T_arr,
                p_input=p_tot_arr,
                add_weights=add_weight_all_arr,
                noise_arr=noise_arr,
                T_degree=T_poly_degree,
                p_degree=p_poly_degree
            )
            model_ACS[:, j] = predicted_ACS

        # Remove baseline correction and scaling (applied before for polynomial fit)
        model_ACS_adjusted = np.zeros_like(model_ACS)
        for i in range(n_ACS):
            model_ACS_adjusted[i, :] = model_ACS[i, :] / scalings[i] + baselines_matrix[i, :]

        if len(obs_in_fit_index) == 0:
            return np.array([])

        # Flattened residuals for least_squares
        calc_flat = model_ACS_adjusted[obs_in_fit_index, :].ravel(order='C')
        obs_flat = ACS_data_matrix[obs_in_fit_index, :].ravel(order='C')
        weights_flat = np.sqrt((1.0 / noise_arr**2 * add_weight_all_arr)[obs_in_fit_index]).repeat(n_sigma)
        weighted_residuals = (calc_flat - obs_flat) * weights_flat
        return weighted_residuals

    # Optimization of parameters using residuals_for_pars function
    if len(pars0) > 0:
        print("Running nonlinear fit for baseline and scaling:")
        lsq_results = least_squares(residuals_for_pars, pars0, verbose=2, max_nfev=1)   # Reminder, needs 1d array
        pars_opt = lsq_results.x
    else:
        print("No baseline and scaling fit selected. Skipping nonlinear fit.")
        pars_opt = np.array([])

    ACS_basecorr_scaled = ACS_data_matrix.copy()
    baselines_matrix = np.zeros_like(ACS_basecorr_scaled)
    scalings = np.ones(n_ACS)

    # Final calculation: apply baseline correction and scaling using optimized parameters and then run polynomial fit
    if len(pars_opt) > 0:
        for (i, start, end) in baseline_param_slices:
            coeffs = pars_opt[start:end]
            baseline = poly.polyval(sigma - sigma[0], coeffs)   # Re-reminder: poynomial.polyval uses increasing-order coeffs, np.polyval decreasing and arguments inverted
            ACS_basecorr_scaled[i, :] -= baseline
            baselines_matrix[i, :] = baseline

        for (i, pos) in scaling_param_slices:
            scal = pars_opt[pos]
            ACS_basecorr_scaled[i, :] *= scal
            scalings[i] = scal

    fitted_pTcoeffs = np.zeros((T_poly_degree + 1, p_poly_degree + 1, n_sigma))
    fitted_pTcoeffs_errs = np.zeros_like(fitted_pTcoeffs)
    model_ACS = np.zeros_like(ACS_basecorr_scaled)

    print("Running final polynomial fit.")
    for j in range(n_sigma):
        obs = ACS_basecorr_scaled[:, j]
        poly_coeff, coeff_covariance, predicted_ACS = fit_pTpoly_for_sigma(
            obs_ACS=obs,
            T_input=T_arr,
            p_input=p_tot_arr,
            add_weights=add_weight_all_arr,
            noise_arr=noise_arr,
            T_degree=T_poly_degree,
            p_degree=p_poly_degree
        )
        model_ACS[:, j] = predicted_ACS
        fitted_pTcoeffs[:, :, j] = poly_coeff.reshape((T_poly_degree + 1, p_poly_degree + 1))
        fitted_pTcoeffs_errs[:, :, j] = np.sqrt(np.abs(np.diag(coeff_covariance))).reshape((T_poly_degree + 1, p_poly_degree + 1))

    model_ACS_adjusted = np.zeros_like(model_ACS)
    for i in range(n_ACS):
        model_ACS_adjusted[i, :] = model_ACS[i, :] / scalings[i] + baselines_matrix[i, :]

    # Optional plotting (to be fixed, currently useless)
    if plot:
        obs_all = ACS_data_matrix.ravel(order='F')
        calc_all = model_ACS_adjusted.ravel(order='F')
        plt.figure(figsize=(12, 4))
        plt.plot(obs_all, label='obs_all')
        plt.plot(calc_all, label='calc_all')
        plt.legend()
        plt.title('IGNORE ME FOR NOW')
        plt.show()

    # Save results
    output_path = os.path.join(outpath, 'fitted_polynomial_test.npz')

    np.savez_compressed(output_path,
                        sigma=sigma,
                        p_poly_degree=p_poly_degree,
                        T_poly_degree=T_poly_degree,
                        fitted_pTcoeffs=fitted_pTcoeffs,
                        fitted_pTcoeffs_errs=fitted_pTcoeffs_errs,
                        model_ACS_adjusted=model_ACS_adjusted,
                        ACS_entries=ACS_entries)
    print(f'Results saved to {output_path}')

if __name__ == '__main__':
    info_file_path = r"C:/Users/prud_do/Desktop/Proton Drive/My files/Nick/Coding/Spectroscopy/ACS poly fit/"
    info_file_name = "info_polyfit.txt"

    run_from_info(info_file_path, info_file_name, region=0, plot=False)
