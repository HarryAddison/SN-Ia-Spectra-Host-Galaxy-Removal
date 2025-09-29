import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from astropy.table import QTable
from importlib.resources import files


def dalek():
    dalek_data_dir = files('host_removal.data').joinpath(f"daleknn-data")
    train_valid_spectra = np.load(f'{dalek_data_dir}/grids/grid1_v2_log_uniform_fluxes_16feb20_part1_train_interp_98k.npy')
    train_valid_paramaters = pd.read_hdf(f'{dalek_data_dir}/grids/grid1_v2_log_uniform_params_16feb20_part1_train_98k.h5')

    test_spectra = np.load(f'{dalek_data_dir}/grids/grid1_v2_log_uniform_fluxes_26may20_part1_train_interp_19k_testing.npz')
    test_paramaters = pd.read_hdf(f'{dalek_data_dir}/grids/grid1_v2_log_uniform_fluxes_26may20_part1_train_interp_19k_testing.h5')

    # Load in the neural networks
    nn_emulator_0 = load_model(f'{dalek_data_dir}/networks/00-260285.h5')
    nn_emulator_1 = load_model(f'{dalek_data_dir}/networks/01-260022.h5')
    nn_emulator_2 = load_model(f'{dalek_data_dir}/networks/02-260627.h5')
    nn_emulator_3 = load_model(f'{dalek_data_dir}/networks/03-261931.h5')
    nn_emulator_4 = load_model(f'{dalek_data_dir}/networks/04-261295.h5')

    # Preprocess the training spectra/parameters so that the scaling can be
    # applied to the desired input parameters and the outputs reconstructed.

    # Pre-processing of training data
    train_spectra_pre_proc = train_valid_spectra
    train_params_pre_proc = train_valid_paramaters

    # Take log10 of values
    train_spectra_pre_proc = np.log10(train_spectra_pre_proc)
    train_params_pre_proc = np.log10(train_params_pre_proc)

    # Standardise spectra using the "StandardScaler"
    scaler_spec = StandardScaler(with_mean=True, with_std=True)
    scaler_params = StandardScaler(with_mean=True, with_std=True)

    scaler_spec.fit(train_spectra_pre_proc)
    scaler_params.fit(train_params_pre_proc)


    # Pre-processing of training data
    train_spectra_pre_proc = train_valid_spectra

    # Take log10 of the spectral fluxes
    train_spectra_pre_proc = np.log10(train_spectra_pre_proc)

    # Standardise spectra using the "StandardScaler"
    scaler = StandardScaler(with_mean=True, with_std=True)

    scaler.fit(train_spectra_pre_proc)

    train_spectra_pre_proc = scaler.transform(train_spectra_pre_proc)

    # Apply pre-processing to the desired input prarameters
    test_paramaters_pre_proc = np.log10(test_paramaters)
    
    test_paramaters_pre_proc = scaler_params.transform(test_paramaters_pre_proc)

    # Predict spectra using the desired pre-processed input parameters
    prediction_0 = nn_emulator_0.predict(test_paramaters_pre_proc)
    prediction_1 = nn_emulator_1.predict(test_paramaters_pre_proc)
    prediction_2 = nn_emulator_2.predict(test_paramaters_pre_proc)
    prediction_3 = nn_emulator_3.predict(test_paramaters_pre_proc)
    prediction_4 = nn_emulator_4.predict(test_paramaters_pre_proc)

    wl = np.load(f'{dalek_data_dir}/grids/wavelength.npy')
    # Average and visualise results
    mean_prediction = np.mean(np.array([prediction_0, prediction_1, prediction_2, prediction_3, prediction_4]), axis=0)
    mean_prediction = scaler_spec.inverse_transform(mean_prediction)
    # Normalise so max value is 1.
    max_flux = np.max(mean_prediction, axis=1)
    min_flux = np.min(mean_prediction, axis=1)
    # mean_prediction = (mean_prediction - min_flux[:, None]) / (max_flux[:, None] - min_flux[:, None]) + 0.3

    return wl, mean_prediction
