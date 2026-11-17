"""
PhysioNet EEGBCI loader / preprocessor.
Loads motor-imagery data, optionally applies DSP filters, epochs it,
and returns per-subject arrays (X, y), it doesn't use a machine learning pipeline
"""

from joblib import Parallel, delayed
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
import numpy as np
import scipy.signal as signal

# Suppress verbose MNE logging outputs to keep the console clean
mne.set_log_level("WARNING")

# Standardized sampling frequency (Hz) so all subjects match in temporal dimension
SFREQ = 160.0  


# DSP FILTERS, works on getting rid of certain frequencies


def apply_notch_filter(raw, freq=60.0, q=30.0):
    """Notch filter to remove 60 Hz power-line interference."""
    out = raw.copy()
    # Design IIR notch filter coefficients (b, a)
    b, a = signal.iirnotch(freq, q, fs=out.info["sfreq"])
    data = out.get_data()
    # Apply zero-phase forward-backward filter across each EEG channel independently
    for i in range(data.shape[0]):
        data[i] = signal.filtfilt(b, a, data[i])
    out._data = data
    return out


def apply_butterworth_bandpass(raw, l_freq=8.0, h_freq=30.0, order=4):
    """Butterworth bandpass filter, it isolates mu + beta frequency bands (8-30 Hz)"""
    out = raw.copy()
    # Design Second-Order Sections (SOS) bandpass filter for numerical stability
    sos = signal.butter(order, [l_freq, h_freq], btype="bandpass",
                        fs=out.info["sfreq"], output="sos")
    data = out.get_data()
    # Apply zero-phase filtering across each channel
    for i in range(data.shape[0]):
        data[i] = signal.sosfiltfilt(sos, data[i])
    out._data = data
    return out


# Maps dictionary to easily select filtering pipelines
FILTER_MODES = {
    "raw": lambda r: r,
    "notch": apply_notch_filter,
    "bandpass": apply_butterworth_bandpass,
    "both": lambda r: apply_notch_filter(apply_butterworth_bandpass(r)),
}



def load_subject(sub, runs, channels, mode="raw", tmin=-0.5, tmax=2.0, baseline=(-0.5, 0)):
    """Returns (X, y) for one subject, or (None, None) if loading fails."""
    try:
        # Download EDF files for the specified subject and experimental runs
        fnames = eegbci.load_data(sub, runs, update_path=False)
        X_runs, y_runs = [], []

        # Process each run separately to prevent edge filter artifacts at boundaries
        for f in fnames:
            raw = read_raw_edf(f, preload=True, verbose=False)
            eegbci.standardize(raw)              # Standardize channel naming conventions (e.g., C3., FC3.)
            raw.resample(SFREQ)                  # Resample to common SFREQ
            raw = FILTER_MODES[mode](raw)        # Apply selected DSP filter mode

            # Extract experimental events and annotations from the raw stream then keeps only motor imagery events 
            # then creates epochs and selects specific channels
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            
            keep = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
            
            epochs = mne.Epochs(raw, events, keep, tmin=tmin, tmax=tmax,
                                baseline=baseline, preload=True, verbose=False)
            
            epochs.pick(channels)

            X_runs.append(epochs.get_data())
            # Convert event IDs to binary labels: T1 -> left (0), T2 -> right (1)
            y_runs.append((epochs.events[:, -1] == keep["T2"]).astype(int))

        return np.concatenate(X_runs), np.concatenate(y_runs)

    except Exception as e:
        print(f"Subject {sub} failed: {e}")
        return None, None



# This part of the code loads the rest of the subjects

# Subjects commonly excluded due to missing files, bad formatting, or unique channel layouts
EXCLUDE = {88, 92, 100, 104}


def load_dataset(subjects=range(1, 110), runs=(4, 8, 12),
                 channels=("C3", "Cz", "C4"), mode="raw",
                 tmin=-0.5, tmax=2.0, baseline=(-0.5, 0), n_jobs=-1):
    """
    Returns dict: {subject_id: (X, y)} for every subject that loaded successfully.
    mode: 'raw' | 'notch' | 'bandpass' | 'both'
    """
    subjects = [s for s in subjects if s not in EXCLUDE]
    
    # Run subject loading tasks concurrently across all CPU cores using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(load_subject)(s, list(runs), list(channels), mode, tmin, tmax, baseline)
        for s in subjects
    )
    
    # Package successful results into a dictionary keyed by subject ID
    data = {s: r for s, r in zip(subjects, results) if r[0] is not None}
    print(f"[{mode}] loaded {len(data)}/{len(subjects)} subjects")
    return data


def save_dataset(data, path):
    """Save multi-subject dictionaries into a single compressed .npz archive."""
    arrays = {}
    for s, (X, y) in data.items():
        arrays[f"X_{s}"] = X
        arrays[f"y_{s}"] = y
    np.savez_compressed(path, **arrays)
    print(f"Saved {len(data)} subjects to {path}")



if __name__ == "__main__":
    # Loop through all filter configurations, process the dataset, and save artifacts
    for mode in ["raw", "notch", "bandpass", "both"]:
        data = load_dataset(subjects=range(1, 110), mode=mode)
        save_dataset(data, f"physionet_{mode}.npz")

        # prints out the shape and label of the first loaded subject
        if data:
            s = next(iter(data))
            X, y = data[s]
            print(f"  subject {s}: X={X.shape}, labels={np.bincount(y)}")