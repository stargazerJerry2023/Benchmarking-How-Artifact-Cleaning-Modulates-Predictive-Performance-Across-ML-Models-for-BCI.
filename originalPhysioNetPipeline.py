"""
PhysioNet EEGBCI Baseline Pipeline, it gives the benchmark scores for the testPhysioNetPrep script and doesn't use any improvements
uses 5x3 Repeated Stratified K-Fold
meant to be a direct comparison improvedPhysioNetPipeline.py to see differences in changes with using Python libraries and not using Python libraries
"""
from joblib import Parallel, delayed
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

# Import shared preprocessing components and configurations from preprocessing script
from testPhysioNetPrep import FILTER_MODES, SFREQ, EXCLUDE

# Suppress verbose MNE logs to keep console output clean during parallel loops
mne.set_log_level("WARNING")



SUBJECTS = range(1, 110)
RUNS = [4, 8, 12]                  # Experimental runs corresponding to motor imagery tasks
MOTOR_CHANNELS = ['C3', 'Cz', 'C4'] # Core motor cortex electrode channels
TMIN, TMAX = -0.5, 2.0             # Epoch time window around cue events (seconds)
BASELINE = (-0.5, 0)               # Baseline correction window

# Different signal processing filter states to evaluate and compare
STATES = [
    ("State 0 (Raw Baseline)", "raw"),
    ("State 1 (Notch 60Hz)", "notch"),
    ("State 3 (Butterworth 8-30Hz)", "bandpass"),
    ("State 5 (Combined DSP)", "both"),
]



# Subject Loader, loads three subjects

def load_subject_original(sub, mode):
    """Loads, filters, epochs, and returns (X, y) arrays for a single subject."""
    try:
        fnames = eegbci.load_data(sub, RUNS, update_path=False)
        X_runs, y_runs = [], []

        for f in fnames:
            raw = read_raw_edf(f, preload=True, verbose=False)
            eegbci.standardize(raw)
            raw.resample(SFREQ)
            raw = FILTER_MODES[mode](raw) # Apply the specified DSP filter mode

            events, event_id = mne.events_from_annotations(raw, verbose=False)
            keep = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
            
            epochs = mne.Epochs(
                raw, events, keep, tmin=TMIN, tmax=TMAX,
                baseline=BASELINE, preload=True, verbose=False
            )
            epochs.pick(MOTOR_CHANNELS) # Isolate 3 motor channels

            X_runs.append(epochs.get_data())
            # Convert event IDs to binary labels: T1 -> 0 (left), T2 -> 1 (right)
            y_runs.append((epochs.events[:, -1] == keep["T2"]).astype(int))

        return np.concatenate(X_runs), np.concatenate(y_runs)
    except Exception as e:
        print(f"Subject {sub} failed: {e}")
        return None


# PER-Subject Scoring, gives a score for each subject

def score_subject_original(X, y):
    """Trains a baseline CSP + LDA model and evaluates accuracy via cross-validation."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    
    clf = Pipeline([
        ("csp", CSP(n_components=2, reg=None, log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])

    accs = []
    for tr, te in cv.split(X, y):
        m = clf.fit(X[tr], y[tr])
        accs.append((m.predict(X[te]) == y[te]).mean())

    return np.mean(accs)



# Executes benchmarks and gives benchmark score


if __name__ == "__main__":
    subs = [s for s in SUBJECTS if s not in EXCLUDE]
    print("--- Running Original Pipeline (Per-Subject Evaluation) ---")

    # Loop through each filtering state and run the full benchmarking pipeline
    for state_name, mode in STATES:
        # Load all subjects in parallel using all available CPU cores
        loaded = Parallel(n_jobs=-1)(
            delayed(load_subject_original)(s, mode) for s in subs
        )
        data = [r for r in loaded if r is not None]

        # Score all successfully loaded subjects in parallel
        sc = np.array(
            Parallel(n_jobs=-1)(delayed(score_subject_original)(X, y) for X, y in data)
        )

        # Calculate benchmarks across the subject population
        resp = sc[sc >= 0.70] # Subjects achieving 70%+ accuracy
        resp_mean = f"{resp.mean() * 100:.2f}%" if len(resp) > 0 else "N/A"
        
        print(
            f"[{state_name}] -> All Mean: {sc.mean() * 100:.2f}% (+/- {sc.std() * 100:.2f}%) "
            f"| Median: {np.median(sc) * 100:.2f}% "
            f"| >=70%: {(sc >= 0.70).sum()}/{len(sc)} (Responsive Mean: {resp_mean}) "
            f"| >=80%: {(sc >= 0.80).sum()}/{len(sc)}"
        )

    print("\nBenchmark Complete")