"""
PhysioNet EEGBCI, this code executes per-subject evaluation with:
  Run-level Euclidean Alignment (EA)
  Log-Euclidean Riemannian Tangent Space
  Spatio-Temporal Time-Delay Embedding (delay=4 samples / 25ms)
  Cross-domain augmentation with executed-movement runs (3, 7, 11)

Models evaluated:
  C.       Spatial Tangent Space (8-30 Hz) + Shrinkage LDA
  C+exec.  Spatial Tangent Space + Executed Training Data + Shrinkage LDA
  E.       Spatio-Temporal Tangent Space (delay=4) + Shrinkage LDA
  E+exec.  Spatio-Temporal Tangent Space (delay=4) + Executed Data + Shrinkage LDA
  F+exec.  Spatio-Temporal Tangent Space (delay=4) + Executed Data + LinearSVC
"""

from joblib import Parallel, delayed
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import oas
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import json


from testPhysioNetPrep import apply_butterworth_bandpass, SFREQ, EXCLUDE

# Suppress verbose MNE logging outputs to keep console clean
mne.set_log_level("WARNING")



SUBJECTS = range(1, 110)       
IMAGERY_RUNS = [4, 8, 12]      
EXEC_RUNS = [3, 7, 11]         
CHANNELS = [                   
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C3', 'C1', 'Cz', 'C2', 'C4',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
]
TMIN, TMAX = 0.5, 2.5          
BANDS = [(8, 30)]              
WIDE_IDX = 0



def euclidean_align(X_run):
    """
    Applies Euclidean Alignment to a run's trials and
    re-centers trials so their mean spatial covariance matrix equals the Identity matrix.
    This gets rid of inter-run drift and aligns execution vs. imagery baselines.
    """
    covs = np.array([np.cov(x) for x in X_run])
    R_mean = covs.mean(axis=0)
    w, V = np.linalg.eigh(R_mean)
    # Compute inverse square root of mean covariance matrix
    R_inv_sqrt = (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    # Project trial data onto aligned space using Einstein summation
    return np.einsum('ij,njt->nit', R_inv_sqrt, X_run)


def load_runs(sub, runs, align=True):
    """
    Downloads, loads, preprocesses, filters, epochs, and optionally 
    Euclidean-aligns a specified set of runs for a given subject.
    """
    fnames = eegbci.load_data(sub, runs, update_path=False)
    X_runs, y_runs = [], []

    for f in fnames:
        raw = read_raw_edf(f, preload=True, verbose=False)
        eegbci.standardize(raw)
        raw.resample(SFREQ)
        raw.pick(CHANNELS)

        # Extract event annotations 
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        keep = {k: v for k, v in event_id.items() if k in ("T1", "T2")}

        band_data = []
        for lo, hi in BANDS:
            # Apply butterworth bandpass filter for certain frequency band
            r = apply_butterworth_bandpass(raw, lo, hi)
            # Create fixed-length epochs around events
            ep = mne.Epochs(
                r, events, keep, tmin=TMIN, tmax=TMAX,
                baseline=None, preload=True, verbose=False
            )
            # Convert units from Volts to microVolts for numerical stability
            X_band = ep.get_data() * 1e6
            if align:
                X_band = euclidean_align(X_band)
            band_data.append(X_band)

        # Stack frequency bands and collect labels 
        X_runs.append(np.stack(band_data, axis=1).astype(np.float32))
        y_runs.append((ep.events[:, -1] == keep["T2"]).astype(int))

    return np.concatenate(X_runs), np.concatenate(y_runs)


def load_subject(sub):
    """
    Loads both imagery and executed runs for a subject.
    Catches and handles missing files or corrupted subject sessions safely.
    """
    try:
        Xi, yi = load_runs(sub, IMAGERY_RUNS, align=True)
        Xe, ye = load_runs(sub, EXEC_RUNS, align=True)
        return Xi, yi, Xe, ye
    except Exception as e:
        print(f"Subject {sub} failed: {e}")
        return None



class SpatioTemporalTangentSpace(BaseEstimator, TransformerMixin):

    def __init__(self, band_idx=(WIDE_IDX,), delay=4):
        self.band_idx = band_idx
        self.delay = delay

    def _embed_and_cov(self, Xb):
        covs = []
        d = self.delay
        for x in Xb:
            # Time-delay embedding to capture temporal phase dynamics
            if d > 0:
                x_aug = np.vstack([x[:, d:], x[:, :-d]])
            else:
                x_aug = x
            # Compute OAS covariance and normalize by its trace
            c = oas(x_aug.T)[0]
            covs.append(c / np.trace(c))
        return np.stack(covs)

    @staticmethod
    def _log_euclidean_mean(C):
        """Computes the geometric Log-Euclidean mean of a stack of covariance matrices."""
        logs = []
        for c in C:
            w, V = np.linalg.eigh(c)
            logs.append((V * np.log(np.maximum(w, 1e-12))) @ V.T)
        mean_log = np.mean(logs, axis=0)
        w, V = np.linalg.eigh(mean_log)
        return (V * np.exp(w)) @ V.T

    def fit(self, X, y=None):
        """Learns the reference covariance matrix from training data."""
        self.isqrts_ = []
        for b in self.band_idx:
            C = self._embed_and_cov(X[:, b].astype(np.float64))
            ref = self._log_euclidean_mean(C)
            w, V = np.linalg.eigh(ref)
            self.isqrts_.append((V / np.sqrt(np.maximum(w, 1e-12))) @ V.T)
        return self

    def transform(self, X):
        """Projects covariance matrices onto the tangent space and flattens upper triangles."""
        n = X.shape[2] * (2 if self.delay > 0 else 1)
        iu = np.triu_indices(n)
        scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))

        band_vecs = []
        for b, isqrt in zip(self.band_idx, self.isqrts_):
            C = self._embed_and_cov(X[:, b].astype(np.float64))
            vecs = []
            for c in C:
                # Tangent space mapping relative to reference point
                M = isqrt @ c @ isqrt
                w, V = np.linalg.eigh(M)
                L = (V * np.log(np.maximum(w, 1e-12))) @ V.T
                # Extract upper triangular elements into vector feature representation
                vecs.append(L[iu] * scale)
            band_vecs.append(np.array(vecs))

        return np.concatenate(band_vecs, axis=1)



def make_C_spatial():
    """Creates Pure spatial Riemannian tangent space + feature scaling + Shrinkage LDA."""
    return Pipeline([
        ("ts", SpatioTemporalTangentSpace((WIDE_IDX,), delay=0)),
        ("scale", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    ])


def make_E_spatiotemporal():
    """Creates Spatio-temporal tangent space with a delay of 4 along with scaling and Shrinkage LDA."""
    return Pipeline([
        ("ts", SpatioTemporalTangentSpace((WIDE_IDX,), delay=4)),
        ("scale", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    ])


def make_F_svm():
    """Creates the same things as Make_E but alos makes a Linear Support Vector Classifer"""
    return Pipeline([
        ("ts", SpatioTemporalTangentSpace((WIDE_IDX,), delay=4)),
        ("scale", StandardScaler()),
        ("svm", LinearSVC(C=0.02, dual="auto", max_iter=3000, random_state=42))
    ])


# Dictionary mapping that calls back to the make functions and whether they are executed or not
MODELS = {
    "C.       Spatial TS (8-30Hz) + shrink LDA":         (make_C_spatial, False),
    "C+exec. Spatial TS + executed training data":        (make_C_spatial, True),
    "E.       Spatio-Temporal TS (delay=4) + shrink LDA": (make_E_spatiotemporal, False),
    "E+exec. Spatio-Temporal TS + exec data + LDA":      (make_E_spatiotemporal, True),
    "F+exec. Spatio-Temporal TS + exec data + SVM":      (make_F_svm, True),
}



def score_subject(Xi, yi, Xe, ye):
    """
    Performs Repeated Stratified K-Fold cross-validation on motor imagery trials
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    splits = list(cv.split(Xi, yi))
    out = {}
    
    for name, (factory, use_exec) in MODELS.items():
        accs = []
        for tr, te in splits:
            Xtr, ytr = Xi[tr], yi[tr]
            # Does Cross-domain augmentation
            if use_exec:
                Xtr = np.concatenate([Xtr, Xe])
                ytr = np.concatenate([ytr, ye])
            # Fit model on training split and evaluate accuracy on test split
            m = factory().fit(Xtr, ytr)
            accs.append((m.predict(Xi[te]) == yi[te]).mean())
        out[name] = np.mean(accs)
    return out





if __name__ == "__main__":
    # Exclude unwanted subjects and load data in parallel across CPU cores
    subs = [s for s in SUBJECTS if s not in EXCLUDE]
    print("Loading subjects (Euclidean-aligned 8-30Hz imagery + execution runs)")
    loaded = Parallel(n_jobs=-1)(delayed(load_subject)(s) for s in subs)
    data = [r for r in loaded if r is not None]
    print(f"Loaded {len(data)}/{len(subs)} subjects")

    print("--- Scoring models per subject ---")
    results = Parallel(n_jobs=-1)(delayed(score_subject)(*d) for d in data)

    results_summary = {}

    # Aggregate and print detailed performance metrics across all subjects
    for name in results[0]:
        sc = np.array([r[name] for r in results])
        sc_pct = sc * 100  
        
        resp = sc_pct[sc_pct >= 70.0] # Filter for clinically viable / responsive subjects (>= 70% accuracy)
        resp_mean = float(resp.mean()) if len(resp) > 0 else 0.0

        # Save metrics into the summary dictionary
        results_summary[name] = {
            "mean": float(sc_pct.mean()),
            "std": float(sc_pct.std()),
            "median": float(np.median(sc_pct)),
            "above_70_count": int((sc_pct >= 70.0).sum()),
            "above_80_count": int((sc_pct >= 80.0).sum()),
            "total_subjects": int(len(sc_pct)),
            "responsive_mean": resp_mean
        }

        resp_mean_str = f"{resp_mean:.2f}%" if len(resp) > 0 else "N/A"
        print(
            f"[{name}] -> All Mean: {sc_pct.mean():.2f}% (+/- {sc_pct.std():.2f}%) "
            f"| Median: {np.median(sc_pct):.2f}% "
            f"| >=70%: {(sc_pct >= 70.0).sum()}/{len(sc_pct)} (Responsive Mean: {resp_mean_str}) "
            f"| >=80%: {(sc_pct >= 80.0).sum()}/{len(sc_pct)}"
        )

    # Dump the complete dictionary to a JSON file for plotting or reporting
    with open("improved_pipeline_results.json", "w") as f:
        json.dump(results_summary, f, indent=4)
    print("\nSaved benchmark results to 'improved_pipeline_results.json'")

    print("\nBenchmark Complete")