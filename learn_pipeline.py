"""
State 0 pipeline: uncleaned EEG epochs, all 64 channels.

Downloads PhysioNet EEGBCI imagery runs (4, 8, 12: imagined left vs. right fist),
epochs from -0.5 to 3.0 s around each T1/T2 cue with NO baseline correction
(baseline subtraction belongs to State 4), and saves:
  X_state0_all.npy : (n_trials, 64, 561) float32, volts
  y_benchmark.npy  : (n_trials,) 0 = left, 1 = right
"""
import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import read_raw_edf

SUBJECTS = range(1, 11)                  # subjects 1 to 10 (dataset has 109)
RUNS = [4, 8, 12]                        # imagined left vs. right fist
MOTOR_CHANNELS = ["C3", "Cz", "C4"]

all_X = []
all_y = []
ch_names = None

for subject in SUBJECTS:
    print(f"\n--- Subject {subject} ---")
    try:
        paths = eegbci.load_data(subject, RUNS)
        raws = [read_raw_edf(p, preload=True, verbose=False) for p in paths]
        raw = mne.concatenate_raws(raws)
        raw.rename_channels(lambda x: x.strip("."))

        # ---- CLEANING GOES HERE (States 1-5 plug in on `raw`) ----
        # e.g. raw = apply_notch(raw)

        events, event_id = mne.events_from_annotations(raw, verbose=False)

        epochs = mne.Epochs(
            raw, events,
            event_id={"Left": 2, "Right": 3},
            tmin=-0.5, tmax=3.0,
            baseline=None,           # State 0: no cleaning yet
            preload=True,
            verbose=False,
        )

        all_X.append(epochs.get_data().astype(np.float32))   # all 64 channels
        all_y.append(epochs.events[:, -1] - 2)                # 2 -> 0, 3 -> 1
        ch_names = epochs.ch_names
        print(f"Got {all_X[-1].shape[0]} trials")

    except Exception as e:
        print(f"Problem with subject {subject}: {e}")

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)

motor_idx = [ch_names.index(c) for c in MOTOR_CHANNELS]
X_motor = X[:, motor_idx, :]

print("\n=== Done ===")
print("X (all channels):", X.shape)
print("X_motor:", X_motor.shape)
print("y:", y.shape, "| Left:", (y == 0).sum(), "Right:", (y == 1).sum())

np.save("X_state0_all.npy", X)
np.save("y_benchmark.npy", y)
print("Saved X_state0_all.npy and y_benchmark.npy")