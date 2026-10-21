import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import read_raw_edf

SUBJECTS = range(1, 11)          # subjects 1 to 10
RUNS = [4, 8, 12]                # imagined left vs. right fist
MOTOR_CHANNELS = ["C3", "Cz", "C4"]

all_X = []
all_y = []

for subject in SUBJECTS:
    print(f"\n--- Subject {subject} ---")
    try:
        paths = eegbci.load_data(subject, RUNS)
        raws = [read_raw_edf(p, preload=True, verbose=False) for p in paths]
        raw = mne.concatenate_raws(raws)
        raw.rename_channels(lambda x: x.strip("."))

        events, event_id = mne.events_from_annotations(raw, verbose=False)

        epochs = mne.Epochs(
            raw, events,
            event_id={"Left": 2, "Right": 3},
            tmin=-0.5, tmax=3.0,
            baseline=None,       # State 0: no cleaning yet
            preload=True,
            verbose=False,
        )

        X_subject = epochs.copy().pick(MOTOR_CHANNELS).get_data()
        y_subject = epochs.events[:, -1] - 2    # 2 -> 0 (left), 3 -> 1 (right)

        all_X.append(X_subject)
        all_y.append(y_subject)
        print(f"Got {X_subject.shape[0]} trials")

    except Exception as e:
        print(f"Problem with subject {subject}: {e}")

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)

print("\n=== Done ===")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Left:", (y == 0).sum(), "Right:", (y == 1).sum())