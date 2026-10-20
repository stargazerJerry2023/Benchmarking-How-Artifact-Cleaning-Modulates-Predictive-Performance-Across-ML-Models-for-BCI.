import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
import numpy as np

"""

The scaled version of `testFileImport.py` contains more data values than the original `testFileImport.py`. 
However, it currently has a slow runtime, 
which we in the future plan to optimize to reduce execution time from the current very long time.

"""


# PhysioNet has 109 subjects this scales it to 10 subjects
SUBJECTS = list(range(1, 11)) 

# 4 (left/right fist imagination), 8 (open/close hands), 12 (imagine opening/closing hands)
RUNS = [4, 8, 12]  
MOTOR_CHANNELS = ['C3', 'Cz', 'C4']

all_epochs_data = []
all_labels = []

print(f"Starting Full Benchmark Pipeline ({len(SUBJECTS)} Subjects)")

for subject in SUBJECTS:
    print(f"\nProcessing Subject {subject}/{len(SUBJECTS)}")
    
    try:
        # Download and load all runs for this subject
        raw_fnames = eegbci.load_data(subject, RUNS, update_path=False)
        raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
        raw = mne.concatenate_raws(raws)
        
        # Clean channel names
        raw.rename_channels(lambda x: x.strip('.'))
        
        # Extract events
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # Create epochs (T1 = Left Hand (label 0), T2 = Right Hand (label 1))
        epochs = mne.Epochs(
            raw, 
            events, 
            event_id={'Left_Hand_T1': 2, 'Right_Hand_T2': 3}, 
            tmin=-0.5, 
            tmax=3.0, 
            baseline=(-0.5, 0), 
            preload=True,
            verbose=False
        )
        
        # Isolate motor channels and gets numpy data: shape (n_epochs, n_channels, n_times)
        epochs_motor = epochs.copy().pick(MOTOR_CHANNELS)
        data = epochs_motor.get_data()
        
        # Map MNE event IDs to binary ML labels (0 and 1)
        # T1 (code 2) -> 0, T2 (code 3) -> 1
        labels = epochs.events[:, -1] - 2 
        
        all_epochs_data.append(data)
        all_labels.append(labels)
        print(f"-> Success! Collected {data.shape[0]} trials for Subject {subject}.")
        
    except Exception as e:
        print(f"-> Error processing Subject {subject}: {e}")

# Stack everything into Master Matrices for Machine Learning
if all_epochs_data:
    X = np.concatenate(all_epochs_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print("MASTER BENCHMARK DATASET READY FOR ML")
    print(f"Total Feature Matrix Shape (X): {X.shape}")  # (N_total_trials, 3 channels, 561 timepoints)
    print(f"Total Labels Shape (y):         {y.shape}")
    print(f"Class Balance:                  Left Hand: {np.sum(y == 0)}, Right Hand: {np.sum(y == 1)}")
    
    # Saves compiled arrays 
    np.save("X_benchmark_state0.npy", X)
    np.save("y_benchmark.npy", y)
    print("Saved master arrays as 'X_benchmark_state0.npy' and 'y_benchmark.npy'!")
else:
    print("No data was successfully collected.")