import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
import matplotlib.pyplot as plt

"""

This file demonstrates our baseline data ingestion. 
It extracts a specific run from the PhysioNet database to show raw EEG brain activity during hand movement tasks, 
plotting the waveform traces across our target motor channels

"""

print("Downloads/Loading PhysioNet Data (Subject 1, Run 3)")
subject = 1
runs = [3]  
raw_fnames = eegbci.load_data(subject, runs)
raw = read_raw_edf(raw_fnames[0], preload=True)
raw.rename_channels(lambda x: x.strip('.'))

print("Extracts Events & Epochs")
events, event_id = mne.events_from_annotations(raw)

tmin, tmax = -0.5, 3.0
epochs = mne.Epochs(
    raw, 
    events, 
    event_id={'Left_Hand_T1': 2, 'Right_Hand_T2': 3}, 
    tmin=tmin, 
    tmax=tmax, 
    baseline=(-0.5, 0), 
    preload=True
)

motor_channels = ['C3', 'Cz', 'C4']
epochs_motor = epochs.copy().pick(motor_channels)

print("Generating & Saves the Visualization")
# Generate the plot and capture the figure object
fig = epochs_motor.plot(picks=motor_channels, title=f"State 0: Raw Motor Channels - Subject {subject}", show=False)

# Save it directly into your CS506 folder as an image file
fig.savefig("state0_raw_channels.png", dpi=300, bbox_inches='tight')
print("The Plot is successfully saved as 'state0_raw_channels.png' in the folder!")

# Force the interactive window to stay open
plt.show()

