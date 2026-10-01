##ReadMe

# Benchmarking How Artifact Cleaning Modulates Predictive Performance Across ML Models for BCI.

**Course:** BU CS 506 - Data Science Tools & Applications  
**Group Members:** Khushi Patel and Matthew Ortega, Gerardo Vega
**Repository:**  


## 1. Project Description

Brain-Computer Interfaces are a promising intervention for neurological disorder and neurodegenerative diseases, currently being tested and utilized for specialized cases of Parkinson's, motor impairment, seizure disorder, and more. The technology extracts EEG signals and then pre-processes the signals with feature extraction and classification networks/regressions in order to detect peaks for motor intent and initiation. The problem with EEG data, however, is that it is messy and heavily susceptible to physiological distortions from artifacts such as eye blinks and saccades (EOG), muscle clenching (EMG), sweating, facial/jaw clenching etc. Additionally, environmental powerline noise at 60Hz masks high-frequency neural dynamics. The problem with simply extracting these frequency bands is that some components may distort phase relationships (McFarland & Wolpaw, 2017), alter waveform morphology (Neural Engineering Monograph, 2024), and accidentally attenuate sensorimotor mu (7–13 Hz) and beta (15–30 Hz) rhythms that encode motor imagery intent (Marissens Cueva et al., 2025; McFarland & Wolpaw, 2017). 

Our project aims to identify a benchmark for signal processing by applying filters and testing the classification of data before and after treatment of each of these artifacts. We intend to utilize a Digital Signal Processing (DSP) pipeline that inputs raw EEG data from a PhysioNet dataset that reads from 64 electrodes EEG scans (64-channel). Our project establishes a systematic artifact ablation benchmarking system to evaluate how isolated artifact removal steps impact classification performance across five distinct machine learning paradigms. By quantifying the Signal-to-Noise Ratio (SNR) gain, we expect for this repository to inform better feature extraction and noise suppression to inform BCI design.

Marissens Cueva, V., Bougrain, L., Lotte, F., & Rimbert, S. (2025). Reliable predictor of BCI motor imagery performance using median nerve stimulation. Journal of Neural Engineering, 22(1), 016012. https://doi.org/10.1088/1741-2552/adc48d[2][9]
McFarland, D. J., & Wolpaw, J. R. (2017). EEG-based brain-computer interfaces. Current Opinion in Biomedical Engineering, 4, 194–200. https://doi.org/10.1016/j.cobme.2017.11.004[4]
Neural Engineering Monograph. (2024). Brain-computer interfaces & signal processing pipeline: Foundations of neural signals in BCI systems. Academic Monograph

---

## 2. Goals 

BCIs that evaluate motor intent typically reach 75-80% accuracy, with 30% of individuals showing BCI illiteracy or difficulty generating discriminative sensorimotor rhythms.

 Our objective is to quantify the performance gain provided by targeted Digital Signal Processing (DSP) by coding a sequential artifact ablation system to test raw vs. cleaned EEG feature representations across five modeling paradigms (Logistic Regression, Decision Trees, XGBoost, 1D-CNN, and K-Means). We aim to achieve a post DSP of 80% for left and right hand motor intent from PhysioNet. Additionally, we plan to use unsupervised K-Means clustering to distinguish trial segments into low- vs. high-noise profiles to automatically identify latent “BCI illiteracy” in the data. The four artifacts we aim to test individually to calculate amount of artifact power removed with the accuracy gain are: 
60 Hz Powerline Peak: Measure decrease in narrow-band power before vs. after filtering
Ocular (EOG) Spikes: Measure decrease in of 100–200 µV frontal voltage deflections
Muscle (EMG) Noise: Measure high-frequency (20–300 Hz) spectral power ratio over motor channels (C3, Cz, C4)
Galvanic DC Drift (Sweat): Measure the reduction in slow (<0.5 Hz) baseline potential shifts


## 3. Data Collection/Procedure 

The primary data set we are pulling from is the OpenSource PhysioNet EEG Motor Movement/Imagery Database  (https://physionet.org/content/eegmmidb/1.0.0/): 
This provides “64-channel EEG recordings sampled at 160Hz from 109 subjects executing motor/imagery tasks”. It can be programmatically downloaded via mne.datasets.eegbci in Python. The same data set is also available via Kaggle,  Kaggle - PhysioNet EEG Motor Movement / Imagery Dataset . 
Automatic ingestion via: `mne.datasets.eegbci.load_data()` API within `src/data_loader.py`
How to Interpret the Dataset
Out of the 14 experimental runs for each participant in the dataset, we are interested in isolating the Left v Right hand tasks. For motor execution, those are runs 3, 7, and 11. For motor imagery, that is runs 4, 8, and 12. Runs 1 and 2 are eyes open and closed at rest, respectively. These can be used as baseline normalizations for baseline noise estimates.  
The annotations from the study are also extractable via mne.events_from_annotations(), therefore, event markers T0 (Rest), T1 (Left Hand), and T2 (Right Hand) can also be implemented. 
We can pinpoint electrode coordinates via  mne.channels.make_standard_montage('standard_1020') 
Even though there were 64 electrodes used, the feature extraction focuses on the motor cortex channels around C3, Cz, C4 ( Fc1, FC2, CP1, and Cp2) to extract the sensorimotor mu at (7-13 Hz) and beta ( 15-30 Hz). 
Signal Processing and Artifact Ablation 

Epoch the data into 3 second intervals centered around the  T1 and T2 cue 
Measure the pre-cue baseline voltage at -1.5s - 0.5s for each channel and trial epoch and perform a voltage subtraction/z-score scaling to account for DC shifts

State 0: Baseline, uncleaned signals
State 1: Apply the narrow band IIR notch filter scipy.signal.iirnotch  to measure the drop in 60 Hz ( powerline) and assess the amplifier range recovery afterwards
State 2: Use independent component analysis (mne.preprocessing.ICA) to isolate frontal components with high amplitudes ( 100-200 ) µV deflections in the 0.5-6 hz delta/theta band Ocular (EOG) Spike
State 3: Apply a zero-phase 4th-order Butterworth bandpass filter (8–30 Hz) to eradicate high-frequency 20–300 Hz muscle contamination that directly overlaps the motor beta band (15–30 Hz) over motor channels (C3, Cz, C4) for Muscle (EMG) Noise Isolation
State 4: Subtracts mean pre-stimulus voltage to eliminate slow <0.5 Hz baseline drifts and stabilize time-domain Root Mean Square (RMS) measurements for Galvanic DC Drift/ Sweat Isolation 
State 5: Combines all four DSP filters to evaluate peak classification performance against our 80% accuracy 
Feature Extraction
Spectral: Power Spectral Density (PSD) estimated via Welch's method across mu (7–13 Hz) and beta (15–30 Hz) 
Temporal : Root Mean Square (RMS) voltage, Mean Absolute Value (MAV), and Signal-to-Noise Ratio
Spatial : Common Spatial Patterns (CSP) spatial variance transformations across motor channels (C3, Cz, C4) to max variance diff

Modeling and Benchmarking

Supervised Intent Prediction:  Feature matrices  pass through to four supervised classifiers (Logistic Regression, Decision Trees, XGBoost, and 1D-CNN) to evaluate Left vs. Right hand classification accuracy, aiming for 80% accuracy
Unsupervised Noise: Unsupervised K-Means clustering for unlabeled feature variance and spectral entropy to split data into low-noise vs. high-noise to automatically disqualify subjects with BCI illiteracy (~30% low-responder population)

Data Visualization
Time-Series Plots: Comparing raw vs. DSP-cleaned .
Frequency Spectra: Power Spectral Density (PSD) showing the elimination of 60 Hz line noise.
Topographic Scalp Heatmaps: 2D cortical maps (mne.viz.plot_topomap) showing spatial mu/beta power distribution over motor areas.
Model Benchmarks: Confusion Matrices, ROC-AUC curves, and classification accuracy tables comparing performance across the 6 ablation states
## 3. Estimated Timeline:
Week 1: Planning/ Downloading relevant toolkits
Week 2: Epoching and measuring pre-cue baseline
Week 3: Artifact Ablation ( task split amongst group members)
Week 4: Artifact Ablation ( task split amongst group members) 
Week 5: Feature Extraction ( split amongst team members)
Week 6: Feature Extraction ( split amongst team members) 
Week 7: Modelling and Benchmarking:  Supervised Intent Prediction:
Week 8: Modelling and Benchmarking: Supervised Intent Prediction:
Week 9: Modelling and Benchmarking : Unsupervised 
