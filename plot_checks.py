import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

X = np.load("X_state0_all.npy")      # (trials, 64, 561), in volts
y = np.load("y_benchmark.npy")       # 0 = left, 1 = right

SFREQ = 160
times = np.linspace(-0.5, 3.0, X.shape[2])

MOTOR = {"C3": 8, "Cz": 10, "C4": 12}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
for name, i in MOTOR.items():
    trace = X[0, i] - X[0, i].mean()
    ax.plot(times, trace * 1e6, label=name)
ax.axvline(0, color="k", linestyle="--")
ax.set(title="Single trial, State 0 (mean removed for display)",
       xlabel="Time (s)", ylabel="µV")
ax.legend()

ax = axes[0, 1]
for label, name in [(0, "Left"), (1, "Right")]:
    for ch in ["C3", "C4"]:
        avg = X[y == label, MOTOR[ch]].mean(axis=0)
        ax.plot(times, (avg - avg.mean()) * 1e6, label=f"{name} – {ch}")
ax.axvline(0, color="k", linestyle="--")
ax.set(title="Average across trials", xlabel="Time (s)", ylabel="µV")
ax.legend(fontsize=8)

ax = axes[1, 0]
for name, i in MOTOR.items():
    freqs, psd = welch(X[:, i, :], fs=SFREQ, nperseg=256, axis=-1)
    ax.semilogy(freqs, psd.mean(axis=0), label=name)
ax.axvline(60, color="r", linestyle=":", label="60 Hz")
ax.axvspan(8, 30, color="gray", alpha=0.15, label="8–30 Hz band")
ax.set(title="Power spectrum (mean over trials)",
       xlabel="Frequency (Hz)", ylabel="Power (V²/Hz)", xlim=(0, 80))
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.bar(["Left", "Right"], [(y == 0).sum(), (y == 1).sum()])
ax.set(title=f"Class balance ({len(y)} trials)", ylabel="Trials")

plt.tight_layout()
plt.savefig("state0_overview.png", dpi=200)
print("Saved state0_overview.png")
plt.show()