import numpy as np 
from radar_simulator.simulate_target_return import simulate_echo
from radar_simulator.generate_radar_pulse import generate_fmcw_chirp
from ai_module.utils import add_awgn  # Must be defined as shown below

X = []
y = []

fs = 10e6  # 10 MHz
num_samples = 512
num_signals = 500

snr_range = (10, 30)  # Battlefield-realistic SNR

for i in range(num_signals):
    chirp, _, _ = generate_fmcw_chirp(
        f_start=77e9, B=150e6, T=1e-3, fs=fs, plot=False
    )

    if i % 2 == 0:
        # Real target
        echo = simulate_echo(chirp, fs, delay_us=12, attenuation=0.8)
        label = 0
    else:
        # Spoofed target
        echo = simulate_echo(chirp, fs, delay_us=18, attenuation=1.0)
        label = 1

    echo = echo[:num_samples]
    
    # Add Gaussian noise
    snr = np.random.uniform(*snr_range)
    echo_noisy = add_awgn(echo, snr)

    X.append(echo_noisy)
    y.append(label)

X = np.array(X)
y = np.array(y)

np.save("X_noisy.npy", X)
np.save("y_noisy.npy", y)

print("✅ Saved noisy dataset with shape:", X.shape)
