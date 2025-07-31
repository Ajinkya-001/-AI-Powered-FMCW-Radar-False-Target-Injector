import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from radar_simulator.simulate_target_return import simulate_echo
from radar_simulator.generate_radar_pulse import generate_fmcw_chirp
from ai_module.utils import add_awgn  # already defined

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


X = []
y = []

fs = 10e6  # 10 MHz
num_samples = 512
num_signals = 1000  # More samples = stronger generalization

snr_range = (5, 25)

for i in range(num_signals):
    chirp, _, _ = generate_fmcw_chirp(
        f_start=77e9, B=150e6, T=1e-3, fs=fs, plot=False
    )

    # Base real echo (always present)
    real_delay = np.random.uniform(10, 14)
    real_atten = np.random.uniform(0.7, 0.95)
    echo = simulate_echo(chirp, fs, delay_us=real_delay, attenuation=real_atten)

    is_spoofed = np.random.rand() < 0.5

    if is_spoofed:
        # Add spoofed signal
        spoof_delay = np.random.uniform(12, 20)
        spoof_atten = np.random.uniform(0.6, 1.0)
        spoof_echo = simulate_echo(chirp, fs, delay_us=spoof_delay, attenuation=spoof_atten)
        echo += spoof_echo
        label = 1
    else:
        label = 0

    # Add ghost echoes
    for _ in range(np.random.randint(0, 3)):
        ghost_delay = np.random.uniform(25, 40)
        ghost_atten = np.random.uniform(0.1, 0.4)
        ghost_echo = simulate_echo(chirp, fs, delay_us=ghost_delay, attenuation=ghost_atten)
        echo += ghost_echo

    # Inject burst noise sometimes
    if np.random.rand() < 0.1:
        echo += np.random.normal(0, 0.5, size=echo.shape)

    # Randomize length
    clip_len = np.random.randint(450, 512)
    echo = echo[:clip_len]
    if clip_len < 512:
        echo = np.pad(echo, (0, 512 - clip_len), mode='constant')

    # AWGN
    snr = np.random.uniform(*snr_range)
    echo_noisy = add_awgn(echo, snr)

    X.append(echo_noisy)
    y.append(label)

X = np.array(X)
y = np.array(y)

np.save("X_noisy_v2.npy", X)
np.save("y_noisy_v2.npy", y)

print("✅ Chaos dataset saved:")
print("   🔸 Shape:", X.shape)
print("   🔹 Real:", np.sum(np.array(y) == 0), "| Spoofed:", np.sum(np.array(y) == 1))
