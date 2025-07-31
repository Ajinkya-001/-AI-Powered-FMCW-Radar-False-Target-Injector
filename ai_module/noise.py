import numpy as np

# Gaussian Noise
def add_gaussian_noise(signal, mean=0, std=0.1):
    noise = np.random.normal(mean, std, size=signal.shape)
    return signal + noise

# Random Amplitude Dropout (simulate weak reflection)
def amplitude_dropout(signal, drop_prob=0.1):
    mask = np.random.binomial(1, 1 - drop_prob, size=signal.shape)
    return signal * mask

# Random Jitter (simulate Doppler chaos or multipath jitter)
def signal_jitter(signal, max_shift=3):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(signal, shift)

# Echo overlap (simulate multiple returns or ghost targets)
def add_echo_overlap(signal, delay=20, attenuation=0.5):
    echo = np.roll(signal, delay) * attenuation
    return signal + echo
