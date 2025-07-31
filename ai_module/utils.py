import numpy as np

def add_awgn(signal, snr_db):
    """
    Add Gaussian noise to a signal given a target SNR (in dB).
    """
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise
