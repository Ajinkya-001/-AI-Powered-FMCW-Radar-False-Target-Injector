import numpy as np

def add_awgn_noise(signal, snr_db):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal.

    Args:
        signal (np.ndarray): Input signal.
        snr_db (float): Desired SNR in dB.

    Returns:
        np.ndarray: Noisy signal.
    """
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise
