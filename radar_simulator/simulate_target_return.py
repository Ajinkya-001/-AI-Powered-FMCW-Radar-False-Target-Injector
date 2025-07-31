import numpy as np
from scipy.signal import resample

def simulate_echo(transmitted, fs, delay_us=10, attenuation=0.5):
    """
    Simulate radar echo by delaying and attenuating the transmitted chirp.

    Args:
        transmitted (np.ndarray): The original transmitted chirp
        fs (float): Sampling rate (Hz)
        delay_us (float): Delay in microseconds (related to target distance)
        attenuation (float): Echo strength [0-1]

    Returns:
        np.ndarray: Simulated received signal
    """
    delay_samples = int((delay_us * 1e-6) * fs)
    echo = np.pad(transmitted, (delay_samples, 0), 'constant')[:len(transmitted)]
    return attenuation * echo
