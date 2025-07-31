# radar_simulator/false_target_injector.py
import numpy as np

def inject_false_targets(rx_signal, tx_signal, fs, false_targets):
    """
    Injects false targets into the received radar signal.

    Args:
        rx_signal (np.ndarray): Original received signal
        tx_signal (np.ndarray): Transmitted signal
        fs (int): Sampling frequency
        false_targets (list of tuples): Each tuple contains (distance_m, amplitude)

    Returns:
        np.ndarray: Modified rx_signal with false targets
    """
    c = 3e8  # Speed of light in m/s
    modified_rx = rx_signal.copy()
    
    for distance, amplitude in false_targets:
        delay = 2 * distance / c  # Round trip delay in seconds
        delay_samples = int(delay * fs)

        # Create a delayed and scaled version of tx_signal
        fake_echo = np.zeros_like(rx_signal)
        if delay_samples + len(tx_signal) < len(rx_signal):
            fake_echo[delay_samples:delay_samples+len(tx_signal)] = amplitude * tx_signal
        else:
            # Clip to avoid out-of-bounds
            available_len = len(rx_signal) - delay_samples
            if available_len > 0:
                fake_echo[delay_samples:] = amplitude * tx_signal[:available_len]
        
        modified_rx += fake_echo

    return modified_rx
