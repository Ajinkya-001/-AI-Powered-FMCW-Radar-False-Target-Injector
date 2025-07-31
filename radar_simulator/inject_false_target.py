import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

def generate_false_echo(tx_signal, sample_rate, delay_s, attenuation=0.5, doppler_shift=0.0):
    """
    Generate a delayed and optionally Doppler-shifted version of tx_signal
    as a false echo.
    
    Parameters:
    - tx_signal: numpy array, original transmitted signal
    - sample_rate: float, samples per second
    - delay_s: float, delay in seconds
    - attenuation: float, scaling factor for false echo amplitude
    - doppler_shift: float, frequency shift in Hz
    
    Returns:
    - false_echo: numpy array, aligned with tx_signal length
    """
    delay_samples = int(delay_s * sample_rate)
    t = np.arange(len(tx_signal)) / sample_rate

    # Apply Doppler shift (frequency offset)
    doppler_phase = np.exp(1j * 2 * np.pi * doppler_shift * t)

    # Shift signal and apply attenuation
    shifted = np.zeros_like(tx_signal, dtype=complex)
    valid_length = len(tx_signal) - delay_samples
    if valid_length > 0:
        shifted[delay_samples:] = attenuation * tx_signal[:valid_length] * doppler_phase[:valid_length]
    
    return shifted


def inject_false_targets(rx_signal, tx_signal, sample_rate, targets):
    """
    Inject multiple false echoes into rx_signal.
    
    targets = [
        {"delay": 2e-6, "atten": 0.8, "doppler": 50},
        {"delay": 3.5e-6, "atten": 0.4, "doppler": -30},
    ]
    """
    rx_modified = rx_signal.astype(complex)
    
    for target in targets:
        echo = generate_false_echo(
            tx_signal,
            sample_rate=sample_rate,
            delay_s=target.get("delay", 0),
            attenuation=target.get("atten", 0.5),
            doppler_shift=target.get("doppler", 0.0)
        )
        rx_modified += echo
    return rx_modified


# Example Usage
if __name__ == "__main__":
    SAMPLE_RATE = 20e6  # 20 MHz
    DURATION = 5e-5     # 50 us
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)

    # Simulate transmitted chirp
    tx = chirp(t, f0=1e6, f1=10e6, t1=DURATION, method='linear')

    # Simulate received echo (clean)
    rx = np.copy(tx)

    # Inject synthetic false targets
    targets = [
        {"delay": 1.5e-6, "atten": 0.7, "doppler": 1500},
        {"delay": 2.5e-6, "atten": 0.4, "doppler": -800}
    ]
    rx_with_false = inject_false_targets(rx, tx, SAMPLE_RATE, targets)

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(t * 1e6, np.real(rx), label='Original Echo')
    plt.plot(t * 1e6, np.real(rx_with_false), label='With False Targets', alpha=0.7)
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude")
    plt.title("Radar Signal with False Echoes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("false_target.png")
