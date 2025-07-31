import matplotlib.pyplot as plt
import numpy as np

def plot_signal(signal, sample_rate, title="Signal", filename="output.png"):
    """
    Plot real part of signal over time.

    Args:
        signal (np.ndarray): Input signal.
        sample_rate (float): Sample rate in Hz.
        title (str): Plot title.
        filename (str): File path to save.
    """
    t = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(t, np.real(signal))
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"radar_simulator/{filename}")
    plt.close()
    plt.savefig("utlis.png")
