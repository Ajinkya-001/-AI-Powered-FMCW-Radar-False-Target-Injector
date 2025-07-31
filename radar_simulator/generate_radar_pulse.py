import numpy as np
import matplotlib.pyplot as plt

def generate_fmcw_chirp(f_start=77e9, B=150e6, T=1e-3, fs=10e6, plot=False):
    k = B / T
    t = np.arange(0, T, 1/fs)
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    signal = np.cos(phase)

    if plot:
        zoom_window = int(fs * 10e-6)
        plt.figure(figsize=(10, 6))
        plt.plot(t[:zoom_window]*1e6, signal[:zoom_window])
        plt.title("FMCW Radar Chirp (Time Domain Zoomed)")
        plt.xlabel("Time (µs)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("fmcw_chirp_zoomed.png")
        plt.show()

    return signal, t ,fs 
