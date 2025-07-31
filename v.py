import numpy as np
import matplotlib.pyplot as plt

X = np.load('X_noisy.npy')
y = np.load('y_noisy.npy')

# Separate by class
real = X[y == 0]
spoofed = X[y == 1]

# Pick 3 random from each
for i in range(3):
    plt.figure(figsize=(10, 4))
    plt.plot(real[i], label='Real Target', color='green')
    plt.plot(spoofed[i], label='Spoofed Target', color='red', linestyle='--')
    plt.title(f"Real vs Spoofed Signal Sample #{i+1}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"sample_{i+1}.png")
    plt.show()
