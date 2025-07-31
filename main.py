from radar_simulator.generate_radar_pulse import generate_fmcw_chirp
from radar_simulator.false_target_injector import inject_false_targets
import matplotlib.pyplot as plt

# Generate radar chirp
tx_signal, t, fs = generate_fmcw_chirp()

# Simulate RX signal as a delayed copy of TX (basic echo)
rx_signal = tx_signal.copy()

# Define false targets: (distance in meters, amplitude)
false_targets = [
    (150, 0.7),  # 150m ghost blip
    (300, 0.5),  # 300m phantom
]

# Inject ghost targets
rx_with_fakes = inject_false_targets(rx_signal, tx_signal, fs, false_targets)

# Plot real vs spoofed
plt.figure(figsize=(12, 6))
plt.plot(tx_signal, label="TX Signal (original)")
plt.plot(rx_signal, label="RX Signal (real echo)", alpha=0.7)
plt.plot(rx_with_fakes, label="RX + False Targets", linestyle="--", linewidth=1)
plt.legend()
plt.title("Radar Echo with Injected False Targets")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.savefig("spoofed_echo.png")
plt.show()

print("[✓] Spoofed radar echo saved as spoofed_echo.png")

# Plot zoomed-in comparison for first 2000 samples
plt.figure(figsize=(12, 5))
plt.plot(tx_signal[:2000], label="TX", color='blue', linewidth=1)
plt.plot(rx_signal[:2000], label="RX", color='orange', alpha=0.6)
plt.plot(rx_with_fakes[:2000], label="RX + Ghosts", color='green', linestyle='--', alpha=0.8)
plt.title("Zoomed-In Radar Echo (First 2000 Samples)")
plt.legend()
plt.tight_layout()
plt.savefig("zoomed_spoofed_echo.png")
print("[✓] Zoomed-in spoofed radar echo saved as zoomed_spoofed_echo.png")

from ai_module.model import FalseTargetGenerator
model = FalseTargetGenerator()
model.load_state_dict(torch.load("false_target_gen.pt"))
model.eval()

# Real-time signal → AI → inject false target
with torch.no_grad():
    modified = model(torch.tensor(real_signal).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
