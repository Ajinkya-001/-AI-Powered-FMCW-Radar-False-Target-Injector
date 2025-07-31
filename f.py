import os
output_path = os.path.join(os.getcwd(), "radar_classifier.h5")
model.save(output_path)
print(f"✅ Model saved at {output_path}")
