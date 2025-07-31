from ai_module.dataset import RadarDataset
from torch.utils.data import DataLoader

dataset = RadarDataset(data_path='X_noisy.npy', labels_path='y_noisy.npy')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for i, (signal, label) in enumerate(loader):
    print(f"Batch {i+1}")
    print("Signal shape:", signal.shape)  # Should be [32, 512]
    print("Labels:", label)               # Should be tensor([0, 1, ...])
    if i == 1: break  # Just show 2 batches to test
