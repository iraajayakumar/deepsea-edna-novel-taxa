from paired_dataset import PairedFCGRDataset
from network import MultiKFCGRNet
import torch

print("=== TESTING DATASET + MODEL ===")
dataset = PairedFCGRDataset('data/interim/fcgr.pkl', 'data/interim/fcgr_mimic.pkl')
print(f"Dataset length: {len(dataset)}")

orig_fcgr, mimic_fcgr = dataset[0]
print("Orig keys:", list(orig_fcgr.keys()))
print("Orig k=4 shape:", orig_fcgr['4'].shape)
print("Mimic k=4 shape:", mimic_fcgr['4'].shape)

# Test model input - ensure string keys
model = MultiKFCGRNet()
print("Testing orig_fcgr keys:", list(orig_fcgr.keys()))  # Should show ['4','5','6']
batch_orig = {k: v.unsqueeze(0) for k,v in orig_fcgr.items()}  # B=1
print("Batch shapes:", {k: v.shape for k,v in batch_orig.items()})
probs = model(batch_orig)
print("✅ Model output shape:", probs.shape)  # torch.Size([1, 80])

print("\n🎉 ALL TESTS PASSED! Ready for training!")

