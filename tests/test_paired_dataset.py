import pickle
from pathlib import Path
import torch

# Paths to your data
FCGR_ORIG_PATH = Path('data/interim/fcgr.pkl')
FCGR_MIMIC_PATH = Path('data/interim/fcgr_mimic.pkl')

print("=== LOADING DATA ===")
with open(FCGR_ORIG_PATH, 'rb') as f:
    orig_data = pickle.load(f)
print(f"Original FCGR: {type(orig_data)} with {len(orig_data)} sequences")
print("Sample orig structure:", list(orig_data.keys())[:2])
print("Sample orig entry keys:", list(orig_data[list(orig_data.keys())[0]].keys()))

with open(FCGR_MIMIC_PATH, 'rb') as f:
    mimic_data = pickle.load(f)
print(f"Mimic FCGR: {type(mimic_data)} with {len(mimic_data)} sequences")

# Handle different structures
if isinstance(mimic_data, dict):
    print("Sample mimic structure:", list(mimic_data.keys())[:2])
    common_ids = sorted(set(orig_data.keys()) & set(mimic_data.keys()))
elif isinstance(mimic_data, list):
    print("Mimic is LIST - checking first entry structure")
    print("First mimic entry:", type(mimic_data[0]) if mimic_data else "Empty")
    # Assume list of dicts with seq_id as key or index matching originals
    common_ids = list(range(min(len(orig_data), len(mimic_data))))
else:
    print(f"Unexpected mimic type: {type(mimic_data)}")
    common_ids = []

print(f"\n=== PAIRED SEQUENCES ===")
print(f"Found {len(common_ids)} paired sequences")
print("First 5 IDs/indices:", common_ids[:5])

# Test single sample
if common_ids:
    if isinstance(mimic_data, dict):
        seq_id = common_ids[0]
        orig_sample = orig_data[seq_id]
        mimic_sample = mimic_data[seq_id]
        print(f"\n=== DICT SAMPLE {seq_id} ===")
    else:  # list
        idx = common_ids[0]
        seq_id = list(orig_data.keys())[idx] if isinstance(orig_data, dict) else idx
        orig_sample = orig_data[seq_id] if isinstance(orig_data, dict) else orig_data[idx]
        mimic_sample = mimic_data[idx]
        print(f"\n=== LIST SAMPLE index {idx} (seq_id: {seq_id}) ===")
    
    print("Orig keys:", list(orig_sample.keys()) if isinstance(orig_sample, dict) else "Not dict")
    print("Orig abundance:", orig_sample.get('abundance', 'N/A'))
    
    if isinstance(orig_sample, dict) and 'fcgr' in orig_sample:
        fcgr = orig_sample['fcgr']
        print("Orig fcgr keys:", list(fcgr.keys()))
        k4_orig = fcgr.get('4')
        print(f"Orig k=4 shape: {k4_orig.shape if k4_orig is not None else 'None'}")
        print(f"Orig k=4 sum: {k4_orig.sum():.4f}" if k4_orig is not None else "")
    
    print("Mimic keys:", list(mimic_sample.keys()) if isinstance(mimic_sample, dict) else "Not dict")
    
    if isinstance(mimic_sample, dict) and 'fcgr' in mimic_sample:
        fcgr_m = mimic_sample['fcgr']
        k4_mimic = fcgr_m.get('4')
        print(f"Mimic k=4 shape: {k4_mimic.shape if k4_mimic is not None else 'None'}")
    
    print("\n✅ Data structure ready for pairing!")
else:
    print("❌ No common sequences found!")
