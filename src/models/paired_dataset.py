import torch
from torch.utils.data import Dataset
import pickle
from pathlib import Path

class PairedFCGRDataset(Dataset):
    def __init__(self, orig_path, mimic_path):
        self.orig_path = Path(orig_path)
        self.mimic_path = Path(mimic_path)
        
        print(f"Loading originals from {self.orig_path}...")
        with open(self.orig_path, 'rb') as f:
            orig_data = pickle.load(f)
        self.orig_seq_ids = sorted(orig_data.keys())
        self.orig_data = {sid: orig_data[sid] for sid in self.orig_seq_ids}
        
        print(f"Loading mimics from {self.mimic_path}...")
        with open(self.mimic_path, 'rb') as f:
            self.mimic_list = pickle.load(f)
        
        n_pairs = min(len(self.orig_seq_ids), len(self.mimic_list))
        self.orig_seq_ids = self.orig_seq_ids[:n_pairs]
        print(f"✅ Paired {n_pairs} sequences")
    
    def __len__(self):
        return len(self.orig_seq_ids)
    
    def __getitem__(self, idx):
        seq_id = self.orig_seq_ids[idx]
        orig_entry = self.orig_data[seq_id]
        orig_fcgr = {str(k): torch.FloatTensor(orig_entry['fcgr'][k]).unsqueeze(0) 
                     for k in [4, 5, 6]}
        
        mimic_entry = self.mimic_list[idx]
        mimic_fcgr = {str(k): torch.FloatTensor(mimic_entry[k]).unsqueeze(0) 
                      for k in [4, 5, 6]}
        
        return orig_fcgr, mimic_fcgr
