
import sys
import pickle
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.preprocessing.fcgr import compute_multi_k_fcgr

def parse_fasta(fasta_path: Path):
    records = []
    with open(fasta_path) as f:
        header = None
        seq = []

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq)))
                header = line
                seq = []
            else:
                seq.append(line)

        if header is not None:
            records.append((header, "".join(seq)))

    return records


def extract_abundance(header: str) -> int:
    for part in header.split(";"):
        if part.startswith("size="):
            return int(part.replace("size=", ""))
    return 1


def main():
    fasta_path = Path("data\processed\clean.fasta (Non-chimera_final)")
    output_path = Path("data/interim/fcgr.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fcgr_data: Dict[str, Dict] = {}

    records = parse_fasta(fasta_path)

    for header, sequence in records:
        seq_id = header.split()[0][1:]  # remove '>'
        abundance = extract_abundance(header)

        fcgr = compute_multi_k_fcgr(sequence)

        fcgr_data[seq_id] = {
            "abundance": abundance,
            "fcgr": fcgr
        }

    with open(output_path, "wb") as f:
        pickle.dump(fcgr_data, f)

    print(f"Saved FCGRs for {len(fcgr_data)} sequences to {output_path}")


if __name__ == "__main__":
    main()
