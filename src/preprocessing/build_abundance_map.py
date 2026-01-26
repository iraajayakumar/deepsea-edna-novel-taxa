from pathlib import Path
import json
import re

def build_abundance_map(derep_fasta_path, output_path):
    abundance_map = {}

    with open(derep_fasta_path, "r") as f:
        current_size = None

        for line in f:
            line = line.strip()

            if line.startswith(">"):
                # Extract size from header
                match = re.search(r"size=(\d+)", line)
                if match:
                    current_size = int(match.group(1))
                else:
                    current_size = 1  # fallback
            else:
                sequence = line.upper()
                abundance_map[sequence] = current_size

    with open(output_path, "w") as out:
        json.dump(abundance_map, out, indent=2)

    print(f"Saved abundance map with {len(abundance_map)} sequences")


if __name__ == "__main__":
    derep_fasta = "data/processed/dereplication.fasta"
    output_json = "data/processed/abundance_map.json"

    build_abundance_map(derep_fasta, output_json)
