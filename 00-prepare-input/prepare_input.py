import os
import shutil

IN_DIR       = "inputs"
OUT_DIR      = "outputs"
EXAMPLE_PATH = "example.fasta"
os.makedirs(OUT_DIR, exist_ok=True)

fasta_path = os.path.join(IN_DIR, "sequence.fasta")
out_path   = os.path.join(OUT_DIR, "sequence.fasta")

if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0:
    print(f"Using user-supplied FASTA: {fasta_path}")
    shutil.copy(fasta_path, out_path)
else:
    print("No input_files/sequence.fasta found — falling back to bundled example sequence (tRNA-Phe-human).")
    shutil.copy(EXAMPLE_PATH, out_path)

print(f"Input prepared at {out_path}")
