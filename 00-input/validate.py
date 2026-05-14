import os
import sys
import shutil
from Bio import SeqIO

IN_DIR  = "inputs"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

fasta_path = os.path.join(IN_DIR, "sequence.fasta")
if not os.path.exists(fasta_path):
    print("ERROR: input_files/sequence.fasta not found. Place your RNA FASTA file there before running.")
    sys.exit(1)

records = list(SeqIO.parse(fasta_path, "fasta"))
if not records:
    print("ERROR: sequence.fasta is empty or not valid FASTA.")
    sys.exit(1)
if len(records) > 1:
    print(f"WARNING: {len(records)} sequences found; only the first will be used.")

seq = str(records[0].seq).upper().replace("T", "U")
if len(seq) > 1022:
    print(f"WARNING: sequence length {len(seq)} exceeds RiNALMo max (1022). It will be truncated.")
    seq = seq[:1022]

print(f"Sequence ID : {records[0].id}")
print(f"Length      : {len(seq)} nt")
print(f"Composition : A={seq.count('A')} U={seq.count('U')} G={seq.count('G')} C={seq.count('C')}")

out_path = os.path.join(OUT_DIR, "sequence.fasta")
with open(out_path, "w") as f:
    f.write(f">{records[0].id}\n{seq}\n")

print(f"Validated sequence written to {out_path}")
