import os
import sys
import shutil

import numpy as np
import torch
from Bio import SeqIO

sys.path.insert(0, "/opt/RiNALMo")
from rinalmo.model.model import RiNALMo
from rinalmo.model.downstream import SecStructPredictionHead
from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.utils.sec_struct import prob_mat_to_sec_struct

IN_DIR     = "inputs"
OUT_DIR    = "outputs"
CHECKPOINT = "/weights/rinalmo_giga_ss_bprna_ft.pt"
os.makedirs(OUT_DIR, exist_ok=True)

# --- load input sequence ---
fasta_path = os.path.join(IN_DIR, "sequence.fasta")
record = next(SeqIO.parse(fasta_path, "fasta"))
seq = str(record.seq).upper()
print(f"Sequence: {record.id} ({len(seq)} nt)")

# --- reconstruct model from checkpoint ---
# The saved file is a flat OrderedDict: keys lm.*, pred_head.*, threshold
print("Loading model...")
ckpt      = torch.load(CHECKPOINT, map_location="cpu")
threshold = float(ckpt.get("threshold", 0.5))

if not torch.cuda.is_available():
    print("ERROR: RiNALMo requires a CUDA GPU. No GPU detected.")
    sys.exit(1)
device = torch.device("cuda:0")
print(f"Running on: {device}")

config = model_config("giga")
embed_dim = config["model"]["transformer"]["embed_dim"]
alphabet  = Alphabet(**config["alphabet"])

lm        = RiNALMo(config)
pred_head = SecStructPredictionHead(embed_dim, num_blocks=2)

lm.load_state_dict({k[3:]: v for k, v in ckpt.items() if k.startswith("lm.")})
pred_head.load_state_dict({k[10:]: v for k, v in ckpt.items() if k.startswith("pred_head.")})

lm.to(device).eval()
pred_head.to(device).eval()

# --- inference ---
tokens = torch.tensor(alphabet.batch_tokenize([seq]), dtype=torch.int64, device=device)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    rep    = lm(tokens)["representation"]          # B x (L+2) x E
    logits = pred_head(rep[..., 1:-1, :])          # B x L x L
    probs  = torch.sigmoid(logits).squeeze(0).float().cpu().numpy()  # L x L, fp32

# symmetrize (should already be symmetric, but ensure)
probs = (probs + probs.T) / 2.0

# --- post-process to binary contact map and dot-bracket ---
contact_map = prob_mat_to_sec_struct(probs.copy(), seq, threshold=threshold)
np.save(os.path.join(OUT_DIR, "contact_map.npy"), contact_map)

pairs = {}
for i in range(len(seq)):
    for j in range(i + 1, len(seq)):
        if contact_map[i, j] == 1:
            pairs[i] = j
            pairs[j] = i

dbn = []
for i in range(len(seq)):
    if i in pairs:
        dbn.append("(" if pairs[i] > i else ")")
    else:
        dbn.append(".")
dbn_str = "".join(dbn)

dbn_path = os.path.join(OUT_DIR, "structure.dbn")
with open(dbn_path, "w") as f:
    f.write(f">{record.id}\n{seq}\n{dbn_str}\n")

shutil.copy(fasta_path, os.path.join(OUT_DIR, "sequence.fasta"))

paired = sum(1 for c in dbn_str if c != ".") // 2
print(f"Base pairs  : {paired} / {len(seq)} positions paired")
print(f"Dot-bracket : {dbn_str}")
print("Done.")
