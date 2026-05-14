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

# --- reconstruct model from Lightning checkpoint ---
print("Loading model...")
ckpt = torch.load(CHECKPOINT, map_location="cpu")
hparams       = ckpt.get("hyper_parameters", {})
lm_cfg_name   = hparams.get("lm_config", "giga")
num_blocks    = hparams.get("num_resnet_blocks", 2)
threshold     = float(ckpt.get("callbacks", {}).get("threshold", 0.5))

config    = model_config(lm_cfg_name)
embed_dim = config["model"]["transformer"]["embed_dim"]
alphabet  = Alphabet(**config["alphabet"])

lm        = RiNALMo(config)
pred_head = SecStructPredictionHead(embed_dim, num_blocks=num_blocks)

state = ckpt["state_dict"]
lm.load_state_dict({k[3:]: v for k, v in state.items() if k.startswith("lm.")})
pred_head.load_state_dict({k[10:]: v for k, v in state.items() if k.startswith("pred_head.")})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
lm.to(device).eval()
pred_head.to(device).eval()

# --- inference ---
tokens = torch.tensor(alphabet.batch_tokenize([seq]), dtype=torch.int64, device=device)
with torch.no_grad():
    rep    = lm(tokens)["representation"]          # B x (L+2) x E
    logits = pred_head(rep[..., 1:-1, :])          # B x L x L
    probs  = torch.sigmoid(logits).squeeze(0)      # L x L
    probs  = probs.cpu().numpy()

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
