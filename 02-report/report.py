import os
import math
import numpy as np
from Bio import SeqIO

IN_DIR  = "inputs"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# --- load inputs ---
record  = next(SeqIO.parse(os.path.join(IN_DIR, "sequence.fasta"), "fasta"))
seq     = str(record.seq).upper()
seq_id  = record.id

with open(os.path.join(IN_DIR, "structure.dbn")) as f:
    lines = [l.strip() for l in f if l.strip()]
dbn = lines[2] if len(lines) >= 3 else lines[-1]

contact_map = np.load(os.path.join(IN_DIR, "contact_map.npy"))

# --- parse base pairs from dot-bracket ---
pairs = {}
stack = []
for i, c in enumerate(dbn):
    if c == "(":
        stack.append(i)
    elif c == ")":
        j = stack.pop()
        pairs[j] = i
        pairs[i] = j

n_pairs  = len(pairs) // 2
n_paired = len(pairs)

# --- build arc diagram SVG ---
W      = max(900, len(seq) * 14 + 40)
CX     = W / 2
BASE_Y = 220
R_NUC  = 8
COLORS = {"A": "#4CAF50", "U": "#E53935", "G": "#1E88E5", "C": "#FB8C00"}

# nucleotide positions (evenly spaced)
spacing = min(14, (W - 40) / max(len(seq), 1))
x_pos   = [20 + i * spacing + spacing / 2 for i in range(len(seq))]

# arcs (draw above the sequence line)
arcs_svg = ""
for i, j in pairs.items():
    if i >= j:
        continue
    cx   = (x_pos[i] + x_pos[j]) / 2
    rx   = (x_pos[j] - x_pos[i]) / 2
    ry   = rx * 0.55
    arcs_svg += (
        f'<ellipse cx="{cx:.1f}" cy="{BASE_Y}" rx="{rx:.1f}" ry="{ry:.1f}" '
        f'fill="none" stroke="#90CAF9" stroke-width="1.2" '
        f'clip-path="url(#above)"/>\n'
    )

# nucleotide circles
nucs_svg = ""
for i, nuc in enumerate(seq):
    color = COLORS.get(nuc, "#9E9E9E")
    label = "paired" if i in pairs else "unpaired"
    nucs_svg += (
        f'<circle cx="{x_pos[i]:.1f}" cy="{BASE_Y}" r="{R_NUC}" '
        f'fill="{color}" stroke="white" stroke-width="1" title="{nuc}{i+1} ({label})"/>\n'
        f'<text x="{x_pos[i]:.1f}" y="{BASE_Y+4:.1f}" text-anchor="middle" '
        f'font-size="9" font-family="monospace" fill="white">{nuc}</text>\n'
    )

# position tick marks every 10 nt
ticks_svg = ""
for i in range(0, len(seq), 10):
    ticks_svg += (
        f'<line x1="{x_pos[i]:.1f}" y1="{BASE_Y+R_NUC+2:.1f}" '
        f'x2="{x_pos[i]:.1f}" y2="{BASE_Y+R_NUC+8:.1f}" stroke="#999" stroke-width="1"/>\n'
        f'<text x="{x_pos[i]:.1f}" y="{BASE_Y+R_NUC+18:.1f}" text-anchor="middle" '
        f'font-size="9" fill="#666">{i+1}</text>\n'
    )

SVG_H = BASE_Y + 50
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{SVG_H}" viewBox="0 0 {W} {SVG_H}">
  <defs>
    <clipPath id="above">
      <rect x="0" y="0" width="{W}" height="{BASE_Y}"/>
    </clipPath>
  </defs>
  <rect width="{W}" height="{SVG_H}" fill="#fafafa" rx="6"/>
  <!-- backbone line -->
  <line x1="{x_pos[0]:.1f}" y1="{BASE_Y}" x2="{x_pos[-1]:.1f}" y2="{BASE_Y}"
        stroke="#ccc" stroke-width="1.5"/>
  {arcs_svg}
  {nucs_svg}
  {ticks_svg}
</svg>"""

# --- composition table ---
comp_rows = "".join(
    f'<tr><td>{nuc}</td><td>{seq.count(nuc)}</td>'
    f'<td>{100*seq.count(nuc)/len(seq):.1f}%</td></tr>'
    for nuc in ("A", "U", "G", "C")
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RiNALMo — {seq_id}</title>
  <style>
    body {{ font-family: sans-serif; margin: 40px; color: #222; }}
    h1   {{ color: #1565C0; }}
    h2   {{ color: #333; margin-top: 32px; }}
    .stats-grid {{ display: flex; gap: 24px; flex-wrap: wrap; margin: 16px 0; }}
    .stat {{ background: #f0f4ff; border-radius: 8px; padding: 16px 24px; min-width: 120px; }}
    .stat .val {{ font-size: 2em; font-weight: bold; color: #1565C0; }}
    .stat .lbl {{ font-size: 0.85em; color: #555; }}
    table {{ border-collapse: collapse; }}
    th {{ background: #1565C0; color: white; padding: 8px 16px; }}
    td {{ padding: 6px 16px; border-bottom: 1px solid #eee; text-align: center; }}
    .dbn {{ font-family: monospace; font-size: 13px; word-break: break-all;
             background: #f5f5f5; padding: 12px; border-radius: 4px; }}
    .legend span {{ display: inline-block; width: 12px; height: 12px;
                    border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
    .scroll {{ overflow-x: auto; }}
    footer {{ margin-top: 48px; font-size: 0.8em; color: #999; }}
  </style>
</head>
<body>
  <h1>RiNALMo — RNA Secondary Structure Prediction</h1>
  <p>Sequence: <strong>{seq_id}</strong> &nbsp;|&nbsp; Model: <em>RiNALMo Giga (650M), bpRNA fine-tuned</em></p>

  <div class="stats-grid">
    <div class="stat"><div class="val">{len(seq)}</div><div class="lbl">Length (nt)</div></div>
    <div class="stat"><div class="val">{n_pairs}</div><div class="lbl">Base pairs</div></div>
    <div class="stat"><div class="val">{100*n_paired/len(seq):.0f}%</div><div class="lbl">Paired</div></div>
  </div>

  <h2>Arc Diagram</h2>
  <p class="legend">
    <span style="background:#4CAF50"></span>A &nbsp;
    <span style="background:#E53935"></span>U &nbsp;
    <span style="background:#1E88E5"></span>G &nbsp;
    <span style="background:#FB8C00"></span>C &nbsp;
    &nbsp; Arcs = predicted base pairs
  </p>
  <div class="scroll">{svg}</div>

  <h2>Dot-Bracket Notation</h2>
  <div class="dbn">{seq}<br>{dbn}</div>

  <h2>Nucleotide Composition</h2>
  <table>
    <thead><tr><th>Nucleotide</th><th>Count</th><th>Fraction</th></tr></thead>
    <tbody>{comp_rows}</tbody>
  </table>

  <footer>
    Generated by RiNALMo workflow &nbsp;|&nbsp;
    <a href="https://github.com/lbcb-sci/RiNALMo">github.com/lbcb-sci/RiNALMo</a> &nbsp;|&nbsp;
    <a href="https://www.nature.com/articles/s41467-025-60872-5">Nature Communications 2025</a>
  </footer>
</body>
</html>"""

out_path = os.path.join(OUT_DIR, "report.html")
with open(out_path, "w") as f:
    f.write(html)
print(f"Report written to {out_path}")
