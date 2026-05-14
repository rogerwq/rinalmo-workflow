# RiNALMo Workflow — RNA Secondary Structure Prediction

Predicts RNA secondary structure from a single FASTA sequence using the [RiNALMo](https://github.com/lbcb-sci/RiNALMo) 650M-parameter language model with bpRNA fine-tuned weights.

**Paper:** [Nature Communications, Jul 2025](https://www.nature.com/articles/s41467-025-60872-5)

## Nodes

| # | Node | Description |
|---|------|-------------|
| 00 | Validate Input | Validates FASTA, normalises T→U, enforces 1022 nt limit |
| 01 | Predict Structure | Downloads weights, runs RiNALMo, outputs contact map + dot-bracket |
| 02 | Report | Arc diagram HTML report |

## Input

Place your RNA sequence as `input_files/sequence.fasta` before running:

```
>my_rna
GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCA
```

- Single sequence only (first record used if multiple)
- DNA sequences accepted (T → U conversion applied)
- Maximum length: 1022 nt

## Output

- `structure.dbn` — dot-bracket notation (FASTA-style: header, sequence, structure)
- `contact_map.npy` — binary N×N base-pair matrix
- `report.html` — arc diagram + statistics

## Weights

Fine-tuned bpRNA weights (~2.5 GB) are baked into the Docker image at `/weights/rinalmo_giga_ss_bprna_ft.pt`. No download at runtime.
