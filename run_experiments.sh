#!/usr/bin/env bash
# Item 3: testa diferentes parâmetros (lr x rank LoRA) e avalia cada um (item 4).
python src/make_dataset.py
for lr in 1e-4 2e-4; do
  for r in 8 16; do
    run="outputs/lr${lr}_r${r}"
    python src/train.py --lr $lr --r $r --epochs 3 --out $run
    python src/evaluate.py --adapter $run
  done
done
