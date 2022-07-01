# Task 2: Reinflection
In this task the input is an inflected clause, accompanied by its features, and a new set of features representing the desired form. The task is to generate the desired output that will represent the desired features. See Example:

## To Do's for Task2
- [x] **Task2 (Reinflection):** Implement Encoder-Decoder (Vaswani) model with [SEP] token as AB[SEP]C.
- [ ] **Task2 (Reinflection):** Train reinflection.

## Environment Set-up
Please set your environment from scratch and document it, i.e.:
```
conda create --name shared_task2_v1 py=3.7
pip install torch
pip install numpy
pip install matplotlib
```

## Code Map
```
.
├── EXPERIMENTS
├── reinf
│   ├── deu.dev - deu.trn - deu_covered.dev
│   ├── eng.dev - eng.trn - eng_covered.dev
│   ├── fra.dev - fra.trn - fra_covered.dev
│   ├── heb.dev - heb.trn - heb_covered.dev
│   ├── heb_unvoc.dev - heb_unvoc.trn - heb_unvoc_covered.dev
│   ├── rus.dev - rus.trn - rus_covered.dev
│   └── tur.dev - tur.trn - tur_covered.dev
├── README.md
└── README.md
```

# Model
Please add/draw your proposed model.