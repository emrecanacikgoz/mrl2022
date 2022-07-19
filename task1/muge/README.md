# Task 1: Inflection
In this task the input is verbal lemma (the form given as a lexicon entry) and a specific set of inflectional features. The task requires generating the desired output clause manifesting the features. 

## To Do's for Task1
- [ ] **Task1 (Muge):** Train baseline prompts for Turkish.
- [X] **Task1 (Muge):** Train baseline rompts for English.
- [X] **Task1 (Muge):** Train baseline prompts for German.
- [ ] **Task1 (Muge):** Train baseline prompts for Russian.
- [X] **Task1 (Muge):** Train baseline prompts for French.
- [ ] **Task1 (Muge):** Train baseline prompts for Hebrew.

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
