# Task 2: Reinflection
In this task the input is an inflected clause, accompanied by its features, and a new set of features representing the desired form. The task is to generate the desired output that will represent the desired features. See Example:


## To Do's for Task1
- [x] **Task1-2-3 (Emre Can):** Check Positional Encodings bug (there was a bug, fixed now).
- [ ] **Task2 (Emre Can):** Train version-1 (Encoder-Decoder Vaswani) for Turkish.
- [ ] **Task2 (Emre Can):** Train version-1 (Encoder-Decoder Vaswani) for English.
- [ ] **Task2 (Emre Can):** Train version-1 (Encoder-Decoder Vaswani) for Germain.
- [ ] **Task2 (Emre Can):** Train version-1 (Encoder-Decoder Vaswani) for Russian.
- [ ] **Task2 (Emre Can):** Train version-1 (Encoder-Decoder Vaswani) for French.
- [ ] **Task2 (Emre Can):** Train version-1 (Encoder-Decoder Vaswani) for Hebrew.

## Code Map
```
.
├── EXPERIMENTS
├── model
│   ├── decoder.py
│   ├── encoder.py
│   ├── layers.py
│   ├── model.py
│   ├── multihead_attention.py
│   └── sublayers.py
├── reinf
│   ├── deu.dev - deu.trn - deu_covered.dev
│   ├── eng.dev - eng.trn - eng_covered.dev
│   ├── fra.dev - fra.trn - fra_covered.dev
│   ├── heb.dev - heb.trn - heb_covered.dev
│   ├── heb_unvoc.dev - heb_unvoc.trn - heb_unvoc_covered.dev
│   ├── rus.dev - rus.trn - rus_covered.dev
│   └── tur.dev - tur.trn - tur_covered.dev
├── dataloader.py
├── main.py # run this
├── README.md
├── test.py
├── training.py
└── utils.py
```

Check "EXPERIMENTS" folder for the results.<br/>
**Please, do not forget to edit experiment name before you run the code.**

# Model
## Task2 (Reinflection) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t2_ver1.png)
