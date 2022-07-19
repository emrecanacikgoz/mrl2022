# Task 2: Reinflection
In this task the input is an inflected clause, accompanied by its features, and a new set of features representing the desired form. The task is to generate the desired output that will represent the desired features.

# Results

| Language  |  Acc.(token)  |  EM.    |    Edit Distance |
|---------- |-------------: |------:  | ----------------:|
|deu        |     0.901     | 0.097   |        3.012     |
|eng        |     0.931     | 0.137   |        1.804     |
|fra        |     0.923     | 0.208   |        2.264     |
|heb        |     0.935     | 0.202   |        1.871     |
|rus        |     0.958     | 0.284   |        2.386     |
|tur        |     0.931     | 0.217   |        1.993     |
|**Average**|   **0.929**   |**0.191**|      **2.222**   |




## To Do's for Task1
- [ ] **Task2 (Emre Can):** Find issue in version-1
- [ ] **Task2 (Emre Can):** Implement Transducer

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
