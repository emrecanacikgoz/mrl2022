# Task 1: Inflection
In this task the input is verbal lemma (the form given as a lexicon entry) and a specific set of inflectional features. The task requires generating the desired output clause manifesting the features. See Example:
```
Languages                           Input	                                Output
English       give IND;FUT;NOM(1,SG);ACC(3,SG,MASC);DAT(3,SG,FEM)       I will give him to her
German        geben IND;FUT;NOM(1,SG);ACC(3,SG,MASC);DAT(3,SG,FEM)      Ich werde ihn ihr geben
Turkish       vermek IND;FUT;NOM(1,SG);ACC(3,SG);DAT(3,SG)              Onu ona vereceğim
Hebrew        נתן IND;FUT;NOM(1,SG);ACC(3,SG,MASC);DAT(3,SG,FEM)        אתן אותו לה
```

## Results

| Language  |  Acc.(token)  |  EM.    |    Edit Distance |
|---------- |-------------: |------:  | ----------------:|
|deu        |     0.965     | 0.505   |        0.995     |
|eng        |     0.982     | 0.686   |        0.421     |
|fra        |     0.987     | 0.530   |        0.847     |
|heb        |     0.991     | 0.065   |        4.338     |
|rus        |     0.982     | 0.483   |        1.502     |
|tur        |     0.984     | 0.640   |        0.731     |
|**Average**|   **0.981**   |**0.484**|      **1.472**   |

## To Do's for Task1
- [ ] **Task1 (Emre Can):** Find the issue in version-1.
- [ ] **Task1 (Emre Can):** Implement/modify Transducer.

## Code Map
```
.
├── EXPERIMENTS
├── inf
│   ├── deu.dev - deu.trn - deu_covered.dev
│   ├── eng.dev - eng.trn - eng_covered.dev
│   ├── fra.dev - fra.trn - fra_covered.dev
│   ├── heb.dev - heb.trn - heb_covered.dev
│   ├── heb_unvoc.dev - heb_unvoc.trn - heb_unvoc_covered.dev
│   ├── rus.dev - rus.trn - rus_covered.dev
│   └── tur.dev - tur.trn - tur_covered.dev
├── model
│   ├── decoder.py
│   ├── encoder.py
│   ├── layers.py
│   ├── model.py
│   ├── multihead_attention.py
│   └── sublayers.py
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
## Task1 (Inflection) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t1_ver1.png)
