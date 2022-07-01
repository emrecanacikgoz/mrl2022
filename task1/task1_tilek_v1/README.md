# Task 1: Inflection
In this task the input is verbal lemma (the form given as a lexicon entry) and a specific set of inflectional features. The task requires generating the desired output clause manifesting the features. See Example:
```
Languages                           Input	                                Output
English       give IND;FUT;NOM(1,SG);ACC(3,SG,MASC);DAT(3,SG,FEM)       I will give him to her
German        geben IND;FUT;NOM(1,SG);ACC(3,SG,MASC);DAT(3,SG,FEM)      Ich werde ihn ihr geben
Turkish       vermek IND;FUT;NOM(1,SG);ACC(3,SG);DAT(3,SG)              Onu ona vereceğim
Hebrew        נתן IND;FUT;NOM(1,SG);ACC(3,SG,MASC);DAT(3,SG,FEM)        אתן אותו לה
```

## To Do's for Task1
- [x] **Task1 (Inflection):** Implement Encoder-Decoder (Vaswani) model.
- [ ] **Task1 (Inflection):** Train Encoder-Decoder (Vaswani) model.

## Environment Set-up
Please set your environment from scratch and document it, i.e.:
```
conda create --name shared_task1_v1 py=3.7
pip install torch
pip install numpy
pip install matplotlib
```


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
└── README.md
```

# Model
# Model
Please add/draw your proposed model.
