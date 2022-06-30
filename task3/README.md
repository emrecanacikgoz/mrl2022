# Task 3: Analysis
This task is the opposite of task 1, where a system is required to analyze given clauses and generate the lemma and features underlying them. See Example:
```
Languages	Input	Output
English	I will give him to her	give IND;FUT;NOM(1,SG);ACC(3,SG,MASC);DAT(3,SG,FEM)
German	Ich werde ihn ihr geben	geben IND;FUT;NOM(1,SG);ACC(3,SG,MASC);DAT(3,SG,FEM)
Turkish	Onu ona vereceğim	vermek IND;FUT;NOM(1,SG);ACC(3,SG);DAT(3,SG)
Hebrew	אתן אותו לה	נתן IND;FUT;NOM(1,SG);ACC(3,SG,MASC);DAT(3,SG,FEM)
```

## To Do's for Task3
- [x] **Task3 (Analysis):** Implement Encoder-Decoder (Vaswani) model-ver1.
- [x] **Task3 (Analysis):** Implement Morse-based-ver2.
- [x] **Task3 (Analysis):** Train Encoder-Decoder (Vaswani) model ver1. <br/>
`Epoch: 500/500 |  avg_test_loss: 2.1593554 | perplexity: 8.6655498 |  test_accuracy: 76.10%` 
- [x] **Task3 (Analysis):** Check Positional Encodings bug (there was a bug, fixed now).
- [ ] **Task3 (Analysis):** Fix dimension problem during cross-attention in Morse-based-ver2.

## Code Map
```
.
├── inf
│   ├── deu.trn - deu.dev - deu_covered.dev
│   ├── eng.trn - eng.dev - eng_covered.dev
│   ├── fra.trn - fra.dev - fra_covered.dev
│   ├── heb.trn - heb.dev - heb_covered.dev
│   ├── heb_unvoc.trn - heb_unvoc.dev - heb_unvoc_covered.dev
│   ├── rus.trn - rus.dev - rus_covered.dev
│   └── tur.trn - tur.dev - tur_covered.dev
├── EXPERIMENTS
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
## Task3 (Analysis) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t3_ver1.png)

## Task3 (Analysis) ver2
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t3_ver2.png)


