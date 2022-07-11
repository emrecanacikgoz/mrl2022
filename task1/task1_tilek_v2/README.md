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
- [x] **Task1 (Inflection):** Implement and train the Neural Transducer model.
- [ ] **Task1 (Inflection):** Implement and train an additional LSTM baseline mode.
- [ ] **Task1 (Inflection):** Experiment with Neural Transducer model's hyperparameters.
- [ ] **Task1 (Inflection):** Implement pretraining for baseline models.


## Environment Set-up
Please set your environment from scratch and document it, i.e.:
```
conda create --name shared_task1_v1 py=3.7
pip install torch
pip install numpy
pip install matplotlib
pip install tqdm
```


## Code Map
```
.
├── checkpoints
├── EXPERIMENTS
├── data
├── evaluation
├── transformer
└── README.md
```

# Model
The transformer-based Neural Transducer (Applying the Transformer to Character-level Transduction) model was trained.
The training instructions provided in author's repo were followed [(https://github.com/shijie-wu/neural-transducer)]
Training scripts are available under the transformer directory.

# Data
The training and validations sets were modified to fit the format used for the SIGMORPHON 2017 shared task (https://sigmorphon.github.io/sharedtasks/2017/task/), subtask. The difference is only in the order of data columns.

# Evaluation results
|                 | TUR   | ENG   | DEU   | FRA   | RUS   | HEB   | AVG   |
|-----------------|-------|-------|-------|-------|-------|-------|-------|
| Accuracy        | 0.969 | 0.946 | 0.792 | 0.935 | 0.865 | 0.921 | 0.905 |
| Edit distance   | 0.092 | 0.158 | 0.843 | 0.268 | 1.72  | 0.922 | 0.667 |
| Word-level acc. | 0.984 | 0.985 | 0.941 | 0.980 | 0.936 | 0.964 | 0.965 |
