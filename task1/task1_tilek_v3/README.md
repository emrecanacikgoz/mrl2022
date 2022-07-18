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
- [x] **Task1 (Inflection):** Implement and train an additional LSTM baseline mode.
- [ ] **Task1 (Inflection):** Experiment with Neural Transducer model's hyperparameters.
- [x] **Task1 (Inflection):** Collect and preprocess Kyrgyz and Kazakh corpora.
- [ ] **Task1 (Inflection):** Train GPT-2 for Kyrgyz and Kazakh.
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
├── data
├── evaluation
├── model
├── output-lstm
├── Evaluation_LSTM.ipynb
└── README.md
```

# Model
The model described in:
Shijie Wu, and Ryan Cotterell. [*Exact Hard Monotonic Attention for Character-Level Transduction*](https://arxiv.org/abs/1905.06319). ACL. 2019.
was implemented.
The training instructions provided in author's repo were followed [https://github.com/shijie-wu/neural-transducer/tree/master/example/hard-monotonic-attention]

# Data
The training and validations sets were modified to fit the format used for the SIGMORPHON 2017 shared task (https://sigmorphon.github.io/sharedtasks/2017/task/), subtask. The difference is only in the order of data columns.

# Evaluation results
|                 | TUR   | ENG   | DEU   | FRA   | RUS   | HEB   | AVG   |
|-----------------|-------|-------|-------|-------|-------|-------|-------|
| Accuracy        | 0.931 | 0.921 | 0.785 | 0.899 | 0.785 | 0.804 | 0.854 |
| Edit distance   | 0.176 | 0.308 | 1.024 | 0.517 | 2.682 | 1.81  | 1.086 |
| Word-level acc. | 0.964 | 0.985 | 0.929 | 0.961 | 0.914 | 0.918 | 0.945 |
