# Competition
KUIS-AI Shared Task Repository, see competition [repo](https://sigtyp.github.io/st2022-mrl.html) and [website](https://sigtyp.github.io/st2022-mrl.html).

## To Do's
- [x] **Task1 (Inflection):** Implement Encoder-Decoder (Vaswani) model.
- [x] **Task2 (Reinflection):** Implement Encoder-Decoder (Vaswani) model with [SEP] token as AB[SEP]C.
- [x] **Task3 (Analysis):** Implement Encoder-Decoder (Vaswani) model-ver1.
- [x] **Task3 (Analysis):** Implement Morse-based-ver2.
- [x] **Task3 (Analysis):** Train Encoder-Decoder (Vaswani) model ver1. <br/>
`Epoch: 500/500 |  avg_test_loss: 2.1593554 | perplexity: 8.6655498 |  test_accuracy: 76.10%` 
- [x] **Task3 (Analysis):** Check Positional Encodings bug (there was a bug, fixed now).
- [ ] **Task1 (Inflection):** Train Encoder-Decoder (Vaswani) model.
- [ ] **Task2 (Reinflection):** Train Encoder-Decoder (Vaswani) model.
- [ ] **Task3 (Analysis):** Fix dimension problem during cross-attention in Morse-based-ver2.
- [ ] Check Gözde Hoca's papers and implement: a pointer network solution and a monotonic hard attention based solution


## Repository Map
```
.
├── figures
├── task1
│   ├── EXPERIMENTS
│   ├── inf
│   ├── model
│   ├── dataloader
│   ├── README.md
│   ├── main.py # run this
│   ├── test.py
│   ├── training.py
│   └── utils.py
├── task2 
│   ├── EXPERIMENTS
│   ├── model
│   ├── reinf
│   ├── dataloader
│   ├── README.md
│   ├── main.py # run this
│   ├── test.py
│   ├── training.py
│   └── utils.py
├── task3
│   ├── EXPERIMENTS
│   ├── analysis
│   ├── model
│   ├── dataloader
│   ├── README.md
│   ├── main.py # run this
│   ├── test.py
│   ├── training.py
│   └── utils.py
└── README.md
```

Check "EXPERIMENTS" folder for the results.<br/>
**Please, do not forget to edit experiment name before you run the code.**

# Models
## Task1 (Inflection) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t1_ver1.png)

## Task2 (Re-Inflection) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t2_ver1.png)

## Task3 (Analysis) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t3_ver1.png)

## Task3 (Analysis) ver2
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t3_ver2.png)

# Suggested Research Materials:
Please add/update this section whenever you find something useful.
1. [Morphology Without Borders: Clause-Level Morphological Annotation](https://arxiv.org/pdf/2202.12832.pdf)
2. [Morphological Inflection Generation with Hard Monotonic Attention](https://aclanthology.org/P17-1183.pdf)
3. [On Biasing Transformer Attention Towards Monotonicity](https://arxiv.org/pdf/2104.03945.pdf)
4. [Monotonic Multihead Attention](https://arxiv.org/pdf/1909.12406.pdf)

