# Competition
KUIS-AI Shared Task Repository, see competition [repo](https://sigtyp.github.io/st2022-mrl.html) and [website](https://sigtyp.github.io/st2022-mrl.html).

## To Do's
- [x] Task1 (Inflection): Implement Encoder-Decoder (Vaswani) model.
- [x] Task3 (Analysis): Implement Encoder-Decoder (Vaswani) model-ver1.
- [x] Task3 (Analysis): Implement Morse-based-ver2.
- [x] Task3 (Analysis): Train Encoder-Decoder (Vaswani) model ver1. <br/>
`Epoch: 499/500 |  avg_test_loss: 2.1593554 | perplexity: 8.6655498 |  test_accuracy: 76.10%` 
- [ ] Task1 (Inflection): Train Encoder-Decoder (Vaswani) model.
- [ ] Task2 (Reinflection): Implement Encoder-Decoder (Vaswani) model with [SEP] token as AB[SEP]C.
- [ ] Task3 (Analysis): Fix dimension problem during cross-attention in Morse-based-ver2.
- [ ] Task3 (Analysis): Fix overfitting problem in Morse-based-ver2; check also positional encodings.
- [ ] Check Gözde Hoca's papers and implement: a pointer network solution and a monotonic hard attention based solution


## Repository Map
```
.
├── task3
│   ├── analysis
│   ├── figures
│   ├── model
│   ├── dataloader
│   ├── main.py # run this
│   ├── test.py
│   ├── training.py
│   └── utils.py
└── README.md
```

# Models
### Task1 (Inflection) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/task3/figures/t1_ver1.png)

### Task2 (Re-Inflection) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/task3/figures/t2_ver1.png)

### Task3 (Analysis) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/task3/figures/t3_ver1.png)

### Task3 (Analysis) ver2
![alt text](https://github.com/ecacikgoz97/competation/blob/main/task3/figures/t3_ver2.png)

# Suggested Research Materials:
Please add/update this section whenever you find something useful.
1. [Morphological Inflection Generation with Hard Monotonic Attention](https://aclanthology.org/P17-1183.pdf)
2. [On Biasing Transformer Attention Towards Monotonicity](https://arxiv.org/pdf/2104.03945.pdf)
3. [Monotonic Multihead Attention](https://arxiv.org/pdf/1909.12406.pdf)

