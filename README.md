# Competition
KUIS-AI Shared Task Repository, see competition [repo](https://sigtyp.github.io/st2022-mrl.html) and [website](https://sigtyp.github.io/st2022-mrl.html).

### Kind Rules of Our Repo
Please try to obey the "rules" of our repo, to make everything as convenient as possible. These are to make everything more easier for everyone. :)
1. Set your environment from scratch. If you installed new packages, please document each of the packges with their versions.
2. When you are implementing a new model, for a specific task, create a subfolder as `taskX_name_vY`, where X and Y is the task and model versions respectively. For example:
```
.
task1
    ├── task1_emrecan_v1
    └── task1_tilek_v1
```
First subfolders are already created for you.
3. Update your To Do's in your **subfolder**, frequently. By this, others can follow your progress easily. I will add your To Do's inside the top-level README, one by one.
4. Try to add helper comments to your code. This way, others can read your code easily.
At the end, lets beat this competition!

## Calendar of The Competition
|Week|Mon|Tue|Wed|Thu|Fri|Sat|Sun|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|28|27|28|29|30|1|2|3|
|29|4|5|6|7|8|9|10|
|30|11|12|13|14|15|16|17|
|31|18|19|20|21|22|23|24|
|32|25|26|27|28|29|30|31|
|33|1|2|3|4|5|6|**7***|
|34|8|9|10|11|12|13|**14***|
|35|15|16|17|18|19|20|21|
|36|***22*** :dart:|23|24|**25**|26|27|28|
|37|29|30|31|1|2|3|4|
```
===> August 7, 2022: Release of testing data (including surprise languages)
===> August 8, 2022: Get a baseline score for each task with each language (6 Language x 3 task = 18 results).
===> August 14, 2022: Deadline to release external data and resources used in systems
===> August 22, 2022: Deadline for submission of systems
===> August 25, 2022: Release of rankings and results
===> September 15, 2022: Deadline for submitting system description papers
===> October 10, 2022: Paper notifications
===> November 9, 2022: Camera-ready papers and posters due
===> December 8, 2022: Workshop

```

## To Do's
- [x] **Task1 (Emre Can):** Implement Encoder-Decoder (Vaswani) model.
- [x] **Task2 (Emre Can):** Implement Encoder-Decoder (Vaswani) model with [SEP] token as AB[SEP]C.
- [x] **Task3 (Emre Can):** Implement Encoder-Decoder (Vaswani) model-ver1.
- [x] **Task3 (Emre Can):** Implement Morse-based-ver2.
- [x] **Task3 (Emre Can):** Train Encoder-Decoder (Vaswani) model ver1. <br/>
`Epoch: 500/500 |  avg_test_loss: 2.1593554 | perplexity: 8.6655498 |  test_accuracy: 76.10%` 
- [x] **Task3 (Emre Can):** Check Positional Encodings bug (there was a bug, fixed now).
- [ ] **Task1 (Emre Can):** Train Encoder-Decoder (Vaswani) model.
- [ ] **Task2 (Emre Can):** Train Encoder-Decoder (Vaswani) model.
- [ ] **Task3 (Emre Can):** Fix dimension problem during cross-attention in Morse-based-ver2.
- [ ] Check Gözde Hoca's papers and implement: a pointer network solution and a monotonic hard attention based solution

## Environment Set-up
Please do the followings to run the baselines successfully, i.e. "taskx_emrecan_v1":
```
conda create --name taskx_emrecan_v1 python=3.7
source activate taskx_emrecan_v1

pip install torch==1.12.0
pip install numpy==1.21.6
pip install matplotlib==3.5.2
```
You can check your installed packages in your environment by doing `conda list`. [Here](https://github.com/ecacikgoz97/competation/blob/main/requirments/package_list.txt) is mine. Please open an issue for any version conflicts.

## Repository Map
```
.
├── figures
├── requirments
│   └── package_list.txt
├── task1
│   ├── task1_emrecan_v1
│   └── task1_tilek_v1
├── task2 
│   ├── task2_emrecan_v1
│   └── task2_muge_v1
├── task3
│   └── task3_emrecan_v1
└── README.md
```

# Models
## Task1 (Emre Can) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t1_ver1.png)

## Task2 (Emre Can) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t2_ver1.png)

## Task3 (Emre Can) ver1
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t3_ver1.png)

## Task3 (Emre Can) ver2
![alt text](https://github.com/ecacikgoz97/competation/blob/main/figures/t3_ver2.png)

# Suggested Research Materials:
Please add/update this section whenever you find something useful.
1. [Morphology Without Borders: Clause-Level Morphological Annotation](https://arxiv.org/pdf/2202.12832.pdf)
2. [Morphological Inflection Generation with Hard Monotonic Attention](https://aclanthology.org/P17-1183.pdf)
3. [On Biasing Transformer Attention Towards Monotonicity](https://arxiv.org/pdf/2104.03945.pdf)
4. [Monotonic Multihead Attention](https://arxiv.org/pdf/1909.12406.pdf)

