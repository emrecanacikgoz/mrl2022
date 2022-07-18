# Competition
KUIS-AI Shared Task Repository, see competition [repo](https://sigtyp.github.io/st2022-mrl.html) and [website](https://sigtyp.github.io/st2022-mrl.html).

### Kind Rules for our Repo
Please try to obey the "rules" of our repo, to make everything as convenient as possible. These are to make everything more easier for everyone. :)
1. Set your environment from scratch. If you installed new packages, please document each of the packges with their versions.
2. When you are implementing a new model, for a specific task, create a subfolder as `taskX_name_vY`, where X and Y is the task and model versions respectively. For example: `./task1/task1_emrecan_v1` and `./task1/task1_tilek_v1`. Initial subfolders are already created for you.
3. Update your To Do's in your **subfolder**, frequently. By this, others can follow your progress easily. I will add your To Do's inside the top-level README, one by one.
4. Try to add helper comments to your code. This way, others can read your code easily.
5. Add your research materials or your suggested materials to **Suggested Research Materials** section in this README file.<br/>

At the end, lets beat this competition!

## Calendar of the Competition
**Days left:** 36 <br/>
|Week|Mon|Tue|Wed|Thu|Fri|Sat|Sun|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|30|11|12|13|14|15|16|17|
|31|18|19|20|21|22|23|24|
|32|25|26|27|28|29|30|31|
|33|1|2|3|4|5|6|**7***|
|34|8|9|10|11|12|13|**14***|
|35|15|16|17|18|19|20|21|
|36|**22*** :dart:|23|24|**25***|26|27|28|
|37|29|30|31|1|2|3|4|

**Important Deadlines:**
```
July 8, 2022: Get a baseline score for each task with each language (6 Languages x 3 tasks = 18 results).
===> August 7, 2022: Release of testing data (including surprise languages)
===> August 14, 2022: Deadline to release external data and resources used in systems
===> August 22, 2022: Deadline for submission of systems
===> August 25, 2022: Release of rankings and results
===> September 15, 2022: Deadline for submitting system description papers
===> October 10, 2022: Paper notifications
===> November 9, 2022: Camera-ready papers and posters due
===> December 8, 2022: Workshop
```


## To Do's
- [x] **Task1 (Emre Can):** Fix/Debug issue, if any (Shuffle Training set).
- [x] **Task1 (Emre Can):** Train new Model-ver2 (3xEncoder+1xDecoder).
- [ ] **Task1 (Emre Can):** Repeat Muge's prompts.
- [ ] **Task1 (Tilek):** Implement and train an additional LSTM baseline mode.
- [ ] **Task1 (Tilek):** Experiment with Neural Transducer model's hyperparameters.
- [ ] **Task1 (Tilek):** Implement pretraining for baseline models.
- [ ] **Task2 (Emre Can):** Fix/Debug issue, if any.
- [ ] **Task2 (Emre Can):** Repeat Muge's prompts.
- [ ] **Task2 (Muge):** Train different pretrained LMs for all languages, especially for Hebrew.
- [ ] **Task2 (Muge):** Literature review for different prompting techniques other than prefix tuning for all languages.
- [ ] **Task2 (Muge):** Train different prompting techniques other than prefix tuning for all languages.
- [ ] **Task2 (Muge):** Train prompts to Masked LMs (such as BERT), for all languages.
- [ ] **Task3 (Emre Can):** Fix/Debug issue, if any.
- [ ] **Task3 (Emre Can):** Train and get good results as baseline model.
- [ ] Check Gözde Hoca's papers and implement: a pointer network solution and a monotonic hard attention based solution.

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

# Suggested Research Materials:
Please add/update this section whenever you find something useful.
1. [Morphology Without Borders: Clause-Level Morphological Annotation](https://arxiv.org/pdf/2202.12832.pdf)
2. [Morphological Inflection Generation with Hard Monotonic Attention](https://aclanthology.org/P17-1183.pdf)
3. [On Biasing Transformer Attention Towards Monotonicity](https://arxiv.org/pdf/2104.03945.pdf)
4. [Monotonic Multihead Attention](https://arxiv.org/pdf/1909.12406.pdf)
5. [Applying the Transformer to Character-level Transduction](https://arxiv.org/pdf/2005.10213.pdf)
6. [Pushing the Limits of Low-Resource Morphological Inflection](https://arxiv.org/pdf/1908.05838.pdf)
7. [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/pdf/2107.13586.pdf)
8. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/pdf/2101.00190.pdf)
9. [OpenPrompt repo](https://github.com/thunlp/OpenPrompt)
10. [Exact Hard Monotonic Attention for Character-Level Transduction](https://arxiv.org/pdf/1905.06319.pdf)

