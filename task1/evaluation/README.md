From neural-transducer repo home directory run the following:
bash eval.sh language_code architecture model_working_directory model_checkpoint_path data_directory GPU_ID

For example:
bash eval.sh tur transformer checkpoints/transformer/transformer/hall-sigmorphon17-task1-dropout0.3/hall-1 checkpoints/transformer/transformer/hall-sigmorphon17-task1-dropout0.3/hall-1/tur-high-.nll_0.6837.acc_99.8.dist_0.002.epoch_663 /pscratch/sd/c/chubakov/code/morphology/inflection/data 0