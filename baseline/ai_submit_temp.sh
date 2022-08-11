#!/bin/bash
#SBATCH --job-name=eacikgoz17_transformer_baseline_analysis_deu # Job name
#SBATCH --nodes=1 # Run on a single node
#SBATCH --ntasks-per-node=2
#SBATCH --partition=ai # Run in ai queue
#SBATCH --qos=ai 
#SBATCH --account=ai 
#SBATCH --gres=gpu:tesla_t4:1 
#SBATCH --mem=20G 
#SBATCH --time=7-0:0:0 # Time limit days-hours:minutes:seconds
#SBATCH --output=test-%j.out # Standard output and error log
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=eacikgoz17@ku.edu.tr # Where to send mail


echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

module load anaconda/3.6
source activate eacikgoz17
nvidia-smi

python transformer.py analysis deu

source deactivate