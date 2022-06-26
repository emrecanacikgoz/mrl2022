import torch, logging, os, argparse
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloader import Parser, WordLoader
from model.model import Morse
from training import train

#### DON'T FORGET TO CHANGE THIS !!! ######
logger_file_name   = 'experiment0'        # Add ExpNUMBER !!!         
logger_folder_name = "EXPERIMENTS/exp0"   # Add ExpNUMBER !!!
###########################################


# Set loggers 
if not os.path.exists(logger_folder_name):
    os.mkdir(logger_folder_name)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | | %(levelname)s | | %(message)s')
logger_file_name = os.path.join(logger_folder_name, logger_file_name)
file_handler = logging.FileHandler(logger_file_name,'w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info('Code started \n')
# set args
parser      = argparse.ArgumentParser(description='')
args        = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Configurations
args.task       = "task3_analysis"
args.pad_to     = 60
args.epochs     = 100
args.batch_size = 16
args.lr         = 5e-4


# Dataset
parser        = Parser()
train_data    = parser.parse_file("./analysis/tur.trn") # 10,000
val_data      = parser.parse_file("./analysis/tur.dev") # 1,000
train_dataset = WordLoader(train_data, pad_to=args.pad_to)
val_dataset   = WordLoader(val_data,   pad_to=args.pad_to)
train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
print(len(train_loader))
print(len(val_loader))

# Set val vocab same with train vocab
val_dataset.vocab = train_dataset.vocab 
surf_vocab        = train_dataset.vocab.surf_decoder
feature_vocab     = train_dataset.vocab.feat_decoder
print(f"surf: {len(surf_vocab)}") # 35
print(f"feat: {len(feature_vocab)}") # 129


# Model
args.mname   = '3Encoder_3Decoder'
embed_dim    = 256
num_heads    = 16
dropout_rate = 0.15 
args.model = Morse(input_vocab=surf_vocab, output_vocab=feature_vocab, embed_dim=embed_dim, num_heads=num_heads, dropout_rate=dropout_rate)
args.model.to(args.device)

# Loss and optimizer
args.criterion = nn.CrossEntropyLoss(ignore_index=0)
args.optimizer = optim.AdamW(args.model.parameters(), lr=args.lr, betas=(0.9, 0.95))

# Terminal operations
logger.info(f"\nUsing device: {str(args.device)}")
logger.info(f"Number of Epochs: {args.epochs}")
logger.info(f"Batch Size: {args.batch_size}")
logger.info(f"Learning rate: {args.lr}")
logger.info(f"Number of parameters {len(torch.nn.utils.parameters_to_vector(args.model.parameters()))}")
logger.info(f"Embedding Dimension: {embed_dim}")
logger.info(f"Number of heads in Attention: {num_heads}")
logger.info(f"Dropout rate: {dropout_rate}\n")

# File Operations
modelname              = args.mname+'/results/'+str(len(train_data))+'_instances'
args.results_file_name = os.path.join(logger_folder_name, modelname)
try:
    os.makedirs(args.results_file_name)
    print("Directory " , args.results_file_name,  " Created ")
except FileExistsError:
    print("Directory " , args.results_file_name,  " already exists")
args.save_path = args.results_file_name + str(args.epochs)+'epochs.pt'
fig_path       = args.results_file_name + str(args.epochs)+'epochs.png'

# Plotting
args.fig, args.axs = plt.subplots(1)
args.plt_style     = pstyle = '-'

# Training
train(train_loader, val_loader, logger, args)

# Save figure
plt.savefig(fig_path)