import torch
from torch.utils.data import DataLoader
from dataloader import Parser, WordLoader, DataLoader

# Configurations
task       = "task3_analysis"
pad_to     = 50
epochs     = 100
batch_size = 16
lr         = 5e-4

# Dataset
parser        = Parser()
train         = parser.parse_file("./task3/analysis/tur.trn") # 10,000
val           = parser.parse_file("./task3/analysis/tur.dev") # 1,000
train_dataset = WordLoader(train, pad_to=pad_to)
val_dataset   = WordLoader(val,   pad_to=pad_to)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
print(len(train_loader))
print(len(val_loader))

# Set val vocab same with train vocab
val_dataset.vocab = train_dataset.vocab 
surf_vocab        = train_dataset.vocab.surf_decoder
feature_vocab     = train_dataset.vocab.feat_decoder
print(f"surf: {len(surf_vocab)}") # 35
print(f"feat: {len(feature_vocab)}") # 129

