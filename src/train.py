"""
Training script for GPT2ForSequenceClassification on 20 Newsgroups dataset.

This script trains a GPT-2 based classifier without using any HuggingFace libraries.
"""

import argparse
import json
import os
import time
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gpt2 import GPT2ForSequenceClassification, GPT2Config
from torch.utils.tensorboard import SummaryWriter


class NewsGroupDataset(Dataset):
    def __init__(self, data_path, max_length=512):
        self.samples = [] #list to stored truncated samples
        with open(data_path, 'r') as f: #open the files for read
            for line in f:
                data = json.loads(line)
                tokens = data['token_ids'][:max_length] #truncated to be 512 max
                tokens = tokens + [0] * (max_length - len(tokens)) # filled in rest with zero if not 512
                self.samples.append({
                    'token_ids': torch.tensor(tokens, dtype=torch.long),
                    'label': torch.tensor(data['label'], dtype=torch.long)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
if __name__ == "__main__":
    # TODO: implement the training loop for GPT2ForSequenceClassification on the 20 Newsgroups dataset.
    # You can use any techniques or implementations you like.
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default='data/20_newsgroups_train.jsonl')
    parser.add_argument("--bin_path", type=str)
    parser.add_argument("--eval_data_path", type=str, default='data/20_newsgroups_val.jsonl')
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    #load training_data
    train_dataset = NewsGroupDataset(args.train_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #load val_data
    val_dataset = NewsGroupDataset(args.eval_data_path)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    config = GPT2Config()
    model = GPT2ForSequenceClassification(config=config, lm_bin_path=args.bin_path)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # optimizer for gradient descent.
    loss_fn = nn.CrossEntropyLoss() #loss function
    best_accuracy = 0
    # before epoch loop
    writer = SummaryWriter()
    global_step = 0
    print(f"Loading model from: {args.bin_path}")
    print(f"File exists: {os.path.exists(args.bin_path)}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['token_ids'].to(device)
            labels = batch['label'].to(device)
            output = model(input_ids)
            logits = output.logits
            loss = loss_fn(logits, labels) #compute loss
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
            optimizer.zero_grad() # zero out old gradients
            loss.backward() # do backward pass
            optimizer.step() #gradient descent
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} | Batch {batch_idx} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f}")
        # after each epoch's training loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['token_ids'].to(device)
                labels = batch['label'].to(device)

                output = model(input_ids)
                logits = output.logits

                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch + 1} | Val Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        # save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'checkpoints/classifier_model.pth')
            print(f"New best model saved! Accuracy: {accuracy:.4f}")
        writer.add_scalar('Accuracy/val', accuracy, epoch)
    writer.close()
