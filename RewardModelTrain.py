# !/usr/bin/env python3
"""

Function: Training Reward Model
Author: TyFang
Date: 2023/11/29
"""
import os
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torchvision.models as models
from RewardModel import RewardModel, compute_rank_list_loss
from iTrainingLogger import iSummaryWriter
from RewardModelDataset import RewardModelDataset


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='bert-base-chinese', type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="train set.")
parser.add_argument("--train_files", default=None, type=str, help="train set　files.")
parser.add_argument("--eval_path", default=None, type=str, help="test set.")
parser.add_argument("--eval_files", default=None, type=str, help="test set　files.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default=None, type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=100, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, data_loader):
    """
    Evaluate the training performance of the current model on the test set.

    """
    model.eval()
    with torch.no_grad():
        batch_rank_rewards = []
        for batch in data_loader:
            for batch_idx in range(len(batch['first'])):
                rank_rewards = []
                first=torch.unsqueeze(batch['first'][batch_idx], 0).to(args.device)
                second=torch.unsqueeze(batch['second'][batch_idx], 0).to(args.device)
                third=torch.unsqueeze(batch['third'][batch_idx], 0).to(args.device)
                forth=torch.unsqueeze(batch['forth'][batch_idx], 0).to(args.device)
                label=torch.unsqueeze(batch['label'][batch_idx], 0).to(args.device)
                rank_rewards.append(model(first,label)[0])
                rank_rewards.append(model(second,label)[0])
                rank_rewards.append(model(third,label)[0])
                rank_rewards.append(model(forth,label)[0])
                batch_rank_rewards.append(rank_rewards)                 


    model.train()
    total_ranklist, right_ranklist = 0, 0
    for rank_rewards in batch_rank_rewards:
        rank_rewards = [t.cpu().float() for t in rank_rewards]
        rank_rewards_sorted = sorted(rank_rewards, reverse=True)
        total_ranklist += 1
        if rank_rewards_sorted == rank_rewards:
            right_ranklist += 1
            # print(rank_rewards)
    return right_ranklist / total_ranklist


def train():

    encoder = models.vgg16(pretrained=True)
    encoder.features[0]= torch.nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
    model = RewardModel(encoder)

    
    train_dataset=RewardModelDataset(args.train_files,args.train_path)
    eval_dataset=RewardModelDataset(args.eval_files,args.eval_files)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model.to(args.device)


    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )


    loss_list = []
    tic_train = time.time()
    global_step, best_acc = 0, 0
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            batch_rank_rewards = []
            for batch_idx in range(len(batch['first'])):
                rank_rewards = []
                first=torch.unsqueeze(batch['first'][batch_idx], 0).to(args.device)
                second=torch.unsqueeze(batch['second'][batch_idx], 0).to(args.device)
                third=torch.unsqueeze(batch['third'][batch_idx], 0).to(args.device)
                forth=torch.unsqueeze(batch['forth'][batch_idx], 0).to(args.device)
                label=torch.unsqueeze(batch['label'][batch_idx], 0).to(args.device)

                rank_rewards.append(model(first,label)[0])
                rank_rewards.append(model(second,label)[0])
                rank_rewards.append(model(third,label)[0])
                rank_rewards.append(model(forth,label)[0])
                
                batch_rank_rewards.append(rank_rewards)                 
            loss = compute_rank_list_loss(batch_rank_rewards, device=args.device)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                torch.save(model, os.path.join(cur_save_dir, 'model.pt'))

                acc = evaluate_model(model, eval_dataloader)
                writer.add_scalar('eval/accuracy', acc, global_step)
                writer.record()
                print("Evaluation acc: %.5f" % (acc))
                if acc > best_acc:
                    print(
                        f"best ACC performence has been updated: {best_acc:.5f} --> {acc:.5f}"
                    )
                    best_acc = acc
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                tic_train = time.time()


if __name__ == '__main__':
    train()
