"""
Function: Build reward model
Author: TyFang
Date: 2023/11/29
"""


from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
import random
import numpy as np



#Check if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RewardModel(nn.Module):

    def __init__(self, encoder):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone,  ernie 3.0
        """
        super().__init__()
        self.encoder = encoder
        self.reward_layer = nn.Linear(1000, 1)

    def forward(self,inputs,label) -> torch.tensor:
        """
         forward function

        Args:
            inputs (torch.tensor,torch.tensor): (batch, seq_len)
        Returns:
            reward: (batch, 1)
        """
        inputs=inputs[:,0,:,:].unsqueeze(1)  
        label=label[:,0,:,:].unsqueeze(1) 
        
        inputs = torch.cat([inputs, label], dim=1)
        pooler_output = self.encoder(inputs)                              # (batch, hidden_size)
        reward = self.reward_layer(pooler_output)       # (batch, 1)
        return reward
    
def compute_rank_list_loss(rank_rewards_list: List[List[torch.tensor]], device='cpu') -> torch.Tensor:
    """
    Calculate the rank loss based on the reward list of a given ordered  rank list.

    Args:
        rank_rewards_list (torch.tensor): 
        device (str):
        
    Returns:
        loss (torch.tensor): 
    """
    if type(rank_rewards_list) != list:
        raise TypeError(f'@param rank_rewards expected "list", received {type(rank_rewards)}.')
    
    loss, add_count = torch.tensor([0]).to(device), 0
    for rank_rewards in rank_rewards_list:
        for i in range(len(rank_rewards)-1):                                   
            for j in range(i+1, len(rank_rewards)):
                diff = F.logsigmoid(rank_rewards[i] - rank_rewards[j])         
                loss = loss + diff
                add_count += 1
    loss = loss / add_count
    return -loss                                                               


# if __name__ == '__main__':
#     encoder = models.vgg16(pretrained=True)
#     encoder.features[0]= torch.nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
#     model = RewardModel(encoder).to(device)
