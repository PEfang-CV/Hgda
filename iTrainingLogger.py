# !/usr/bin/env python3
"""
Function: Recorder during training process
Author: TyFang
Date: 2023/11/29
"""
import os

import numpy as np
import matplotlib.pyplot as plt


class iSummaryWriter(object):

    def __init__(self, log_path: str, log_name: str, params=[], extention='.png', max_columns=2,
                 log_title=None, figsize=None):
        """
        Initialize the function and create a log class.

        Args:
            log_path (str): Log storage folder
            log_name (str): log file name
            parmas (list): List of parameter names to be recorded，e.g. -> ["loss", "reward", ...]
            extension (str): Image storage format
            max_columns (int): 
        """
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_name = log_name
        self.extention = extention
        self.max_param_index = -1
        self.max_columns_threshold = max_columns
        self.figsize = figsize
        self.params_dict = self.create_params_dict(params)
        self.log_title = log_title
        self.init_plt()
        self.update_ax_list()

    def init_plt(self) -> None:
        plt.style.use('seaborn-darkgrid')

    def create_params_dict(self, params: list) -> dict:
        """
        Create a monitoring variable dictionary based on the list of variable names that need to be recorded as input.

        Args:
            params (list): List of monitoring variable names

        Returns:
            dict: Monitor Variable Name Dictionary -> {
                'loss': {'values': [0.44, 0.32, ...], 'epochs': [10, 20, ...], 'index': 0},
                'reward': {'values': [10.2, 13.2, ...], 'epochs': [10, 20, ...], 'index': 1},
                ...
            }
        """
        params_dict = {}
        for i, param in enumerate(params):
            params_dict[param] = {'values': [], 'epochs': [], 'index': i}
            self.max_param_index = i
        return params_dict

    def update_ax_list(self) -> None:
        """
        Assign a graph area to each variable based on the current monitoring variable dictionary.
        """

        params_num = self.max_param_index + 1
        if params_num <= 0:
            return

        self.max_columns = params_num if params_num < self.max_columns_threshold else self.max_columns_threshold
        max_rows = (params_num - 1) // self.max_columns + 1   # * 所有变量最多几行
        figsize = self.figsize if self.figsize else (self.max_columns * 6,max_rows * 3)    # 根据图个数计算整个图的figsize
        self.fig, self.axes = plt.subplots(max_rows, self.max_columns, figsize=figsize)

        if params_num > 1 and len(self.axes.shape) == 1:
            self.axes = np.expand_dims(self.axes, axis=0)

        log_title = self.log_title if self.log_title else '[Training Log] {}'.format(
            self.log_name)
        self.fig.suptitle(log_title, fontsize=15)

    def add_scalar(self, param: str, value: float, epoch: int) -> None:
        """
       Add a new variable value record.

        Args:
            param (str): -> 'loss'
            value (float):
            epoch (int): 
        """
       
        if param not in self.params_dict:
            self.max_param_index += 1
            self.params_dict[param] = {'values': [],
                                       'epochs': [], 'index': self.max_param_index}
            self.update_ax_list()

        self.params_dict[param]['values'].append(value)
        self.params_dict[param]['epochs'].append(epoch)

    def record(self, dpi=200) -> None:
        """
        Call this interface to record the status of all currently monitored variables in the class and save the results to a local file.
        """
        for param, param_elements in self.params_dict.items():
            param_index = param_elements["index"]
            param_row, param_column = param_index // self.max_columns, param_index % self.max_columns
            ax = self.axes[param_row, param_column] if self.max_param_index > 0 else self.axes
            # ax.set_title(param)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(param)
            ax.plot(self.params_dict[param]['epochs'],
                    self.params_dict[param]['values'],
                    color='darkorange')

        plt.savefig(os.path.join(self.log_path,
                    self.log_name + self.extention), dpi=dpi)


# if __name__ == '__main__':
#     import random
#     import time

#     n_epochs = 10
#     log_path, log_name = './', 'test'
#     writer = iSummaryWriter(log_path=log_path, log_name=log_name)
#     for i in range(n_epochs):
#         loss, reward = 100 - random.random() * i, random.random() * i
#         writer.add_scalar('loss', loss, i)
#         writer.add_scalar('reward', reward, i)
#         writer.add_scalar('random', reward, i)
#         writer.record()
#         print("Log has been saved at: {}".format(
#             os.path.join(log_path, log_name)))
#         time.sleep(3)