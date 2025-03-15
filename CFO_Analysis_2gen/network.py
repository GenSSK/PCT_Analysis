import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



# network definition
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(63, 600)
        # self.fc2 = nn.Linear(600, 300)
        # self.fc3 = nn.Linear(300, 100)
        # self.fc4 = nn.Linear(100, 50)
        # self.fc5 = nn.Linear(50, 1)

        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

        # self.fc1 = nn.Linear(6, 32)
        # self.fc2 = nn.Linear(32, 16)
        # self.fc3 = nn.Linear(16, 8)
        # self.fc4 = nn.Linear(8, 4)
        # self.fc5 = nn.Linear(4, 1)

        # self.fc1 = nn.Linear(6, 400)
        # self.fc2 = nn.Linear(400, 400)
        # self.fc3 = nn.Linear(400, 400)
        # self.fc4 = nn.Linear(400, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        # x = F.tanh(self.fc3(x))
        # x = F.tanh(self.fc4(x))
        # x = self.fc5(x)
        # return x

class EarlyStopping:
    """早期停止を助けるクラス"""

    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): パフォーマンスが向上しないエポック数。この数を超えると訓練が停止される。
            verbose (bool): 早期停止の出力を表示するかどうか。
            delta (float): 改善と見なされる最小の変化。
            path (str): モデルを保存するパス。
            trace_func (function): 出力用の関数。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """検証損失が改善された場合、モデルを保存します。"""
        # if self.verbose:
            # self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss