import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



# network definition
class Net(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=1, activation='relu'):
        super(Net, self).__init__()

        self.activation = activation
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 最後の層以外に活性化関数を適用
                if self.activation == 'relu':
                    x = F.relu(x)
                elif self.activation == 'tanh':
                    x = torch.tanh(x)
                # 他の活性化関数も必要なら追加可能
        return x

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