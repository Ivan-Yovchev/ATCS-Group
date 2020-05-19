import torch

from torch import nn


class EvalModeBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, **kw):
        super(EvalModeBatchNorm1d, self).__init__(*args, **kw)
        self.training = False

    def train(self, mode: bool = True):
        for module in self.children():
            module.train(mode)
        return self


class CNNBlock(nn.Module):
    """docstring for ClassName"""

    def __init__(self, in_c, out_c, k_size, momentum, device, batch_norm_eval=False):
        super(CNNBlock, self).__init__()
        self.device = device

        self.block = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=k_size),
            nn.ReLU(),
            EvalModeBatchNorm1d(num_features=out_c, momentum=momentum) if batch_norm_eval else nn.BatchNorm1d(
                num_features=out_c, momentum=momentum)
        )
        self.to(device)

    def forward(self, x):
        return torch.max(self.block(x), axis=2).values

    def train(self, mode: bool = True):
        self.block.train()
        self.block.parameters()


class CNNModel(nn.Module):
    """docstring for CNNModel"""

    def __init__(self, embed_size, device, n_filters=128, momentum=0.7, filter_sizes=[2, 3, 4, 5, 6],
                 batch_norm_eval=False):
        super(CNNModel, self).__init__()

        self.device = device
        self.filter_sizes = filter_sizes
        self.cnn_blocks = nn.ModuleList([])

        for f_size in filter_sizes:
            self.cnn_blocks.append(
                CNNBlock(embed_size, n_filters, f_size, momentum, device, batch_norm_eval=batch_norm_eval))

        self.to(device)

    def forward(self, x):
        block_outs = []
        for block in self.cnn_blocks:
            block_outs.append(block.forward(x))
 
        input_to_dense = torch.cat(block_outs, 1)

        return input_to_dense

    def initialize_weights(self, init_function):
        for t in self.parameters():
            if len(t.shape) >= 2:
                init_function(t)

    def get_n_blocks(self):
        return len(self.cnn_blocks)

    def get_max_kernel(self):
        return max(self.filter_sizes)
