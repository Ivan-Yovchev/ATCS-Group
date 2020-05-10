import torch

from torch import nn


class CNNBlock(nn.Module):
    """docstring for ClassName"""

    def __init__(self, in_c, out_c, k_size, max_len, momentum, device):
        super(CNNBlock, self).__init__()
        self.device = device

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=k_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=out_c, momentum=momentum),
            nn.MaxPool1d(kernel_size=(max_len - k_size))
        )

        self.to(device)

    def forward(self, x):
        return self.block(x)


class CNNModel(nn.Module):
    """docstring for CNNModel"""

    def __init__(self, embed_size, max_len, device, n_filters=128, momentum=0.7, filter_sizes=[2, 3, 4, 5, 6]):
        super(CNNModel, self).__init__()

        self.device = device
        self.cnn_blocks = nn.ModuleList([])
        for f_size in filter_sizes:
            self.cnn_blocks.append(CNNBlock(embed_size, n_filters, f_size, max_len, momentum, device))

        # self.classifier = classifier
        self.to(device)

    def forward(self, x):
        block_outs = []
        for block in self.cnn_blocks:
            block_outs.append(block.forward(x))

        input_to_dense = torch.cat(block_outs, 1).flatten(start_dim=1)
        # return self.classifier(input_to_dense)
        return input_to_dense


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dense = nn.Sequential(nn.Linear(5 * 128, 1), nn.Sigmoid())
    dense.to(device=device)
    model = CNNModel(768, 300, device=device)

    x = torch.randn(8, 768, 300).to(device)

    print(model(x))
