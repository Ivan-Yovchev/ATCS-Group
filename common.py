from torch import nn
import torch


class Common(nn.Module):

    def __init__(self, cnn, n_filters=None, encoder=lambda x: x):
        super().__init__()

        self.encoder = encoder
        self.cnn = cnn

        # Proto learner
        self.n_filters = n_filters

    def forward(self, x):
        """Forward method of nn.Module. Takes in input a list or a tuple of tensors.

        :param x: a list or a tuple that will be fed to the encoder.
        If there is no finetuning, a singleton containing the output of BERT is passed.
        If there is finetuning, the encoder will be the BERT model and it will require (indices, mask) produced by the
        tokenizer.
        :return:
        """
        return self.cnn(self.encoder(*x))

    def get_outputlayer(self, S):

        if self.n_filters is None:
            return None

        class_set = S.get_classes()
        n_classes = len(class_set)

        C = [torch.zeros(self.n_filters) for _ in range(n_classes)]
        counts = torch.zeros(n_classes, 1)

        for x, label in S:

            x = (el.cpu() for el in x)
            outputs = self(x)

            for i in range(label.shape[0]):
                # Label to index
                idx = label[i].item()

                # Accumulate latent vectors
                C[idx] = C[idx] + outputs[i]
                counts[idx, :] += 1

        # Assume equal number of examples for each class
        C = torch.stack(C) / counts

        # Replace W and b in linear layer
        linear = nn.Linear(self.n_filters, n_classes)

        weight = 2 * C.detach()
        bias = -torch.diag(C.detach() @ C.detach().T)

        # normalize
        linear.weight.data = nn.Parameter(weight / weight.abs().sum(dim=-1).unsqueeze(-1))
        linear.bias.data = nn.Parameter(bias / bias.abs().sum())

        linear.to(self.cnn.device)
        return linear, C  # C should already be detached

    def train(self, mode: bool = True):

        if hasattr(self.encoder, "train"):
            self.encoder.train(mode)

        if hasattr(self.cnn, "train"):
            self.cnn.train(mode)
