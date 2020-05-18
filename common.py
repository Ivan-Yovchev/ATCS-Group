from torch import nn
import torch

class Common(nn.Module):

    def __init__(self, cnn, n_filters=None, encoder = lambda x : x):
        super().__init__()

        self.encoder = encoder
        self.cnn = cnn

        # Proto learner
        self.n_filters = n_filters

    def forward(self, *args):
        return self.cnn(self.encoder(*args))

    def get_outputlayer(self, S, n_classes):

        if self.n_filters is None:
            return None

        C = torch.zeros(n_classes, self.n_filters)
        l2i = {}

        for x, label in S:

            x[0] = x[0].cpu()
            outputs = self(*x).detach().squeeze()

            for i in range(label.shape[0]):
                # Label to index
                l2i[label[i].item()] = l2i.get(label[i].item(), len(l2i))
                idx = l2i[label[i].item()]

                # Accumulate latent vectors
                C[idx] += outputs[i]

        # Assume equal number of examples for each class
        samples_per_class = len(S) / len(l2i)
        C /= samples_per_class

        # Replace W and b in linear layer
        linear = nn.Linear(self.n_filters, n_classes)
        linear.weight = nn.Parameter(2*C)
        linear.bias = nn.Parameter(-torch.diag(C @ C.T))

        linear.to(self.cnn.device)
        return linear, C.detach() # C should already be detached

    def train(self):

        if hasattr(self.encoder, "train"):
            self.encoder.train()

        if hasattr(self.cnn, "train"):
            self.cnn.train()

    def eval(self):

        if hasattr(self.encoder, "eval"):
            self.encoder.eval()

        if hasattr(self.cnn, "eval"):
            self.cnn.eval()
