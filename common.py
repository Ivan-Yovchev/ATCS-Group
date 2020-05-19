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

    def get_outputlayer(self, S):

        if self.n_filters is None:
            return None

        class_set = S.get_classes()
        n_classes = len(class_set)

        assert (n_classes == len(class_set))

        relabeled = self.__relabel(sorted(class_set))

        C = torch.zeros(n_classes, self.n_filters)
        counts = torch.zeros(n_classes, 1)
        l2i = {}
        
        for x, label in S:

            x = (el.cpu() for el in x)
            outputs = self(*x)

            for i in range(label.shape[0]):
                # Label to index
                new_label = relabeled[label[i].item()]
                l2i[new_label] = l2i.get(new_label, len(l2i))
                idx = l2i[new_label]

                # Accumulate latent vectors
                C[idx] += outputs[i]
                counts[idx, :] += 1

        # Assume equal number of examples for each class
        C /= counts

        # Replace W and b in linear layer
        linear = nn.Linear(self.n_filters, n_classes)
        linear.weight = nn.Parameter(2*C)
        linear.bias = nn.Parameter(-torch.diag(C @ C.T))

        linear.to(self.cnn.device)
        return linear, C.detach() # C should already be detached

    def __relabel(self, sorted_classes):
        return {label: new_label for new_label, label in enumerate(sorted_classes)}

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
