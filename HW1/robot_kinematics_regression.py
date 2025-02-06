from torch import nn


class MLP(nn.Module):
    """
    Regression approximation via 3-FC NN layers.
    The network input features are one-dimensional as well as the output features.
    The network hidden sizes are 128.
    Activations are ReLU
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """
        :param x: Tensor of size (N, 3)
        :return: y_hat: Tensor of size (N, 2)
        """
        y_hat = self.model(x)
        return y_hat
