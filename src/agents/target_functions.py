import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

import random

from torch.utils.data import DataLoader


class RecurrentScalarPredictor(nn.Module):
    def __init__(self, input_dims: int, n_output_dims: int, n_latent_var: int):
        """
        Similar to the RecurrentCritic class but with a linear instead of convolutional layer.
        :param input_dims: Dims of the flattened of the input space.
        :param n_output_dims: Number of output dimensions.
        :param n_latent_var: Number of hidden units in the hidden linear layer.
        """
        super(RecurrentScalarPredictor, self).__init__()
        # Set up critic architecture
        self.input_dims = input_dims
        self.output_dims = n_output_dims
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dims, n_latent_var),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(n_latent_var, n_latent_var, num_layers=1, batch_first=False)
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_latent_var, n_output_dims),
        )
        self.hidden_state = None

    def reset(self):
        self.hidden_state = None

    def forward(self, state_bs: torch.Tensor, time_dim: bool = True):
        if time_dim:
            T, bsz = state_bs.shape[:2]
            state_bs = state_bs.flatten(end_dim=1)
            state_encoding = self.state_encoder(state_bs).view(T, bsz, -1)  # (T, bsz, n_latent_var)
            out, _ = self.rnn(state_encoding)
        else:
            state_encoding = self.state_encoder(state_bs).unsqueeze(dim=0)  # (1, bsz, n_latent_var)
            if self.hidden_state is None:
                out, self.hidden_state = self.rnn(state_encoding)
            else:
                out, self.hidden_state = self.rnn(state_encoding, self.hidden_state)

        return self.critic(out)


class ScalarPredictor(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, hidden_layer_size: int = 32):
        super(ScalarPredictor, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_layer_size = hidden_layer_size

        self.model = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, self.output_dims)
        )

    def forward(self, x):
        return self.model(x)


class DiscreteClassifier(nn.Module):
    def __init__(self, num_classes: int, input_dims: int, hidden_layer_size: int = 32, num_output_dims: int = 1):
        super(DiscreteClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_dims = input_dims
        self.hidden_layer_size = hidden_layer_size
        self.num_output_dims = num_output_dims

        self.model = self.init_model()

    def init_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, self.num_classes * self.num_output_dims),
        )
        return model

    def forward(self, x):
        """
        Can have distributions over multiple output dimensions. The output tensor is then of shape
        (bsz * num_output_dims, num_classes), softmaxed over the last dimension - it is assumed each output
        dimension has the same number of possible classes.
        :param x: A tensor of shape (bsz, input_dims) representing the input.
        """
        logits = self.model(x)
        if self.num_output_dims > 1:
            logits = logits.view(x.size(0) * self.num_output_dims, self.num_classes)

        return F.log_softmax(logits, dim=-1)


class TestContinuousPredictor(L.LightningModule):
    def __init__(self, input_dims: int, hidden_layer_size: int = 32, batch_size: int = 64):
        super().__init__()
        self.num_workers = 4
        self.bsz = batch_size

        self.input_dims = input_dims
        self.hidden_layer_size = hidden_layer_size

        self.model = ScalarPredictor(input_dims, hidden_layer_size)
        self.mse_loss = nn.MSELoss()
        self.train_dataset = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.bsz,
            shuffle=True,
            num_workers=self.num_workers)

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def loss(self, y_hat, y):
        return self.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        """
        :param batch: A tuple of (X, y) where X is a batch of input variables (bsz, **feature_dims) and y is a batch
        of labels (bsz, ).
        :param batch_idx: The index of the current batch.
        """
        X, y = batch
        y_hat = self(X)
        loss = self.loss(y_hat, y)
        self.log_dict({
            'train_loss': loss,
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss


class TestDiscreteClassifier(L.LightningModule):
    def __init__(self, num_classes: int, num_input_dims: int, hidden_layer_size: int = 32, batch_size: int = 64,
                 num_output_dims: int = 1):
        """
        Variational inference of the posterior distribution over a discrete goal space conditioned on fixed values s_1
        and s_2 as well as an input variable.
        :param num_classes: An integer representing the number of discrete output classes.
        :param num_input_dims: An integer representing the number of input dimensions.
        """
        super().__init__()
        self.num_workers = 4
        self.bsz = batch_size

        self.input_dims = num_input_dims
        self.num_output_dims = num_output_dims
        self.num_classes = num_classes
        self.hidden_layer_size = hidden_layer_size

        self.model = DiscreteClassifier(num_classes * num_output_dims, num_input_dims, hidden_layer_size)
        self.log_likelihood_loss = nn.NLLLoss()
        self.train_dataset = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.bsz,
            shuffle=True,
            num_workers=self.num_workers)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def loss(self, y_hat, y):
        return self.log_likelihood_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        """
        :param batch: A tuple of (X, y) where X is a batch of input variables (bsz, **feature_dims) and y is a batch
        of labels (bsz, ) or (bsz, num_output_dims). If the latter, the output of the model is of shape
        (bsz * num_output_dims, num_classes).
        :param batch_idx: The index of the current batch.
        """
        X, y = batch
        if len(y.shape) == 2:
            bsz, num_output_dims = y.shape
            assert num_output_dims == self.num_output_dims
            y = y.view(bsz * num_output_dims)
        y_hat = self(X)
        loss = self.loss(y_hat, y)
        self.log_dict({
            'train_loss': loss,
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss


if __name__ == '__main__':
    input_dims = 1
    num_input_classes = 5
    probs = [0.1, 0.2, 0.3, 0.4]
    bsz = 1024
    num_classes = len(probs)
    class_probs = [random.sample(probs, num_classes) for _ in range(num_input_classes)]
    class_dists = [torch.distributions.categorical.Categorical(torch.tensor(class_prob)) for class_prob in class_probs]
    Xs = []
    ys = []
    for i in range(bsz):
        idx = random.randint(0, len(probs))
        Xs.append(torch.tensor([idx]))
        ys.append(class_dists[idx].sample())
    Xs = torch.stack(Xs).float()
    ys = torch.stack(ys)
    train_dataset = torch.utils.data.TensorDataset(Xs, ys)

    dde = TestDiscreteClassifier(num_classes, input_dims, batch_size=bsz)
    dde.train_dataset = train_dataset
    trainer = L.Trainer(max_epochs=40)
    trainer.fit(dde)
    test_data = torch.tensor(range(num_input_classes)).unsqueeze(-1).float()
    print("True distributions:")
    print(torch.tensor(class_probs))
    print("Estimated distributions:")
    print(torch.exp(dde(test_data)))
