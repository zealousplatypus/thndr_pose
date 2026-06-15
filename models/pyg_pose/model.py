import torch
from torch import nn, optim
import lightning as L
from torch_geometric.nn import GCNConv, global_mean_pool

# define the LightningModule
class LitPoseGNN(L.LightningModule):
    def __init__(self, num_atom_features, hidden_dim, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        num_atom_features = num_atom_features + 3 # atom features, coordinates.
        self.conv1 = GCNConv(in_channels=num_atom_features, out_channels=hidden_dim)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        x, coords, edge_index, batch_idx = batch.x.float(), batch.pos.float(), batch.edge_index, batch.batch
        # coords = torch.zeros_like(coords) # no coordinates for now. coords negative control
        x = torch.cat([x, coords], dim=1)
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = nn.functional.relu(x)
        x = global_mean_pool(x, batch_idx)
        x = self.regressor(x)
        return x


    def training_step(self, batch, batch_idx=None):
        # training_step defines the train loop.
        # it is independent of forward
        x = self.forward(batch)
        y = batch.y.view_as(x).float()
        loss = nn.functional.mse_loss(x, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx=None):
        x = self.forward(batch)
        y = batch.y.view_as(x).float()
        loss = nn.functional.mse_loss(x, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer