from models.chemprop_affinity.config import ExperimentConfig

from chemprop import data, featurizers, models, nn
from typing import Any

def build_chemprop_model(
    config: ExperimentConfig,
    scaler: Any | None = None 
) -> Any:
    """Build the Chemprop MPNN to predict affinity."""
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform)
    batch_norm = config.model.batchnorm
    # metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()] # Only the first metric is used for training and early stopping
    mpnn = models.MPNN(mp, agg, ffn, batch_norm) # TODO add metrics to be implemented?
    return mpnn