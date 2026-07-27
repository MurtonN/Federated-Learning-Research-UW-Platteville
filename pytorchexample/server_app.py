"""pytorchexample: A Flower / PyTorch app."""

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from pytorchexample.custom_strategy import ReputationFedAvg
from pytorchexample.task import Net

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main execution flow for ServerApp."""

    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)
    num_rounds: int = context.run_config.get("num-server-rounds", 10)

    # Initialize PyTorch network and convert to ArrayRecord
    net = Net()
    initial_arrays = ArrayRecord(net.state_dict())

    train_config = ConfigRecord({"lr": 0.01})

    # Initialize custom strategy
    strategy = ReputationFedAvg(
        fraction_evaluate=fraction_evaluate,
        decay_factor=0.5, # determines how far the quality drops/rises when rejected/accepted
        min_reputation_threshold=0.2, # determines the cutoff for ignored nodes
    )

    # Start Federated Learning workflow
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=train_config,
        num_rounds=num_rounds,
    )