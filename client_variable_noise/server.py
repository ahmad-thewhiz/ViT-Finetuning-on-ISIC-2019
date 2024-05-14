import flwr as fl
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping

strategy = fl.server.strategy.FedAvg()

# dp_strategy = DifferentialPrivacyClientSideFixedClipping(strategy=strategy, num_sampled_clients=6, noise_multiplier=1.0, clipping_norm=0.5)

fl.server.start_server(
    server_address="0.0.0.0:8083",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
)
