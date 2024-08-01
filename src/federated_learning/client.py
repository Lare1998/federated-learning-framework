
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from typing import Dict, List

class ClientModel(nn.Module):
    """A simple neural network for client-side training."""
    def __init__(self):
        super(ClientModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class FederatedClient:
    """Represents a client participating in federated learning."""
    def __init__(self, client_id: int, data_loader, model: nn.Module, learning_rate: float = 0.01):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def get_weights(self) -> OrderedDict:
        """Returns the current model weights."""
        return self.model.state_dict()

    def set_weights(self, weights: OrderedDict):
        """Sets the model weights from the server."""
        self.model.load_state_dict(weights)

    def train(self, epochs: int = 1) -> Dict[str, OrderedDict]:
        """Trains the client model locally for a specified number of epochs."""
        self.model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        print(f"Client {self.client_id} trained for {epochs} epochs.")
        return {"client_id": self.client_id, "weights": self.get_weights()}

if __name__ == "__main__":
    # Dummy DataLoader for demonstration
    class DummyDataLoader:
        def __init__(self, num_samples: int = 100, batch_size: int = 10):
            self.num_samples = num_samples
            self.batch_size = batch_size

        def __iter__(self):
            for _ in range(self.num_samples // self.batch_size):
                data = torch.randn(self.batch_size, 10) # 10 features
                target = torch.randint(0, 2, (self.batch_size,)) # 2 classes
                yield data, target

    # Example Usage
    client_id = 1
    dummy_data = DummyDataLoader()
    client_model = ClientModel()
    client = FederatedClient(client_id, dummy_data, client_model)

    print(f"Initial weights of client {client.client_id}:")
    # print(client.get_weights())

    trained_info = client.train(epochs=2)
    print(f"\nWeights after training for client {client.client_id}:")
    # print(trained_info["weights"])

# Update on 2023-01-02 00:00:00
# Update on 2023-01-02 00:00:00
# Update on 2023-01-04 00:00:00
# Update on 2023-01-05 00:00:00
# Update on 2023-01-06 00:00:00
# Update on 2023-01-09 00:00:00
# Update on 2023-01-11 00:00:00
# Update on 2023-01-12 00:00:00
# Update on 2023-01-16 00:00:00
# Update on 2023-01-16 00:00:00
# Update on 2023-01-18 00:00:00
# Update on 2023-01-23 00:00:00
# Update on 2023-01-24 00:00:00
# Update on 2023-01-24 00:00:00
# Update on 2023-01-26 00:00:00
# Update on 2023-01-26 00:00:00
# Update on 2023-01-27 00:00:00
# Update on 2023-01-30 00:00:00
# Update on 2023-02-02 00:00:00
# Update on 2023-02-06 00:00:00
# Update on 2023-02-07 00:00:00
# Update on 2023-02-07 00:00:00
# Update on 2023-02-07 00:00:00
# Update on 2023-02-08 00:00:00
# Update on 2023-02-10 00:00:00
# Update on 2023-02-13 00:00:00
# Update on 2023-02-13 00:00:00
# Update on 2023-02-14 00:00:00
# Update on 2023-02-14 00:00:00
# Update on 2023-02-20 00:00:00
# Update on 2023-02-21 00:00:00
# Update on 2023-02-21 00:00:00
# Update on 2023-02-21 00:00:00
# Update on 2023-02-22 00:00:00
# Update on 2023-02-22 00:00:00
# Update on 2023-02-24 00:00:00
# Update on 2023-02-24 00:00:00
# Update on 2023-02-28 00:00:00
# Update on 2023-03-01 00:00:00
# Update on 2023-03-02 00:00:00
# Update on 2023-03-03 00:00:00
# Update on 2023-03-03 00:00:00
# Update on 2023-03-07 00:00:00
# Update on 2023-03-09 00:00:00
# Update on 2023-03-09 00:00:00
# Update on 2023-03-13 00:00:00
# Update on 2023-03-13 00:00:00
# Update on 2023-03-14 00:00:00
# Update on 2023-03-15 00:00:00
# Update on 2023-03-17 00:00:00
# Update on 2023-03-20 00:00:00
# Update on 2023-03-21 00:00:00
# Update on 2023-03-23 00:00:00
# Update on 2023-03-24 00:00:00
# Update on 2023-03-27 00:00:00
# Update on 2023-03-28 00:00:00
# Update on 2023-03-29 00:00:00
# Update on 2023-03-31 00:00:00
# Update on 2023-04-03 00:00:00
# Update on 2023-04-03 00:00:00
# Update on 2023-04-03 00:00:00
# Update on 2023-04-04 00:00:00
# Update on 2023-04-06 00:00:00
# Update on 2023-04-10 00:00:00
# Update on 2023-04-14 00:00:00
# Update on 2023-04-14 00:00:00
# Update on 2023-04-17 00:00:00
# Update on 2023-04-17 00:00:00
# Update on 2023-04-18 00:00:00
# Update on 2023-04-20 00:00:00
# Update on 2023-04-21 00:00:00
# Update on 2023-04-25 00:00:00
# Update on 2023-04-25 00:00:00
# Update on 2023-04-26 00:00:00
# Update on 2023-04-27 00:00:00
# Update on 2023-05-01 00:00:00
# Update on 2023-05-04 00:00:00
# Update on 2023-05-04 00:00:00
# Update on 2023-05-05 00:00:00
# Update on 2023-05-08 00:00:00
# Update on 2023-05-09 00:00:00
# Update on 2023-05-16 00:00:00
# Update on 2023-05-17 00:00:00
# Update on 2023-05-17 00:00:00
# Update on 2023-05-18 00:00:00
# Update on 2023-05-18 00:00:00
# Update on 2023-05-19 00:00:00
# Update on 2023-05-22 00:00:00
# Update on 2023-05-23 00:00:00
# Update on 2023-05-26 00:00:00
# Update on 2023-05-29 00:00:00
# Update on 2023-05-31 00:00:00
# Update on 2023-06-01 00:00:00
# Update on 2023-06-02 00:00:00
# Update on 2023-06-05 00:00:00
# Update on 2023-06-07 00:00:00
# Update on 2023-06-07 00:00:00
# Update on 2023-06-08 00:00:00
# Update on 2023-06-09 00:00:00
# Update on 2023-06-09 00:00:00
# Update on 2023-06-12 00:00:00
# Update on 2023-06-21 00:00:00
# Update on 2023-06-28 00:00:00
# Update on 2023-06-28 00:00:00
# Update on 2023-06-29 00:00:00
# Update on 2023-06-29 00:00:00
# Update on 2023-06-29 00:00:00
# Update on 2023-06-30 00:00:00
# Update on 2023-07-04 00:00:00
# Update on 2023-07-11 00:00:00
# Update on 2023-07-12 00:00:00
# Update on 2023-07-12 00:00:00
# Update on 2023-07-12 00:00:00
# Update on 2023-07-18 00:00:00
# Update on 2023-07-18 00:00:00
# Update on 2023-07-19 00:00:00
# Update on 2023-07-25 00:00:00
# Update on 2023-07-28 00:00:00
# Update on 2023-08-04 00:00:00
# Update on 2023-08-07 00:00:00
# Update on 2023-08-09 00:00:00
# Update on 2023-08-09 00:00:00
# Update on 2023-08-10 00:00:00
# Update on 2023-08-11 00:00:00
# Update on 2023-08-15 00:00:00
# Update on 2023-08-16 00:00:00
# Update on 2023-08-23 00:00:00
# Update on 2023-08-24 00:00:00
# Update on 2023-08-24 00:00:00
# Update on 2023-08-25 00:00:00
# Update on 2023-08-28 00:00:00
# Update on 2023-08-29 00:00:00
# Update on 2023-08-30 00:00:00
# Update on 2023-09-01 00:00:00
# Update on 2023-09-05 00:00:00
# Update on 2023-09-06 00:00:00
# Update on 2023-09-14 00:00:00
# Update on 2023-09-18 00:00:00
# Update on 2023-09-19 00:00:00
# Update on 2023-09-22 00:00:00
# Update on 2023-09-25 00:00:00
# Update on 2023-09-27 00:00:00
# Update on 2023-09-28 00:00:00
# Update on 2023-09-29 00:00:00
# Update on 2023-10-02 00:00:00
# Update on 2023-10-03 00:00:00
# Update on 2023-10-03 00:00:00
# Update on 2023-10-04 00:00:00
# Update on 2023-10-05 00:00:00
# Update on 2023-10-05 00:00:00
# Update on 2023-10-06 00:00:00
# Update on 2023-10-06 00:00:00
# Update on 2023-10-09 00:00:00
# Update on 2023-10-09 00:00:00
# Update on 2023-10-10 00:00:00
# Update on 2023-10-10 00:00:00
# Update on 2023-10-12 00:00:00
# Update on 2023-10-12 00:00:00
# Update on 2023-10-16 00:00:00
# Update on 2023-10-18 00:00:00
# Update on 2023-10-18 00:00:00
# Update on 2023-10-18 00:00:00
# Update on 2023-10-23 00:00:00
# Update on 2023-10-24 00:00:00
# Update on 2023-10-25 00:00:00
# Update on 2023-10-27 00:00:00
# Update on 2023-11-01 00:00:00
# Update on 2023-11-02 00:00:00
# Update on 2023-11-02 00:00:00
# Update on 2023-11-03 00:00:00
# Update on 2023-11-06 00:00:00
# Update on 2023-11-06 00:00:00
# Update on 2023-11-06 00:00:00
# Update on 2023-11-16 00:00:00
# Update on 2023-11-17 00:00:00
# Update on 2023-11-20 00:00:00
# Update on 2023-11-21 00:00:00
# Update on 2023-11-22 00:00:00
# Update on 2023-11-23 00:00:00
# Update on 2023-11-24 00:00:00
# Update on 2023-11-28 00:00:00
# Update on 2023-11-28 00:00:00
# Update on 2023-11-29 00:00:00
# Update on 2023-12-04 00:00:00
# Update on 2023-12-04 00:00:00
# Update on 2023-12-04 00:00:00
# Update on 2023-12-07 00:00:00
# Update on 2023-12-07 00:00:00
# Update on 2023-12-12 00:00:00
# Update on 2023-12-13 00:00:00
# Update on 2023-12-15 00:00:00
# Update on 2023-12-15 00:00:00
# Update on 2023-12-20 00:00:00
# Update on 2023-12-21 00:00:00
# Update on 2023-12-25 00:00:00
# Update on 2023-12-25 00:00:00
# Update on 2023-12-27 00:00:00
# Update on 2023-12-29 00:00:00
# Update on 2024-01-01 00:00:00
# Update on 2024-01-01 00:00:00
# Update on 2024-01-02 00:00:00
# Update on 2024-01-03 00:00:00
# Update on 2024-01-04 00:00:00
# Update on 2024-01-05 00:00:00
# Update on 2024-01-10 00:00:00
# Update on 2024-01-11 00:00:00
# Update on 2024-01-11 00:00:00
# Update on 2024-01-11 00:00:00
# Update on 2024-01-15 00:00:00
# Update on 2024-01-16 00:00:00
# Update on 2024-01-22 00:00:00
# Update on 2024-01-23 00:00:00
# Update on 2024-01-26 00:00:00
# Update on 2024-01-29 00:00:00
# Update on 2024-01-30 00:00:00
# Update on 2024-01-31 00:00:00
# Update on 2024-02-05 00:00:00
# Update on 2024-02-06 00:00:00
# Update on 2024-02-06 00:00:00
# Update on 2024-02-07 00:00:00
# Update on 2024-02-07 00:00:00
# Update on 2024-02-08 00:00:00
# Update on 2024-02-09 00:00:00
# Update on 2024-02-09 00:00:00
# Update on 2024-02-12 00:00:00
# Update on 2024-02-13 00:00:00
# Update on 2024-02-15 00:00:00
# Update on 2024-02-20 00:00:00
# Update on 2024-02-21 00:00:00
# Update on 2024-02-22 00:00:00
# Update on 2024-02-26 00:00:00
# Update on 2024-02-29 00:00:00
# Update on 2024-03-04 00:00:00
# Update on 2024-03-07 00:00:00
# Update on 2024-03-07 00:00:00
# Update on 2024-03-08 00:00:00
# Update on 2024-03-08 00:00:00
# Update on 2024-03-11 00:00:00
# Update on 2024-03-12 00:00:00
# Update on 2024-03-14 00:00:00
# Update on 2024-03-20 00:00:00
# Update on 2024-03-22 00:00:00
# Update on 2024-03-25 00:00:00
# Update on 2024-03-26 00:00:00
# Update on 2024-03-26 00:00:00
# Update on 2024-03-27 00:00:00
# Update on 2024-03-28 00:00:00
# Update on 2024-03-29 00:00:00
# Update on 2024-04-01 00:00:00
# Update on 2024-04-03 00:00:00
# Update on 2024-04-09 00:00:00
# Update on 2024-04-09 00:00:00
# Update on 2024-04-10 00:00:00
# Update on 2024-04-11 00:00:00
# Update on 2024-04-11 00:00:00
# Update on 2024-04-15 00:00:00
# Update on 2024-04-17 00:00:00
# Update on 2024-04-17 00:00:00
# Update on 2024-04-18 00:00:00
# Update on 2024-04-19 00:00:00
# Update on 2024-04-22 00:00:00
# Update on 2024-04-25 00:00:00
# Update on 2024-04-29 00:00:00
# Update on 2024-04-29 00:00:00
# Update on 2024-04-30 00:00:00
# Update on 2024-04-30 00:00:00
# Update on 2024-05-02 00:00:00
# Update on 2024-05-03 00:00:00
# Update on 2024-05-03 00:00:00
# Update on 2024-05-06 00:00:00
# Update on 2024-05-07 00:00:00
# Update on 2024-05-08 00:00:00
# Update on 2024-05-09 00:00:00
# Update on 2024-05-09 00:00:00
# Update on 2024-05-14 00:00:00
# Update on 2024-05-16 00:00:00
# Update on 2024-05-16 00:00:00
# Update on 2024-05-17 00:00:00
# Update on 2024-05-17 00:00:00
# Update on 2024-05-21 00:00:00
# Update on 2024-05-22 00:00:00
# Update on 2024-05-23 00:00:00
# Update on 2024-05-23 00:00:00
# Update on 2024-05-24 00:00:00
# Update on 2024-05-24 00:00:00
# Update on 2024-05-27 00:00:00
# Update on 2024-05-29 00:00:00
# Update on 2024-05-31 00:00:00
# Update on 2024-06-03 00:00:00
# Update on 2024-06-04 00:00:00
# Update on 2024-06-05 00:00:00
# Update on 2024-06-05 00:00:00
# Update on 2024-06-05 00:00:00
# Update on 2024-06-06 00:00:00
# Update on 2024-06-11 00:00:00
# Update on 2024-06-12 00:00:00
# Update on 2024-06-12 00:00:00
# Update on 2024-06-13 00:00:00
# Update on 2024-06-14 00:00:00
# Update on 2024-06-17 00:00:00
# Update on 2024-06-19 00:00:00
# Update on 2024-06-20 00:00:00
# Update on 2024-06-21 00:00:00
# Update on 2024-06-26 00:00:00
# Update on 2024-06-26 00:00:00
# Update on 2024-06-28 00:00:00
# Update on 2024-07-08 00:00:00
# Update on 2024-07-08 00:00:00
# Update on 2024-07-09 00:00:00
# Update on 2024-07-11 00:00:00
# Update on 2024-07-12 00:00:00
# Update on 2024-07-15 00:00:00
# Update on 2024-07-15 00:00:00
# Update on 2024-07-16 00:00:00
# Update on 2024-07-17 00:00:00
# Update on 2024-07-17 00:00:00
# Update on 2024-07-22 00:00:00
# Update on 2024-07-23 00:00:00
# Update on 2024-07-23 00:00:00
# Update on 2024-07-25 00:00:00
# Update on 2024-07-29 00:00:00
# Update on 2024-07-30 00:00:00
# Update on 2024-07-30 00:00:00
# Update on 2024-08-01 00:00:00
# Update on 2024-08-01 00:00:00