
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