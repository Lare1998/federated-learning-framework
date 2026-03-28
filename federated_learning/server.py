import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict, Tuple

class FederatedServer:
    """
    Represents the central server in a federated learning setup.
    It aggregates model updates from clients and maintains the global model.
    """
    def __init__(self, global_model: nn.Module, num_clients: int):
        """
        Initializes the FederatedServer.

        Args:
            global_model (nn.Module): The initial global model architecture.
            num_clients (int): The total number of clients participating in federated learning.
        """
        self.global_model = global_model
        self.num_clients = num_clients
        self.global_model_weights = self._get_model_weights(self.global_model)
        print(f"[Server] Initialized with {num_clients} clients. Global model ready.")

    def _get_model_weights(self, model: nn.Module) -> OrderedDict:
        """
        Extracts the state dictionary (weights) from a PyTorch model.

        Args:
            model (nn.Module): The PyTorch model.

        Returns:
            OrderedDict: The state dictionary of the model.
        """
        return model.state_dict()

    def _set_model_weights(self, model: nn.Module, weights: OrderedDict):
        """
        Loads the provided weights into a PyTorch model.

        Args:
            model (nn.Module): The PyTorch model to load weights into.
            weights (OrderedDict): The state dictionary to load.
        """
        model.load_state_dict(weights)

    def aggregate_models(self, client_weights_list: List[OrderedDict]) -> OrderedDict:
        """
        Aggregates model weights received from multiple clients using Federated Averaging (FedAvg).

        Args:
            client_weights_list (List[OrderedDict]): A list of state dictionaries (weights) from participating clients.

        Returns:
            OrderedDict: The aggregated global model weights.
        """
        if not client_weights_list:
            print("[Server] No client weights received for aggregation.")
            return self.global_model_weights

        print(f"[Server] Aggregating updates from {len(client_weights_list)} clients.")

        # Initialize aggregated weights with zeros or the first client's weights
        aggregated_weights = OrderedDict()
        for key in self.global_model_weights.keys():
            aggregated_weights[key] = torch.zeros_like(self.global_model_weights[key])

        # Sum up all client weights
        for client_weights in client_weights_list:
            for key in aggregated_weights.keys():
                aggregated_weights[key] += client_weights[key]

        # Average the weights
        for key in aggregated_weights.keys():
            aggregated_weights[key] /= len(client_weights_list)

        # Update the global model with aggregated weights
        self._set_model_weights(self.global_model, aggregated_weights)
        self.global_model_weights = aggregated_weights
        print("[Server] Aggregation complete. Global model updated.")
        return self.global_model_weights

    def get_global_model_weights(self) -> OrderedDict:
        """
        Returns the current global model weights.

        Returns:
            OrderedDict: The state dictionary of the global model.
        """
        return self.global_model_weights

    def evaluate_global_model(self, data_loader: torch.utils.data.DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Evaluates the global model on a given dataset.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
            criterion (nn.Module): Loss function for evaluation.

        Returns:
            Tuple[float, float]: A tuple containing (average_loss, accuracy).
        """
        self.global_model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self.global_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_samples
        print(f"[Server] Global model evaluation: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        return avg_loss, accuracy

if __name__ == "__main__":
    # Dummy model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    # Initialize a global model
    initial_model = SimpleModel()
    server = FederatedServer(initial_model, num_clients=3)

    # Simulate client updates
    client_updates = []
    for i in range(3):
        client_model = SimpleModel()
        # Simulate some training by slightly changing weights
        for param in client_model.parameters():
            param.data += torch.randn_like(param.data) * 0.1
        client_updates.append(server._get_model_weights(client_model))

    # Aggregate client updates
    aggregated_weights = server.aggregate_models(client_updates)

    # Verify global model weights are updated
    print("\nGlobal model weights after aggregation:")
    for name, param in server.global_model.named_parameters():
        print(f"{name}: {param.data.mean():.4f}")

    # Simulate evaluation data
    dummy_data = torch.randn(100, 10)
    dummy_labels = torch.randint(0, 2, (100,))
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=10)

    # Evaluate the global model
    criterion = nn.CrossEntropyLoss()
    server.evaluate_global_model(dummy_dataloader, criterion)
