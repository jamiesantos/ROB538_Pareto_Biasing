import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, config_filename, input_size=10, output_size=2):
        """
        Initializes the policy network.

        Parameters:
        - input_size (int): Number of input features.
        - output_size (int): Number of output actions.
        - hidden_layers (list of int): List with the number of neurons in each hidden layer.
        """
        self.config_filename = config_filename
        self._read_config()

        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Build the network layers
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        # Initialize weights and biases with small random values
        self.reset_parameters()
    
    def _read_config(self):
        """Read and load Policy configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)
            # print('[Policy]: YAML config read.')

        self._load_config()
    
    def _load_config(self):
        """Load internal Policy configuration."""
        self.hidden_layers = self.config_data['Policy']['hidden_layers']
        # Store the mutation parameters
        self.mutation_rate  = self.config_data['Policy']['mutation_rate']
        self.mutation_scale = self.config_data['Policy']['mutation_scale']
        self.weight_init_lim = self.config_data['Policy']['weight_init_lim']
        self.bias_init_lim = self.config_data['Policy']['bias_init_lim']

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.uniform_(layer.weight, -self.weight_init_lim, self.weight_init_lim)
            nn.init.uniform_(layer.bias, -self.bias_init_lim, self.bias_init_lim)

    def forward(self, x, final_activation="tanh"):
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)  # Hidden layers activation
            else:
                if final_activation=="tanh":
                    x = torch.tanh(x)  # Output layer activation to constrain outputs to [-1, +1]
                elif final_activation=="softmax":
                    x = torch.softmax(x, dim=0) # Output layer activation to get probability distribution over actions
        return x

    def mutate(self):
        """
        Applies mutations to the network's parameters.

        Parameters:
        - mutation_rate (float): Probability of mutating each parameter.
        - mutation_scale (float): Maximum change for mutations.
        """
        with torch.no_grad():
            for param in self.parameters():
                # Generate a mask for which parameters to mutate
                mutation_mask = torch.rand_like(param) < self.mutation_rate
                # Generate random mutations
                mutations = torch.empty_like(param).uniform_(-self.mutation_scale, self.mutation_scale)
                # Apply mutations
                param.add_(mutation_mask * mutations)
