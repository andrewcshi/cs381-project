# models.py

import torch
from torch import nn
from config import MODEL

class BaseNetwork(nn.Module):
    """Base class for all network architectures."""
    def __init__(self, robot_dim, obstacle_dim, hidden_dim, num_actions):
        super().__init__()
        self.robot_dim = robot_dim
        self.obstacle_dim = obstacle_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Common MLP head for Q-values
        self.mlp = nn.Sequential(
            nn.Linear(robot_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    
    def encode_obstacles(self, obstacle_state, batch_size):
        """
        Encode obstacle states - to be implemented by subclasses.
        
        Args:
            obstacle_state: Tensor containing obstacle states
            batch_size: Batch size
            
        Returns:
            Encoded representation of obstacles
        """
        raise NotImplementedError
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        size = state.shape
        robot_state = state[:, 0, :self.robot_dim]
        obstacle_state = state[:, :, self.robot_dim:]
        
        # Encode obstacles
        obstacle_encoding = self.encode_obstacles(obstacle_state, size[0])
        
        # Concatenate robot state with obstacle encoding
        joint_state = torch.cat([robot_state, obstacle_encoding], dim=1)
        
        # Generate Q-values
        q_value = self.mlp(joint_state)
        
        return q_value

class LSTMNetwork(BaseNetwork):
    """Network using LSTM to encode obstacle states."""
    def __init__(self, robot_dim, obstacle_dim, lstm_hidden_dim, num_actions):
        super().__init__(robot_dim, obstacle_dim, lstm_hidden_dim, num_actions)
        
        # LSTM for processing obstacle states
        self.lstm = nn.LSTM(obstacle_dim, lstm_hidden_dim, batch_first=True)
    
    def encode_obstacles(self, obstacle_state, batch_size):
        """
        Encode obstacles using LSTM.
        
        Args:
            obstacle_state: Tensor containing obstacle states
            batch_size: Batch size
            
        Returns:
            LSTM encoding of obstacles
        """
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        c0 = torch.zeros(1, batch_size, self.hidden_dim)
        
        _, (hn, _) = self.lstm(obstacle_state, (h0, c0))
        return hn.squeeze(0)

class TransformerNetwork(BaseNetwork):
    """Network using Transformer to encode obstacle states."""
    def __init__(self, robot_dim, obstacle_dim, transformer_hidden_dim, num_actions, 
                nhead=4, num_layers=2, dropout=0.1):
        super().__init__(robot_dim, obstacle_dim, transformer_hidden_dim, num_actions)
        
        # Linear projection to transformer dimension
        self.input_projection = nn.Linear(obstacle_dim, transformer_hidden_dim)
        
        # Transformer encoder for processing obstacle states
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim,
            nhead=nhead,
            dim_feedforward=transformer_hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Learnable position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 20, transformer_hidden_dim))
        
        # For pooling transformer outputs
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def encode_obstacles(self, obstacle_state, batch_size):
        """
        Encode obstacles using Transformer.
        
        Args:
            obstacle_state: Tensor containing obstacle states
            batch_size: Batch size
            
        Returns:
            Transformer encoding of obstacles
        """
        # Project obstacle features to transformer dimension
        x = self.input_projection(obstacle_state)
        
        # Add positional encoding (up to max sequence length of 20)
        seq_len = min(x.size(1), 20)
        x = x[:, :seq_len, :] + self.pos_embedding[:, :seq_len, :]
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(x)
        
        # Global pooling across sequence dimension to get fixed-size representation
        pooled = self.pooling(transformer_output.transpose(1, 2)).squeeze(2)
        
        return pooled