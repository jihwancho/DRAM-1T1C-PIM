import torch
import torch.nn as nn
from .pim import PIM_function

# Hardware Constraints defined by the PIM array architecture
NUM_CYCLE = 50
NUM_ROWS = 96
NUM_COLS = 128

# Example PIM Layer 1
# This layer assumes input and weight match the dimensions of PIM_function (Input: 50x96, Weight: 128x96)
# For larger layers, split the input and weight into tiles that fit the PIM array
class PIM_layer(nn.Module):
    def __init__(self, in_features, out_features, mode='sim'):
        super(PIM_layer, self).__init__()
        self.mode = mode
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, input):
        # Weight Clamping: Keep latent weights between -1 and 1
        if self.training:
            self.weight.data.clamp_(-1, 1)
        
        output = PIM_function.apply(input, self.weight, self.mode)
        return output

# Example PIM Layer 2
# This layer implements tiling to handle larger input and weight dimensions by splitting them into smaller tiles that fit the PIM array
# and then merging the outputs of the tiles
class Tiled_PIM_layer(nn.Module):
    def __init__(self, tile_in, tile_out, tile_rows, tile_cols, mode='sim'):
        super(Tiled_PIM_layer, self).__init__()
        self.tile_in = tile_in
        self.tile_out = tile_out
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.mode = mode
        self.tiles = nn.ParameterList([
            nn.Parameter(torch.randn(tile_out, tile_in)*0.1)
            for _ in range(self.tile_rows * self.tile_cols)
        ])

    def forward(self, x):
        # Weight Clamping: Keep latent weights between -1 and 1
        if self.training:
            for tile in self.tiles:
                tile.data.clamp_(-1, 1)

        x = x.view(-1, self.tile_rows, self.tile_in) # Reshape input to (Batch, tile_rows, tile_in)
        outputs = [] # List of output of each group (row)
        for i in range(self.tile_rows):
            tile_outputs = []
            for j in range(self.tile_cols):
                tile_weight = self.tiles[i * self.tile_cols + j]
                tile_output = PIM_function.apply(x[:, i, :], tile_weight, self.mode) # (Batch, tile_out)
                tile_outputs.append(tile_output) # List of outputs of tiles in the same group
            group_output = torch.cat(tile_outputs, dim=1) # Concatenate outputs of tiles in the same group (Batch, tile_cols * tile_out)
            outputs.append(group_output) # List of outputs of each group -> tile_rows * (Batch, tile_cols * tile_out)
        final_output = torch.stack(outputs, dim=1) # Stack outputs of all groups (Batch, tile_rows, tile_cols * tile_out)
        merged_output = final_output.sum(dim=1) # Sum outputs of all groups (Batch, tile_cols * tile_out)
        return merged_output

# Example MLP model 1, using the PIM layer for MNIST classification
class PIM_MNIST_MLP(nn.Module):
    def __init__(self):
        super(PIM_MNIST_MLP, self).__init__()
        
        # Flatten 28x28 image to 784
        self.flatten = nn.Flatten()
        
        # First Layer
        self.fc1 = nn.Linear(784, NUM_ROWS)
        
        # Second Layer (PIM)
        self.fc2 = PIM_layer(NUM_ROWS, NUM_COLS, mode='sim')
        
        # Final Output Layer
        self.fc3 = nn.Linear(NUM_COLS, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)  
        x = self.fc2(x)  
        x = self.fc3(x)
        return x
    
# Example MLP model 2, using the Tiled PIM layer for MNIST classification
class TILED_PIM_MNIST_MLP(nn.Module):
    def __init__(self):
        super(TILED_PIM_MNIST_MLP, self).__init__()
        
        # Flatten 28x28 image to 784
        self.flatten = nn.Flatten()
        
        # First Layer
        self.fc1 = nn.Linear(784, 20*NUM_ROWS) # 20 groups of 96 input features each
        self.htanh1 = nn.Hardtanh(min_val=-30, max_val=30)
        
        # Second Layer (PIM)
        self.fc2 = Tiled_PIM_layer(tile_in=NUM_ROWS, tile_out=NUM_COLS, tile_rows=20, tile_cols=15, mode='sim') # 20 groups of 15 tiles each

        # Third Layer (PIM)
        self.fc3 = Tiled_PIM_layer(tile_in=NUM_ROWS, tile_out=NUM_COLS, tile_rows=20, tile_cols=15, mode='sim') # 20 groups of 15 tiles each
        
        # Final Output Layer
        self.fc4 = nn.Linear(15*NUM_COLS, 10)
        self.bn4 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.htanh1(x) 
        x = self.fc2(x)  
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        return x