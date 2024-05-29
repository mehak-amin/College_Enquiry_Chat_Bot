import torch.nn as nn

# DEFINE A CUSTOM NEURAL NETWORK CLASS
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        
        # DEFINE THE LAYERS OF THE NEURAL NETWORK
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()  # RECTIFIED LINEAR UNIT ACTIVATION FUNCTION
    
    def forward(self, x):
        # FORWARD PROPAGATION THROUGH THE NETWORK
        out = self.l1(x)      # FIRST LINEAR LAYER
        out = self.relu(out)  # APPLY RELU ACTIVATION
        out = self.l2(out)    # SECOND LINEAR LAYER
        out = self.relu(out)  # APPLY RELU ACTIVATION
        out = self.l3(out)    # THIRD LINEAR LAYER
        # NO ACTIVATION AND NO SOFTMAX AT THE END
        return out