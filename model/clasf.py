import torch.nn as nn

# Define the neural network architecture
class ClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, drop_rate):
        super(ClassificationModel, self).__init__()
        
        self.projection_net = nn.Sequential(
            nn.Linear(input_dim, 128),

            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(128, hidden_dim),
        )

        self.classification_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, num_classes)
        )


    def forward(self, x):
        hidden_state = self.projection_net(x)

        output = self.classification_layer(hidden_state)
        
        return output, hidden_state  # Return both output and hidden state
