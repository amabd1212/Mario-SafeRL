import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer for feature flattening
        self.fc1 = nn.Linear(7 * 7 * 64, 512)

        # Actor and Critic heads
        self.actor = nn.Linear(512, num_actions)  # Outputs policy probabilities
        self.critic = nn.Linear(512, 1)          # Outputs value function
        print(self.actor,self.critic)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        # Compute actor and critic outputs
        policy = F.softmax(self.actor(x), dim=1)
        value = self.critic(x)
        return policy, value

# Initialize the model
num_inputs = 4  # Stacked frames
num_actions = len(SIMPLE_MOVEMENT)  # Number of possible actions
model = ActorCritic(num_inputs, num_actions)
