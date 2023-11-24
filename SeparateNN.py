import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Actor, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Initialize models
num_inputs = 4  # Stacked frames
num_actions = len(SIMPLE_MOVEMENT)  # Number of possible actions
actor_model = Actor(num_inputs, num_actions)
critic_model = Critic(num_inputs)
