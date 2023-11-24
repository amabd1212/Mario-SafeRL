import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class PartiallySharedActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PartiallySharedActorCritic, self).__init__()
        
        # Shared convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer for feature flattening
        self.shared_fc = nn.Linear(7 * 7 * 64, 512)

        # Separate fully connected layers for actor and critic
        self.actor_fc = nn.Linear(512, 256)  # Actor-specific layer
        self.critic_fc = nn.Linear(512, 256) # Critic-specific layer

        # Actor and Critic heads
        self.actor = nn.Linear(256, num_actions)  # Outputs policy probabilities
        self.critic = nn.Linear(256, 1)          # Outputs value function

    def forward(self, x):
        # Apply shared convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output for the shared fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.shared_fc(x))

        # Separate processing paths for actor and critic
        actor_x = F.relu(self.actor_fc(x))
        critic_x = F.relu(self.critic_fc(x))

        # Compute actor and critic outputs
        policy = F.softmax(self.actor(actor_x), dim=1)
        value = self.critic(critic_x)

        return policy, value

num_inputs = 4 
num_actions = len( SIMPLE_MOVEMENT) 
model = PartiallySharedActorCritic(num_inputs, num_actions)
