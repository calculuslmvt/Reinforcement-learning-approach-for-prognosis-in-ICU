import numpy as np
import pandas as pd
from mimic import load_mimic
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Load the MIMIC-III dataset
mimic_data = load_mimic()

# Extract relevant tables
admissions = mimic_data['admissions.csv']
patients = mimic_data['patients.csv']
icustays = mimic_data['icustays.csv']
chartevents = mimic_data['chartevents.csv']

# Merge relevant tables
patient_admissions = pd.merge(admissions, patients, on='subject_id', how='inner')
patient_icustays = pd.merge(patient_admissions, icustays, on=['subject_id', 'hadm_id'], how='inner')
patient_vitals = pd.merge(patient_icustays, chartevents, on=['subject_id', 'hadm_id', 'stay_id'], how='inner')

# Filter for relevant vital signs
vital_signs = ['heart_rate', 'respiratory_rate', 'oxygen_saturation', 'temperature', 'blood_pressure']
patient_vitals = patient_vitals[patient_vitals['category'].isin(vital_signs)]

# Preprocess data
patient_vitals = preprocess_data(patient_vitals)

# Define state space
state_bins = {
    'heart_rate': np.linspace(30, 200, 10),
    'respiratory_rate': np.linspace(10, 50, 10),
    'oxygen_saturation': np.linspace(70, 100, 10),
    'temperature': np.linspace(35, 41, 10),
    'blood_pressure': np.linspace(60, 180, 10)
}

def get_state(vitals):
    state = []
    for vital, bins in state_bins.items():
        bin_idx = np.digitize(vitals[vital], bins)
        state.append(bin_idx)
    return np.array(state)

# Define action space
action_space = ['increase_oxygen', 'decrease_oxygen', 'increase_medication', 'decrease_medication']
num_actions = len(action_space)

# Define reward function
def get_reward(state):
    # Define desired ranges for vitals
    desired_ranges = {
        'heart_rate': (60, 100),
        'respiratory_rate': (12, 20),
        'oxygen_saturation': (95, 100),
        'temperature': (36.5, 37.5),
        'blood_pressure': (90, 140)
    }
    
    reward = 0
    for i, vital in enumerate(['heart_rate', 'respiratory_rate', 'oxygen_saturation', 'temperature', 'blood_pressure']):
        vital_value = state[i]
        min_value, max_value = desired_ranges[vital]
        if min_value <= vital_value <= max_value:
            reward += 1
    
    return reward

# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Deep Q-learning parameters
gamma = 0.99
batch_size = 32
buffer_size = 10000
update_target_freq = 1000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

# Initialize replay buffer
replay_buffer = deque(maxlen=buffer_size)

# Initialize DQN
state_size = len(state_bins)
dqn = DQN(state_size, num_actions)
target_dqn = DQN(state_size, num_actions)
target_dqn.load_state_dict(dqn.state_dict())
optimizer = optim.Adam(dqn.parameters())
loss_fn = nn.MSELoss()

# Training loop
num_episodes = 10000
for episode in range(num_episodes):
    epsilon = max(eps_end, eps_start * eps_decay**episode)
    patient_idx = np.random.choice(patient_vitals.index)
    state = get_state(patient_vitals.iloc[patient_idx])
    done = False
    episode_reward = 0

    while not done:
        # Exploration or exploitation
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)  # Explore
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = dqn(state_tensor)
            action = torch.argmax(q_values).item()  # Exploit

        # Take action and observe next state and reward
        next_patient_idx = np.random.choice(patient_vitals.index)
        next_state = get_state(patient_vitals.iloc[next_patient_idx])
        reward = get_reward(next_state)
        episode_reward += reward

        # Store transition in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        # Sample a batch from the replay buffer
        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        # Compute Q-values for current state
        q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q-values
        next_q_values = target_dqn(next_states).max(1)[0].detach()
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

        # Compute loss and update DQN
        loss = loss_fn(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target network
        if episode % update_target_freq == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        state = next_state

        # Check if episode is done
        if reward == len(state_bins):
            done = True

    # Print episode statistics (optional)
    print(f"Episode: {episode}, Reward: {episode_reward}")

# Use the learned DQN for prognosis and treatment recommendations
def get_treatment_recommendation(patient_state):
    state = get_state(patient_state)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = dqn(state_tensor)
    action = torch.argmax(q_values).item()
    return action_space[action]