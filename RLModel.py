import numpy as np
import pandas as pd
from mimic import load_mimic
from collections import deque

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
    return tuple(state)

# Define action space
action_space = ['increase_oxygen', 'decrease_oxygen', 'increase_medication', 'decrease_medication']

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

# Define Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Exploration decay rate

# Initialize Q-table
state_space_size = [len(bins) for bins in state_bins.values()]
q_table = np.zeros(state_space_size + [len(action_space)])

# Q-learning training loop
episodes = 20000
for episode in range(episodes):
    patient_idx = np.random.choice(patient_vitals.index)
    state = get_state(patient_vitals.iloc[patient_idx])
    done = False
    episode_reward = 0

    while not done:
        # Exploration or exploitation
        if np.random.rand() < epsilon:
            action = np.random.choice(action_space)  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        # Take action and observe next state and reward
        next_patient_idx = np.random.choice(patient_vitals.index)
        next_state = get_state(patient_vitals.iloc[next_patient_idx])
        reward = get_reward(next_state)
        episode_reward += reward

        # Update Q-table
        q_table[state + (action,)] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state + (action,)])
        state = next_state

        # Check if episode is done
        if reward == len(state_bins):
            done = True

    # Decay exploration rate
    epsilon *= epsilon_decay

    # Print episode statistics (optional)
    print(f"Episode: {episode}, Reward: {episode_reward}")

# Use the learned Q-table for prognosis and treatment recommendations
def get_treatment_recommendation(patient_state):
    state = get_state(patient_state)
    action = np.argmax(q_table[state])
    return action_space[action]