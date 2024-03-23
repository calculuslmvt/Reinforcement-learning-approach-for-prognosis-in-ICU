import pandas as pd
from mimic import load_mimic

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

# Preprocess data (e.g., handle missing values, convert units, etc.)
patient_vitals = preprocess_data(patient_vitals)

# Print sample data
print(patient_vitals.head())