import pandas as pd
import json

mapping = {
    'pid': 'Unique patient identifier.',
    'Household_Id': "Unique identifier for the patient's household.",
    'agedays': 'Age of the child in days.',
    'Acute diarrheal episode (<7 days)': 'Number of acute diarrheal episodes lasting less than 7 days.',
    'Bacille Calmette-Guerin (BCG) vaccine dose number': 'Number of BCG vaccine doses received.',
    'Decreased appetite at episode (days)': 'Duration of decreased appetite during an episode, measured in days.',
    'Diarrheal episode duration (days)': 'Duration of a diarrheal episode in days.',
    'Diphtheria, pertussis, and tetanus (DPT) vaccine dose number': 'Number of DPT vaccine doses received.',
    'Fever at episode, max temp (C)': 'Maximum recorded body temperature during a fever episode, in Celsius.',
    'Height (cm)': "Patient's height measured in centimeters.",
    'Inactivated poliovirus vaccine (IPV) dose number': 'Number of Inactivated Poliovirus Vaccine doses received.',
    'Mean pathogen (excluding NRV) count in 1st diarrheal stools': 'Average pathogen count (excluding Non-Rotavirus) in the first diarrheal stool sample.',
    'Mean pathogen (excluding NRV) count in 1st monthly stools': 'Average pathogen count (excluding Non-Rotavirus) in the first monthly routine stool sample.',
    'Mean pathogen (excluding NRV) count in stools': 'Average pathogen count (excluding Non-Rotavirus) across all stool samples.',
    'Mean virus (excluding NRV) count in 1st diarrheal stools': 'Average viral load (excluding Non-Rotavirus) in the first diarrheal stool sample.',
    'Measles, mumps, and rubella (MMR) vaccine dose number': 'Number of MMR vaccine doses received.',
    'Oral poliovirus vaccine (OPV) dose number': 'Number of Oral Poliovirus Vaccine doses received.',
    'Other liquids (soup, broth, sugar water, carbonated drinks)': 'Frequency/amount of other liquids consumed by the child.',
    'Persistent diarrheal episode (>14 days)': 'Number of persistent diarrheal episodes lasting more than 14 days.',
    'Pneumococcal conjugate vaccine (PCV) dose number': 'Number of PCV doses received.',
    'Prolonged diarrheal episode (7 to 14 days)': 'Number of prolonged diarrheal episodes lasting between 7 and 14 days.',
    'Time between diarrheal episodes (days)': 'Duration in days between consecutive diarrheal episodes.',
    'Time since last diarrheal episode (days)': 'Duration in days since the most recent diarrheal episode.',
    'Vomiting at episode (days)': 'Duration of vomiting during an illness episode, in days.',
    'Weight (kg)': "Patient's body weight measured in kilograms.",
    'Age 1st animal milk or formula given (days)': 'Age of the child when first introduced to animal milk or formula, in days.',
    'Age 1st animal milk or solids given (days)': 'Age of the child when first introduced to animal milk or solid foods, in days.',
    'Age 1st clear liquid given (days)': 'Age of the child when first introduced to clear liquids, in days.',
    'Age 1st solid food given (days)': 'Age of the child when first introduced to solid foods, in days.',
    'Age at 1st day of biweekly breastfeeding surveillance (days)': 'Age of the child at the initiation of biweekly breastfeeding monitoring, in days.',
    'Age at complete weaning (days)': 'Age of the child when completely weaned from breast milk, in days.',
    'Age at last illness surveillance (days)': 'Age of the child at the conclusion of illness monitoring, in days.',
    'Age last exclusively breastfed (days)': 'Age of the child when exclusive breastfeeding ended, in days.',
    'Birth weight (kg)': 'Weight of the child at birth, measured in kilograms.',
    'Sex': 'Biological sex of the patient.',
    'Household Food Insecurity Access Scale (HFIAS)': 'Measure of household food insecurity based on the HFIAS framework.',
    'Household Food Insecurity Access Scale (HFIAS) score': 'Quantitative score representing the severity of household food insecurity.',
    'Maternal education (years)': "Number of years of formal education completed by the patient's mother."
}

df = pd.read_csv('mal_ed_data/enriched_codebook.csv')
for i, row in df.iterrows():
    if row['Feature'] in mapping and row['Definition'] == 'Study variable.':
        df.at[i, 'Definition'] = mapping[row['Feature']]

df.to_csv('mal_ed_data/enriched_codebook.csv', index=False)
print("Updated enriched codebook.")
