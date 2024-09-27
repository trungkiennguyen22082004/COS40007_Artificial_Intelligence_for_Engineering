import pandas as pd

# Load the dataset
data_path = 'Preprocessed_Data.csv'
data = pd.read_csv(data_path)

# Determine percentiles
threshold_low = data['Total Traffic (GB)'].quantile(0.50)
threshold_high = data['Total Traffic (GB)'].quantile(0.75)

# Function to apply labeling
def label_traffic(row):
    if row['Total Traffic (GB)'] > threshold_high:
        return 'High Traffic'
    elif row['Total Traffic (GB)'] > threshold_low:
        return 'Medium Traffic'
    else:
        return 'Low Traffic'

# Apply the labeling
data['Traffic Label'] = data.apply(label_traffic, axis=1)

# Display updated data
print(data[['Total Traffic (GB)', 'Traffic Label']].head())

# Save the labeled data
data.to_csv('Labeled_Data.csv', index=False)
