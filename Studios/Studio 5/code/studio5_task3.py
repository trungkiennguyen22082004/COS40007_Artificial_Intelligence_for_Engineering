import os
import json
import shutil

# Define the path to the image and annotation folders
annotation_dir = './log-labelled/test'
updated_dir = './updated_log_labels'

# Create a new directory to save updated labels
if not os.path.exists(updated_dir):
    os.makedirs(updated_dir)

def update_labels(json_file, save_path):
    """
    Update the labels in the JSON file: Replace 'log' with 'detected_log' if a custom condition is met.
    
    Args:
    - json_file (str): The path to the JSON file to update.
    - save_path (str): The path where the updated JSON file will be saved.
    """
    # Open the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    # There is another JSON file: test_annotations.json that we have created in task 2, skip this 
    if 'shapes' not in data:
        print(f"'shapes' key not found in {json_file}, skipping...")
        return  # Skip this file if 'shapes' key is missing
    
    # Update each shape label in the JSON file
    for shape in data['shapes']:
        if shape['label'] == 'log':
            # For now,  logs with small bounding boxes as broken
            points = shape['points']
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            # Example condition: if the width or height of the log is below a threshold, mark it as 'detected_log'
            if width < 100 or height < 100:
                shape['label'] = 'detected_log'
    
    # Save the updated JSON file to the new directory
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
        
        
def process_json_files(annotation_dir, updated_dir):
    """
    Process all JSON files in the given directory to update log labels.
    
    Args:
    - annotation_dir (str): Directory containing the original JSON annotation files.
    - updated_dir (str): Directory to save updated JSON files.
    """
    # Create a new directory to save updated labels if it doesn't exist
    if not os.path.exists(updated_dir):
        os.makedirs(updated_dir)

    # Loop through all JSON files in the annotation directory
    for filename in os.listdir(annotation_dir):
        if filename.endswith('.json'):
            json_file = os.path.join(annotation_dir, filename)
            save_path = os.path.join(updated_dir, filename)
            
            # Update the labels in the JSON file
            update_labels(json_file, save_path)
            
            print(f"Updated labels in {filename} and saved to {updated_dir}")

# Main method
def main():
    print("\n==========================================================================")
    print("TASK 3 - EXTEND LOG LABELLING TO ANOTHER CLASS")
    
    # Define paths for original and updated label directories
    annotation_dir = './log-labelled/test'  # Original directory with JSON files
    updated_dir = './updated_log_labels'    # Directory where updated files will be saved

    # Call the process_json_files function to update labels
    process_json_files(annotation_dir, updated_dir)

if __name__ == "__main__":
    main()

