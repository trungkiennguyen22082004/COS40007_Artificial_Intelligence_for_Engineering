import os

# Specify the directory containing the files
directory = './dataset/images/train'

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.startswith("img_"):  # Find files starting with 'img_'
        new_filename = filename.replace("img_", "IMG_", 1)  # Replace 'img_' with 'IMG_'
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))  # Rename the file

print("All files renamed successfully.")
