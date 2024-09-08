# DEVELOP CNN AND RESNET50

    #   1. DATA PREPARATION

import numpy as np

import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_data():
    # Define the dataset path
    data_dir = './Corrosion'  # Update this to the path where your images are stored
    classes = ['rust', 'no rust']

    # Create train and test directories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
        # Get all images in class directory
        src_dir = os.path.join(data_dir, cls)
        images = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        
        # Split data
        train_imgs, test_imgs = train_test_split(images, test_size=10, random_state=1)  # Ensures 10 images per class go to test
        
        # Copy images to respective directories
        for img in train_imgs:
            shutil.copy(img, os.path.join(train_dir, cls))
        for img in test_imgs:
            shutil.copy(img, os.path.join(test_dir, cls))
            
    # Check the saved train/test directories
    def count_images_in_directory(directory):
        """Count files in subdirectories of the given directory."""
        categories = os.listdir(directory)
        count_dict = {}
        for category in categories:
            path = os.path.join(directory, category)
            count = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
            count_dict[category] = count
        return count_dict

    # Paths to the train and test directories
    train_dir = './Corrosion/train' 
    test_dir = './Corrosion/test'

    # Count images
    train_counts = count_images_in_directory(train_dir)
    test_counts = count_images_in_directory(test_dir)

    print("\n--------------------------------------------------------------------------")
    print("Training data:")
    for category, count in train_counts.items():
        print(f"Number of '{category}' images: {count}")

    print("\n--------------------------------------------------------------------------")
    print("\nTesting data:")
    for category, count in test_counts.items():
        print(f"Number of '{category}' images: {count}")
        
    return train_dir, test_dir

#   2. CNN MODEL

# Disable oneDNN Optimization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def simple_cnn(train_dir, test_dir):
    # Define model parameters - Image dimensions
    img_width, img_height = 150, 150

    # Initialize the model
    model = Sequential([
        Input(shape=(img_width, img_height, 3)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Setup data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=20,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=20,
            class_mode='categorical')

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=100,  # Adjust based on actual sample size
        epochs=10,
        validation_data=validation_generator,
        validation_steps=50)  # Adjust based on actual sample size

    # Evaluate the model
    accuracy = model.evaluate(validation_generator)
    print(f'Test accuracy: {accuracy[1] * 100:.2f}%')


#   3. RESNET50

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def resnet50(train_dir, test_dir)
    # Define model input dimensions
    img_width, img_height = 224, 224

    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    # Adding custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model_resnet = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Setup data generators with resizing and rescaling
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical')

    # Train the model
    model_resnet.fit(train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator, validation_steps=50)

    # Evaluate the model
    accuracy_resnet = model_resnet.evaluate(validation_generator)
    print(f'ResNet50 Test accuracy: {accuracy_resnet[1] * 100:.2f}%')

def main():
    print("\n==========================================================================")
    print("TASK 1 - DEVELOP CNN AND RESNET50")

    print("\n--------------------------------------------------------------------------")
    print("     1. DATA PREPARATION")
    
    train_dir, test_dir = prepare_data()
    
    print("\n--------------------------------------------------------------------------")
    print("     2. CNN MODEL")
    simple_cnn(train_dir, test_dir)
    
    print("\n--------------------------------------------------------------------------")
    print("     3. RESNET50")
    resnet50(train_dir, test_dir)
    
        
if __name__ == '__main__':
    main()

# [Removed. See studio5_task_2.py for Task 2]

# DEVELOP MASK RCNN FOR DETECTING LOG
# print("\n==========================================================================")
# print("TASK 2 - DEVELOP MASK RCNN FOR DETECTING LOG")

# #   1. DATA PREPARATION
# print("\n--------------------------------------------------------------------------")
# print("     1. DATA PREPARATION")


# # Define paths
# data_dir = './log-labelled'

# # Create a list of all image files
# image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]

# # Create a corresponding list of JSON files
# json_files = [f.replace('.png', '.json') for f in image_files]

# # Split the dataset into training and testing sets
# train_images, test_images, train_jsons, test_jsons = train_test_split(image_files, json_files, test_size=10, random_state=1)

# # Create directories for the training and testing datasets if not already present
# train_dir = os.path.join(data_dir, 'train')
# test_dir = os.path.join(data_dir, 'test')
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)

# # Copy the testing images and JSON files to the test directory
# for img, json_file in zip(test_images, test_jsons):
#     shutil.copy(img, test_dir)
#     shutil.copy(json_file, test_dir)
    
# # Copy remaining training images and JSON files to the train directory
# for img, json_file in zip(train_images, train_jsons):
#     shutil.copy(img, train_dir)  # Copy train images
#     shutil.copy(os.path.join(data_dir, os.path.basename(json_file)), train_dir) 

# # Output results
# print("Training images and JSONs:")
# for img, json_file in zip(train_images, train_jsons):
#     print(os.path.basename(img), os.path.basename(json_file))

# print("\nTesting images and JSONs moved to test directory:")
# for img, json_file in zip(test_images, test_jsons):
#     print(os.path.basename(img), os.path.basename(json_file))
    
    
# #   2. MODEL SETUP AND TRAINING
# print("\n--------------------------------------------------------------------------")
# print("     2. MODEL SETUP AND TRAINING")

# import json
# import cv2

# # Convert the prepared Dataset to YOLOv5 Format
# def convert_labelme_to_yolo(data_dir, label_dir, class_name="log"):
#     """
#     Convert labelme JSON annotations to YOLO format (txt files), and organize png images into 'images' subdirectory.
    
#     Args:
#     - data_dir (str): Directory containing PNG images and JSON files.
#     - label_dir (str): Directory where YOLO txt files will be saved.
#     - class_name (str): Class name for your objects (e.g., "log").
#     """
#     if not os.path.exists(label_dir):
#         os.makedirs(label_dir)

#     # Create 'images' subdirectory inside data_dir if it doesn't exist
#     images_dir = os.path.join(data_dir, 'images')
#     if not os.path.exists(images_dir):
#         os.makedirs(images_dir)

#     class_id = 0  # YOLO expects classes starting from 0
#     json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

#     for json_file in json_files:
#         json_path = os.path.join(data_dir, json_file)
#         img_name = json_file.replace('.json', '.png')
#         img_path = os.path.join(data_dir, img_name)

#         # Copy the image to 'images' subdirectory
#         new_img_path = os.path.join(images_dir, img_name)
#         if os.path.exists(img_path):
#             shutil.copy(img_path, new_img_path)
#         else:
#             print(f"Image file {img_name} not found, skipping.")

#         # Open the image to get its dimensions
#         if not os.path.exists(new_img_path):
#             continue

#         with open(json_path, 'r') as f:
#             data = json.load(f)

#         label_file = os.path.join(label_dir, json_file.replace('.json', '.txt'))
#         with open(label_file, 'w') as f:
#             for shape in data['shapes']:
#                 if shape['label'] == class_name:
#                     # Get the bounding box or points
#                     points = shape['points']
#                     x_coords = [p[0] for p in points]
#                     y_coords = [p[1] for p in points]

#                     # Calculate YOLO format (x_center, y_center, width, height)
#                     x_min = min(x_coords)
#                     x_max = max(x_coords)
#                     y_min = min(y_coords)
#                     y_max = max(y_coords)

#                     x_center = (x_min + x_max) / 2 / data['imageWidth']
#                     y_center = (y_min + y_max) / 2 / data['imageHeight']
#                     bbox_width = (x_max - x_min) / data['imageWidth']
#                     bbox_height = (y_max - y_min) / data['imageHeight']

#                     # Write to YOLO format (class_id, x_center, y_center, width, height)
#                     f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

#         print(f"Converted {json_file} to YOLO format and moved image to {images_dir}.")

# # Directories for your data
# train_json_dir = './log-labelled/train'
# test_json_dir = './log-labelled/test'
# train_label_dir = './log-labelled/train/labels'
# test_label_dir = './log-labelled/test/labels'

# Convert train and test annotations
# convert_labelme_to_yolo(train_json_dir, train_label_dir)
# convert_labelme_to_yolo(test_json_dir, test_label_dir)


# from ultralytics import YOLO


# Train with YOLO 

# #   Initialize YOLOv5 model
# model = YOLO('yolov8n.yaml')  # Specify the architecture YAML file (YOLOv8n)

# #   Train the model on your custom dataset from scratch
# model.train(data='./data.yaml', epochs=50, imgsz=1024, batch=16)

# # Evaluate the trained model
# metrics = model.val(data='./data.yaml', imgsz=1024)
# print(metrics)

#   2. MODEL TESTING
# print("\n--------------------------------------------------------------------------")
# print("     2. MODEL TESTING")

# model = YOLO('./runs/detect/train4/weights/best.pt')  # Load the best model after training

# # Run inference on the test images
# test_images_dir = './log-labelled/test/images'
# results = model.predict(source=test_images_dir, imgsz=1024, conf=0.25)

# output_dir = './log-labelled/test/results'
# # Create a directory to save the output images if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Count the number of logs detected in each image and draw bounding boxes
# for i, result in enumerate(results):
#     image_path = result.path
#     img = cv2.imread(image_path)

#     log_count = 0  # Initialize log count for this image

#     # Extract bounding box information and confidence scores
#     for box in result.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#         conf = box.conf[0].item()  # Confidence score
#         label = "log"  # Class label (since we only have one class: "log")
        
#         # Increment log count for each detected log
#         log_count += 1

#         # Draw the bounding box
#         color = (0, 255, 0)  # Green color for the bounding box
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

#         # Draw the label and confidence score
#         text = f"{label} {conf:.2f}"
#         cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     # Save the resulting image with bounding boxes
#     output_image_path = os.path.join(output_dir, f"result_{i}.png")
#     cv2.imwrite(output_image_path, img)

#     # Print log count for this image
#     print(f"Image {image_path}: Detected {log_count} logs")

