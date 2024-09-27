import os
import random
import shutil

import pandas as pd

def convert_to_yolo_format(df, image_folder):
    """
    Converts bounding box annotations from pixel-based to YOLO format.
    
    Args:
    - df: A pandas DataFrame containing the bounding box information.
    - image_folder: Directory where the images are stored.
    
    Returns:
    - YOLO formatted annotation DataFrame with numeric class IDs.
    """
    yolo_annotations = []
    
    # Define a class mapping, e.g., 'Graffiti' -> 0
    class_mapping = {'Graffiti': 0}
    
    for index, row in df.iterrows():
        image_width, image_height = row['width'], row['height']
        
        # Compute YOLO coordinates (normalized)
        x_center = (row['xmin'] + row['xmax']) / 2 / image_width
        y_center = (row['ymin'] + row['ymax']) / 2 / image_height
        bbox_width = (row['xmax'] - row['xmin']) / image_width
        bbox_height = (row['ymax'] - row['ymin']) / image_height
        
        # Convert class name to its corresponding index (e.g., 'Graffiti' -> 0)
        class_id = class_mapping.get(row['class'], -1)  # Default to -1 if class not found
        
        if class_id != -1:
            # YOLO format: <class> <x_center> <y_center> <width> <height>
            yolo_annotations.append([row['filename'], class_id, x_center, y_center, bbox_width, bbox_height])
        else:
            print(f"Warning: Class '{row['class']}' not found in class mapping.")
    
    yolo_df = pd.DataFrame(yolo_annotations, columns=['filename', 'class', 'x_center', 'y_center', 'width', 'height'])
    return yolo_df

def save_yolo_annotations(df, output_dir):
    """
    Saves YOLO annotations to individual text files, one for each image.
    
    Args:
    - df: The YOLO formatted pandas DataFrame.
    - output_dir: Directory to save the YOLO annotation files.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in df['filename'].unique():
        # Filter annotations for the current image
        annotations = df[df['filename'] == filename]
        
        # Write annotations to a .txt file
        with open(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt"), 'w') as f:
            for _, row in annotations.iterrows():
                f.write(f"{row['class']} {row['x_center']} {row['y_center']} {row['width']} {row['height']}\n")

# =========================================================================================

# Function to sample 400 random images and their corresponding labels
def sample_data(source_img_dir, source_label_dir, target_img_dir, target_label_dir, num_samples=400):
    """
    Randomly samples a given number of images and their corresponding labels from source to target directories.
    """
    if not os.path.exists(target_img_dir):
        os.makedirs(target_img_dir)
    if not os.path.exists(target_label_dir):
        os.makedirs(target_label_dir)

    # List all image files from source directory
    image_files = os.listdir(source_img_dir)
    
    # Convert only the file extensions to lowercase
    for image_file in image_files:
        name, ext = os.path.splitext(image_file)  # Split the file name and extension
        lower_ext = ext.lower()  # Convert extension to lowercase
        if ext != lower_ext:  # If the extension was not already lowercase
            new_file_name = name + lower_ext
            os.rename(os.path.join(source_img_dir, image_file), os.path.join(source_img_dir, new_file_name))

    # Randomly sample image files
    sampled_images = random.sample(image_files, num_samples)

    # Move the sampled images and corresponding labels
    for image_file in sampled_images:
        # Move image file
        shutil.move(os.path.join(source_img_dir, image_file), os.path.join(target_img_dir, image_file))
        
        # Move corresponding label file (.txt)
        label_file = image_file.replace('.jpg', '.txt').replace('.JPG', '.txt')
        shutil.move(os.path.join(source_label_dir, label_file), os.path.join(target_label_dir, label_file))

    print(f"Moved {num_samples} images and labels from {source_img_dir} to {target_img_dir}.")

# Function to create the data_for_part_2.yaml file
def create_data_yaml(output_path, train_path, val_path, nc=1, names=['Graffiti']):
    """
    Creates the YOLOv5 data.yaml file to define paths for training and validation datasets.
    
    Args:
    - output_path: Path where the .yaml file will be saved.
    - train_path: Path to the directory containing the training images.
    - val_path: Path to the directory containing the validation images.
    - nc: Number of classes.
    - names: List of class names.
    """
    data_yaml_content = f"""train: {train_path}\nval: {val_path}\nnc: {nc}\nnames: {names}"""
    
    with open(output_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"The YAML data file created at {output_path}")
    
        
from ultralytics import YOLO

def train_yolo_model(yaml_file_path='F:/COS40007_Artificial_Intelligence_for_Engineering/Studios/Studio 6/dataset/sampled_0/data_for_part_2.yaml', model=None, epochs=10):
    """
    Trains the YOLOv8 model using the pre-trained model 'yolov8n.pt' and sampled graffiti detection data,
    and saves the fine-tuned model after training.
    """
    # Load the pre-trained YOLOv8n model
    if mode == None:
        model = YOLO('yolov8n.pt')  # Pre-trained model as starting point
    
    
    # Train the model on the custom dataset
    model.train(
        data=yaml_file_path,  # Path to your data.yaml file created earlier
        epochs=epochs,                           # Number of epochs
        imgsz=864,                               # Image size 
        batch=16,                                # Batch size
        name='graffiti_detection',               # Name for the run (saved in runs/train)
        save_period=1                            # Save model after each epoch
    )

    # Save the trained model manually if needed
    model_path = './model.pt'  # Path to the best weights file
    model.save(model_path)  # Save the fine-tuned model

    print(f"Model saved to {model_path}")

    
def compute_iou(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
    - boxA: The first bounding box in (x1, y1, x2, y2) format.
    - boxB: The second bounding box in (x1, y1, x2, y2) format.
    
    Returns:
    - IoU: The Intersection over Union of the two boxes.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the Intersection over Union (IoU)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def convert_yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """
    Converts YOLO normalized coordinates to bounding box (x1, y1, x2, y2) format.
    
    Args:
    - x_center: Normalized x center of the bounding box.
    - y_center: Normalized y center of the bounding box.
    - width: Normalized width of the bounding box.
    - height: Normalized height of the bounding box.
    - img_width: Width of the image in pixels.
    - img_height: Height of the image in pixels.
    
    Returns:
    - Bounding box in (x1, y1, x2, y2) format.
    """
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    
    return [x1, y1, x2, y2]

# This function will take the 40 random images and compute IoU
def process_sampled_images_for_iou(model, sampled_image_dir, sampled_label_dir):
    """
    Runs inference on sampled images, computes IoU values, and generates a CSV file.
    
    Args:
    - model: The trained YOLO model.
    - sampled_image_dir: Directory where sampled test images are stored.
    - sampled_label_dir: Directory where corresponding YOLO label files are stored.
    
    Returns:
    - None
    """
    image_files = [f for f in os.listdir(sampled_image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    iou_data = []
    
    for image_file in image_files:
        image_path = os.path.join(sampled_image_dir, image_file)
        label_path = os.path.join(sampled_label_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Perform inference using the trained model
        results = model.predict(source=image_path, imgsz=864, save=False)
        
        # Load ground truth bounding boxes
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                ground_truths = [line.strip().split() for line in f.readlines()]
                ground_truth_boxes = [
                    convert_yolo_to_bbox(float(gt[1]), float(gt[2]), float(gt[3]), float(gt[4]), 720, 960) for gt in ground_truths
                ]
        else:
            ground_truth_boxes = []
        
        # If no graffiti is detected, IoU is 0
        if len(results[0].boxes) == 0:
            iou_data.append([image_file, 0, 0])
            continue
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Predicted bounding box
            conf = box.conf  # Confidence score
            predicted_box = [x1, y1, x2, y2]
            
            # Compute IoU with ground truth
            if len(ground_truth_boxes) > 0:
                ious = [compute_iou(predicted_box, gt_box) for gt_box in ground_truth_boxes]
                best_iou = max(ious)
            else:
                best_iou = 0
            
            iou_data.append([image_file, conf.item(), best_iou])
            
        # Convert list to DataFrame and save as CSV
    iou_df = pd.DataFrame(iou_data, columns=['image_name', 'confidence_value', 'iou_value'])
    iou_df.to_csv('./dataset/sampled_0/iou_results_sampled.csv', index=False)
    
    # Inform the user that the CSV was created
    print(f"IoU results saved to 'iou_results_sampled.csv'.")

def check_iou_threshold(iou_data_file_path, threshold=0.90, percentage=0.80):
    """
    Checks if the IoU value of a certain percentage of images in the dataset exceeds a given threshold.
    
    Args:
    - data_file_path: Path to the CSV file containing IoU values.
    - threshold: The IoU value threshold to check (default is 0.90).
    - percentage: The percentage of images that should exceed the IoU threshold (default is 80%).

    Returns:
    - bool: True if the required percentage of images exceed the IoU threshold, False otherwise.
    """
    # Read the dataset
    iou_df = pd.read_csv(iou_data_file_path)

    # Check how many images have IoU values greater than the threshold
    valid_iou_count = iou_df[iou_df['iou_value'] > threshold].shape[0]

    # Calculate the total number of images
    total_images = iou_df.shape[0]

    # Calculate the percentage of images that exceed the threshold
    percentage_exceeding_threshold = valid_iou_count / total_images

    # Check if the percentage exceeds the given requirement
    return percentage_exceeding_threshold >= percentage


import cv2

def test_model_and_generate_images(model, test_images_dir, output_dir):
    """
    Tests a YOLO model on test images and generates images with detected bounding boxes.
    
    Args:
    - model: YOLO model.
    - test_images_dir: Directory containing the test images.
    - output_dir: Directory where the outcome images with bounding boxes will be saved.
    
    Returns:
    - None
    """
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all image files in the test directory
    image_files = os.listdir(test_images_dir)

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(test_images_dir, image_file)
        
        # Perform inference using the trained model
        results = model.predict(source=image_path, imgsz=864, save=False)

        # Load the image using OpenCV
        img = cv2.imread(image_path)
        
        # Draw bounding boxes and labels on the image
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Predicted bounding box coordinates
            conf = box.conf  # Confidence score
            label = f'{int(box.cls.item())} {float(conf):.2f}'  # Class ID and confidence score

            # Draw rectangle and label on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label

        # Save the processed image with bounding boxes
        output_image_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_image_path, img)

        print(f"Processed and saved: {output_image_path}")

    print(f"All test images processed and saved to {output_dir}.")
    

def iteratively_train_and_test(initial_model_path, train_folder, test_folder, labels_folder, max_iterations=10):
    """
    Iteratively train and test a YOLO model on sampled data. Stops if IoU requirement is satisfied or if there are not
    enough images left in the train or test folders.
    
    Args:
    - initial_model_path: Path to the initial model file to start training from.
    - train_folder: Path to the original training images folder.
    - test_folder: Path to the original test images folder.
    - labels_folder: Path to the original labels folder.
    - max_iterations: Maximum number of iterations to try (default is 10).
    
    Returns:
    - None
    """
    index = 1  # Start with index 1 for sampled data
    model_path = initial_model_path

    while index <= max_iterations:
        print(f"\n--- Iteration {index} ---")

        # If not the first iteration, check IoU of the previous iteration
        if index > 1:
            prev_sampled_folder = f'./dataset/sampled_{index - 1}'
            prev_iou_file_path = f'{prev_sampled_folder}/iou_results_sampled.csv'

            if os.path.exists(prev_iou_file_path) and check_iou_threshold(prev_iou_file_path):
                print(f"IoU requirement has already satisfied after iteration {index - 1}. Stopping.")
                break

        sampled_folder = f'./dataset/sampled_{index}'

        # Count the number of images in the original train and test folders (assuming all files are images)
        train_images_count = len(os.listdir(train_folder))
        test_images_count = len(os.listdir(test_folder))

        if train_images_count < 400 or test_images_count < 40:
            print(f"Not enough images left in train ({train_images_count}) or test ({test_images_count}) folders. Stopping.")
            break
        
        # Sample new data for training and testing
        print(f"Sampling new data for iteration {index}...")
        sample_data(train_folder, f'{labels_folder}/train', f'{sampled_folder}/images/train', f'{sampled_folder}/labels/train', num_samples=400)
        sample_data(test_folder, f'{labels_folder}/test', f'{sampled_folder}/images/test', f'{sampled_folder}/labels/test', num_samples=40)
        
        # Create data.yaml for the new sample
        create_data_yaml(f'{sampled_folder}/data.yaml', 
                         f'F:/COS40007_Artificial_Intelligence_for_Engineering/Studios/Studio 6/dataset/sampled_{index}/images/train', 
                         f'F:/COS40007_Artificial_Intelligence_for_Engineering/Studios/Studio 6/dataset/sampled_{index}/images/test',
                         nc=1, names=['Graffiti'])
        
        # Train the model using the train_yolo_model method and pass the new sampled YAML path
        print(f"Training the model with data from iteration {index}...")
        model = YOLO(model_path)  # Load the previously trained model
        train_yolo_model(yaml_file_path=f'{sampled_folder}/data.yaml', model=model, epochs=50)

        # Test the model and compute IoU
        print(f"Testing the model and computing IoU for iteration {index}...")
        process_sampled_images_for_iou(model, f'{sampled_folder}/images/test', f'{sampled_folder}/labels/test')
        
        # Save the newly trained model for the next iteration
        model_path = f'./model_trained_{index}.pt'
        model.save(model_path)
        
        # Generate test outcome images
        test_model_and_generate_images(
            model=model,
            test_images_dir=f'{sampled_folder}/images/test',
            output_dir=f'{sampled_folder}/output_images')

        # Move to the next iteration
        index += 1

    if index > max_iterations:
        print(f"Reached maximum iterations ({max_iterations}) without satisfying the IoU requirement.")

    
def detect_graffiti_in_video(model, video_path, output_path=None, frame_width=1920, frame_height=1080):
    """
    Detects graffiti in real-time video data using the trained YOLO model.

    Args:
    - model: Trained YOLO model.
    - video_path: Path to the input video file (mp4 format).
    - output_path: Path to save the processed video with detected bounding boxes (optional).
    - frame_width: Width of the output video
    - frame_heigh: Heigh of the output video
    
    Returns:
    - None
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Define the codec and create a VideoWriter object if you want to save the output video
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for MP4
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))  # Set frame size

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Finished processing the video.")
            break

        # Resize frame to the size expected by the model (optional, based on model configuration)
        img_height, img_width = frame.shape[:2]

        # Perform inference using the trained YOLO model
        results = model.predict(source=frame, imgsz=864, save=False)

        # Draw bounding boxes and labels on the video frame
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf  # Confidence score
            label = f'{int(box.cls.item())} {float(conf):.2f}'  # Class ID and confidence score

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label

        # Show the video frame with detections in a window
        cv2.imshow("Graffiti Detection", frame)

        # Save the frame to the output video file if output_path is provided
        if output_path:
            out.write(frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    print(f"Finished processing the video and saved output to {output_path}" if output_path else "Real-time detection completed.")


# Main function to execute the process
def main():
    
    # Define the paths to the train and test label files
    train_labels_path = './dataset/bounding_boxes/train_labels.csv'
    test_labels_path = './dataset/bounding_boxes/test_labels.csv'

    # Load the train and test labels into pandas DataFrames
    train_labels_df = pd.read_csv(train_labels_path)
    test_labels_df = pd.read_csv(test_labels_path)

    # # Display the first few rows of each DataFrame to confirm successful loading
    # print("Train Labels DataFrame:")
    # print(train_labels_df.head())
    # print("\nTest Labels DataFrame:")
    # print(test_labels_df.head())
    
    # # Convert the training and testing labels to YOLO format and save them in the appropriate directory
    # train_yolo = convert_to_yolo_format(train_labels_df, './dataset/images/train')
    # save_yolo_annotations(train_yolo, './dataset/yolo_labels/train')
    # test_yolo = convert_to_yolo_format(test_labels_df, './dataset/images/test')
    # save_yolo_annotations(test_yolo, './dataset/yolo_labels/test')

    # # Sample 400 random images and their corresponding labels for training into the dataset/sampled_0 directory
    # sample_data('./dataset/images/train', './dataset/yolo_labels/train', './dataset/sampled_0/images/train', './dataset/sampled_0/labels/train', num_samples=400)
    
    # # Sample 40 random images and their corresponding labels for testing into the dataset/sampled_0 directory
    # sample_data('./dataset/images/test', './dataset/yolo_labels/test', './dataset/sampled_0/images/test', './dataset/sampled_0/labels/test', num_samples=40)

    # # Create the data_for_part_2.yaml file for YOLO training AFTER the sampled data is created
    # create_data_yaml('./dataset/sampled_0/data.yaml', 'F:/COS40007_Artificial_Intelligence_for_Engineering/Studios/Studio 6/dataset/sampled_0/images/train', 'F:/COS40007_Artificial_Intelligence_for_Engineering/Studios/Studio 6/dataset/sampled_0/images/test', nc=1, names=['Graffiti'])
    
    # # Train the model
    # train_yolo_model(yaml_file_path='F:/COS40007_Artificial_Intelligence_for_Engineering/Studios/Studio 6/dataset/sampled_0/data.yaml', epochs=50)
    
    # model = YOLO('./model.pt')  # Load the trained model
    # process_sampled_images_for_iou(model, './dataset/sampled_0/images/test', './dataset/sampled_0/labels/test')
    
    # Usage example:
    # iteratively_train_and_test(initial_model_path='./model.pt',
    #                            train_folder='./dataset/images/train',
    #                            test_folder='./dataset/images/test',
    #                            labels_folder='./dataset/yolo_labels')
    
    model = YOLO('./model_trained_1.pt')
    # detect_graffiti_in_video(model, video_path='./dataset/videos/sample_1.mp4', output_path='./dataset/videos/output_detected_for_sample_1.mp4', frame_width=1080, frame_height=1920)
    detect_graffiti_in_video(model, video_path='./dataset/videos/sample_2.mp4', output_path='./dataset/videos/output_detected_for_sample_2.mp4', frame_width=1920, frame_height=1080)
    # detect_graffiti_in_video(model, video_path='./dataset/videos/sample_3.mp4', output_path='./dataset/videos/output_detected_for_sample_3.mp4', frame_width=1920, frame_height=1080)
    detect_graffiti_in_video(model, video_path='./dataset/videos/sample_4.mp4', output_path='./dataset/videos/output_detected_for_sample_4.mp4', frame_width=1920, frame_height=1080)
    detect_graffiti_in_video(model, video_path='./dataset/videos/sample_5.mp4', output_path='./dataset/videos/output_detected_for_sample_5.mp4', frame_width=1920, frame_height=1080)
    
# Execute main function
if __name__ == "__main__":
    main()