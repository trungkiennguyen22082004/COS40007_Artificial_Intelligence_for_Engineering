import os

from sklearn.model_selection import train_test_split

def split_data():
    # Define paths
    data_dir = './log-labelled'

    # Create a list of all image files
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]

    # Create a corresponding list of JSON files
    json_files = [f.replace('.png', '.json') for f in image_files]

    # Split the dataset into training and testing sets
    train_images, test_images, train_jsons, test_jsons = train_test_split(image_files, json_files, test_size=10, random_state=1)

    # Create directories for the training and testing datasets if not already present
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy the testing images and JSON files to the test directory
    for img, json_file in zip(test_images, test_jsons):
        shutil.copy(img, test_dir)
        shutil.copy(json_file, test_dir)
        
    # Copy remaining training images and JSON files to the train directory
    for img, json_file in zip(train_images, train_jsons):
        shutil.copy(img, train_dir)  # Copy train images
        shutil.copy(os.path.join(data_dir, os.path.basename(json_file)), train_dir) 

    # Output results
    print("Training images and JSONs:")
    for img, json_file in zip(train_images, train_jsons):
        print(os.path.basename(img), os.path.basename(json_file))

    print("\nTesting images and JSONs moved to test directory:")
    for img, json_file in zip(test_images, test_jsons):
        print(os.path.basename(img), os.path.basename(json_file))

import json
import cv2
import shutil

# Copy images to the 'images' subdirectory for both train and test datasets
def setup_image_directories(data_dir, image_subdir):
    """
    Set up a subdirectory to store images and copy images from the root to this subdirectory.
    
    Args:
    - data_dir (str): Directory containing the PNG images and JSON files.
    - image_subdir (str): Subdirectory where PNG images will be moved/copied.
    """
    image_dir = os.path.join(data_dir, image_subdir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # Move PNG files to the 'images' subdirectory
    png_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    for png_file in png_files:
        shutil.copy(os.path.join(data_dir, png_file), os.path.join(image_dir, png_file))
        
        print(f"Images has been copied to {os.path.join(image_dir, png_file)}")

# Convert the prepared Dataset to COCO Format
def generate_coco_annotations(data_dir, image_subdir, output_json_path):
    """
    Generate a COCO-style annotations.json file from individual JSON files.
    
    Args:
    - data_dir (str): Directory containing the JSON files and images.
    - image_subdir (str): Subdirectory where PNG images are stored.
    - output_json_path (str): Path where the annotations.json will be saved.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    image_dir = os.path.join(data_dir, image_subdir)
    
    # Basic structure for COCO dataset
    coco = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [{"id": 1, "name": "log", "supercategory": "none"}]
    }
    
    annotation_id = 1  # Unique ID for each annotation
    
    for i, filename in enumerate(files):
        with open(os.path.join(data_dir, filename)) as f:
            data = json.load(f)
            
         # Get corresponding image file path and load its dimensions
        image_path = os.path.join(data_dir, filename.replace('.json', '.png'))
        height, width = (None, None)
        try:
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
        except:
            print(f"Image file {image_path} not found, skipping.")

        image_info = {
            "file_name": filename.replace('.json', '.png'),
            "height": height, 
            "width": width,
            "id": i + 1
        }
        coco['images'].append(image_info)

        # Process each shape
        for shape in data['shapes']:
            points = shape['points']  # The vertices of the polygon
            
            # Flatten points for segmentation in COCO format
            segmentation = [list(sum(points, []))]
            
            x_min = min(pt[0] for pt in points)
            y_min = min(pt[1] for pt in points)
            x_max = max(pt[0] for pt in points)
            y_max = max(pt[1] for pt in points)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            area = bbox_width * bbox_height  # Calculate area
            
            annotation = {
                "id": annotation_id,
                "image_id": i + 1,
                "category_id": 1,  # Assuming all shapes are 'log'
                "segmentation": segmentation,
                "area": area,  # Calculated area
                "bbox": [x_min, y_min, bbox_width, bbox_height],  # Bounding box
                "iscrowd": 0  # No crowd annotations
            }
            coco['annotations'].append(annotation)
            annotation_id += 1

    # Write the COCO data to a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(coco, f, indent=4)

    print(f"COCO format Annotations has been created at {output_json_path}")

# Train with Torchvision

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from pycocotools.coco import COCO

import numpy as np

# Define the COCODataset class
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        # Load image
        path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to a PyTorch tensor
        img = torch.from_numpy(img).float()  # Convert to float and PyTorch tensor
        img = img.permute(2, 0, 1)  # Change the shape from (H, W, C) to (C, H, W)

        # Load masks
        num_objs = len(coco_annotation)
        height, width = img.shape[1], img.shape[2]  # Get height and width of the image
        masks = np.zeros((height, width, num_objs), dtype=np.uint8)  # Correct mask shape (H, W, num_objs)

        boxes = []
        for i in range(num_objs):
            mask = self.coco.annToMask(coco_annotation[i])  # Get mask for each object
            masks[:, :, i] = mask  # Store the mask in the masks array
            bbox = coco_annotation[i]['bbox']  # Get bounding box
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

        # Convert everything into torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # Assuming all objects are of class 'log'
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

# Define the collate function (instead of using a lambda function)
def collate_fn(batch):
    return tuple(zip(*batch))

def train_model():
    # Load datasets
    train_dataset = COCODataset(root='./log-labelled/train/images', annFile='./log-labelled/train/train_annotations.json')

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Device setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the Mask R-CNN model pre-trained on COCO
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1  # Use the new 'weights' argument
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Modify the model to fit the dataset (1 class: "log" + background)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 2)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels  # This is 256 for ResNet50
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_channels, 256, 2)

    # Move the model to the appropriate device
    model.to(device)

    # Set up the optimizer
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch + 1}, Iteration {i + 1}: Loss = {losses.item()}")
                
    # Save the trained model after training
    save_model_path = './log-labelled/mask_rcnn_model.pth'
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

def test_model():
    
    # Device setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize the Mask R-CNN model pre-trained on COCO
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Modify the model to fit the dataset (1 class: "log" + background)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 2)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels  # This is 256 for ResNet50
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_channels, 256, 2)
    
    # Load the saved model
    load_model_path = './log-labelled/mask_rcnn_model.pth'
    model.load_state_dict(torch.load(load_model_path))
    model.to(device)
    model.eval()
    
    test_dataset = COCODataset(root='./log-labelled/test/images', annFile='./log-labelled/test/test_annotations.json')
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Inference on test data
    model.eval()

    output_dir = './log-labelled/test/results_task_2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize a global image counter
    global_img_idx = 0

    for imgs, _ in test_loader:
        imgs = list(img.to(device) for img in imgs)

        with torch.no_grad():
            predictions = model(imgs)

        # Process the predictions and save images with bounding boxes and masks
        for i, pred in enumerate(predictions):
            img = imgs[i].cpu().numpy().transpose(1, 2, 0)  # Convert from tensor to NumPy array
            masks = pred['masks'].cpu().numpy()
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()

            for j in range(len(masks)):
                if scores[j] > 0.5:  # Only consider high-confidence detections
                    mask = masks[j, 0] > 0.5
                    img[mask] = [255, 0, 0]  # Color mask red

                    box = boxes[j]
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(img, f"log {scores[j]:.2f}", (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save the resulting image using a global index for unique naming
            output_image_path = os.path.join(output_dir, f"result_{global_img_idx}.png")
            cv2.imwrite(output_image_path, img)
            print(f"Saved {output_image_path}")
            global_img_idx += 1  # Increment global index after saving each image
            
def main():
    print("\n==========================================================================")
    print("TASK 2 - DEVELOP MASK RCNN FOR DETECTING LOG")
    
    print("\n--------------------------------------------------------------------------")
    print("     1. DATA PREPARATION")
    
    # Split data (JSON files and corresponding png files) into train and test folder
    # split_data()
            
    # Copy images to the 'images' subdirectory for both train and test datasets
    # setup_image_directories('./log-labelled/train', 'images')
    # setup_image_directories('./log-labelled/test', 'images')

    # Convert the prepared Dataset to COCO Format - an annotations file 
    # generate_coco_annotations('./log-labelled/train', 'images', './log-labelled/train/train_annotations.json')
    # generate_coco_annotations('./log-labelled/test', 'images', './log-labelled/test/test_annotations.json') 
   
    print("\n--------------------------------------------------------------------------")
    print("     2. MODEL SETUP AND TRAINING")
    # train_model()
    
    print("\n--------------------------------------------------------------------------")
    print("     3. MODEL TESTING")
    test_model()
    
    
    
if __name__ == '__main__':
    main()