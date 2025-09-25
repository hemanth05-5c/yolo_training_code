#!/usr/bin/env python3
"""
YOLO Batch Segmentation Script
Processes a folder of images using YOLO segmentation model and creates
side-by-side visualizations with original image on left and predictions on right.
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
from tqdm import tqdm

def create_side_by_side_visualization(original_img, result, save_path):
    """
    Create side-by-side visualization with original image on left and predictions on right.
    """
    # Get the annotated image from YOLO result
    annotated_img = result.plot()
    
    # Ensure both images have the same height
    h1, w1 = original_img.shape[:2]
    h2, w2 = annotated_img.shape[:2]
    
    # Resize to same height if needed
    target_height = min(h1, h2)
    if h1 != target_height:
        original_img = cv2.resize(original_img, (int(w1 * target_height / h1), target_height))
    if h2 != target_height:
        annotated_img = cv2.resize(annotated_img, (int(w2 * target_height / h2), target_height))
    
    # Create side-by-side image
    combined_img = np.hstack([original_img, annotated_img])
    
    # Add labels with background for better visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    thickness = 2
    
    # Add "Original" label with background
    (text_width, text_height), _ = cv2.getTextSize("Original", font, font_scale, thickness)
    cv2.rectangle(combined_img, (5, 5), (text_width + 15, text_height + 15), bg_color, -1)
    cv2.putText(combined_img, "Original", (10, 25), font, font_scale, color, thickness)
    
    # Add "Predicted" label with background
    right_start = original_img.shape[1] + 5
    cv2.rectangle(combined_img, (right_start, 5), (right_start + text_width + 10, text_height + 15), bg_color, -1)
    cv2.putText(combined_img, "Predicted", (right_start + 5, 25), font, font_scale, color, thickness)
    
    # Save the combined image
    cv2.imwrite(str(save_path), combined_img)
    return combined_img

def process_images(input_folder, output_folder, model_path, confidence_threshold=0.25):
    """
    Process all images in the input folder using YOLO segmentation.
    """
    # Load YOLO model
    print(f"Loading YOLO model from: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output directories
    output_folder = Path(output_folder)
    predictions_folder = output_folder / "predictions"
    visualizations_folder = output_folder / "visualizations"
    
    predictions_folder.mkdir(parents=True, exist_ok=True)
    visualizations_folder.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files
    input_folder = Path(input_folder)
    image_files = [f for f in input_folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load original image
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            # Run YOLO prediction
            results = model(str(img_path), conf=confidence_threshold, verbose=False)
            
            # Save prediction results
            pred_filename = f"pred_{img_path.stem}.txt"
            pred_path = predictions_folder / pred_filename
            
            # Save prediction details to text file
            with open(pred_path, 'w') as f:
                f.write(f"Image: {img_path.name}\n")
                f.write(f"Model: {model_path}\n")
                f.write(f"Confidence threshold: {confidence_threshold}\n\n")
                
                result = results[0]
                if result.masks is not None and len(result.masks) > 0:
                    f.write(f"Found {len(result.masks)} segmentation masks\n")
                    for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                        cls = int(box.cls)
                        conf = float(box.conf)
                        class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"
                        f.write(f"Mask {i+1}: {class_name} (confidence: {conf:.3f})\n")
                else:
                    f.write("No segmentation masks detected\n")
            
            # Create side-by-side visualization
            viz_filename = f"viz_{img_path.stem}.jpg"
            viz_path = visualizations_folder / viz_filename
            
            create_side_by_side_visualization(original_img, results[0], viz_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Predictions saved to: {predictions_folder}")
    print(f"Visualizations saved to: {visualizations_folder}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Batch Segmentation with Side-by-Side Visualization')
    parser.add_argument('input_folder', help='Input folder containing images')
    parser.add_argument('output_folder', help='Output folder for results')
    parser.add_argument('model_path', help='Path to YOLO model file (.pt)')
    parser.add_argument('--confidence', '-c', type=float, default=0.25, 
                       help='Confidence threshold for predictions (default: 0.25)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' does not exist")
        sys.exit(1)
    
    # Process images
    process_images(args.input_folder, args.output_folder, args.model_path, args.confidence)

if __name__ == "__main__":
    main() 