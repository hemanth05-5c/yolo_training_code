#!/usr/bin/env python3
"""
Simple script to save annotated samples to a folder for inspection
"""

import os
import cv2
import json
import pandas as pd
import numpy as np
import ast
from pathlib import Path

def parse_filtered_rows(annotation_str: str):
    """Parse the filtered_rows annotation string."""
    try:
        annotations = json.loads(annotation_str.replace("'", '"'))
        return annotations
    except json.JSONDecodeError:
        try:
            annotations = ast.literal_eval(annotation_str)
            return annotations
        except:
            return []

def convert_path_format(study_path: str) -> str:
    """Convert study path to image filename format."""
    parts = study_path.split('/')
    return '_'.join(parts)

def scale_coordinates_to_image(points, img_shape):
    """Scale coordinates from 0-100 range to actual image dimensions."""
    height, width = img_shape
    scaled_points = []
    
    for point in points:
        x, y = point
        pixel_x = int((x / 100.0) * width)
        pixel_y = int((y / 100.0) * height)
        pixel_x = max(0, min(width - 1, pixel_x))
        pixel_y = max(0, min(height - 1, pixel_y))
        scaled_points.append([pixel_x, pixel_y])
    
    return scaled_points

def save_sample_annotations(num_samples=5, output_dir="visualized_samples"):
    """Save annotated samples to a folder for inspection."""
    csv_path = "/home/ai-user/RELA PROJECT/Pleural-effusion_Consolidation/filtered_rows.csv"
    images_dir = "/home/ai-user/RELA PROJECT/Pleural-effusion_Consolidation/data"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    print(f"Saving {min(num_samples, len(df))} annotated samples to '{output_dir}' folder...")
    
    saved_count = 0
    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        study_path = row['study_path']
        annotation_str = row['filtered_rows']
        
        # Convert path format
        image_filename = convert_path_format(study_path)
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            continue
        
        # Parse annotations
        annotations = parse_filtered_rows(annotation_str)
        if not annotations:
            print(f"No annotations for: {image_filename}")
            continue
        
        # Create annotated image
        img_annotated = img.copy()
        
        print(f"\nSample {i+1}: {image_filename}")
        print(f"Image size: {img.shape}")
        print(f"Annotations: {len(annotations)}")
        
        for j, annotation in enumerate(annotations):
            if not annotation.get('closed', False):
                continue
                
            points = annotation.get('points', [])
            labels = annotation.get('polygonlabels', [])
            
            if not points:
                continue
            
            print(f"  Annotation {j+1}: {labels} - {len(points)} points")
            print(f"    Coordinate range: X({min(p[0] for p in points):.1f}-{max(p[0] for p in points):.1f}), Y({min(p[1] for p in points):.1f}-{max(p[1] for p in points):.1f})")
            
            # Scale coordinates to image dimensions
            scaled_points = scale_coordinates_to_image(points, img.shape[:2])
            
            # Convert to numpy array for OpenCV
            pts = np.array(scaled_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw polygon
            color = (0, 0, 255) if j == 0 else (0, 255, 0)  # Red for first, green for others
            cv2.polylines(img_annotated, [pts], True, color, 3)
            
            # Fill polygon with transparency
            overlay = img_annotated.copy()
            cv2.fillPoly(overlay, [pts], color)
            img_annotated = cv2.addWeighted(img_annotated, 0.7, overlay, 0.3, 0)
            
            # Add label
            label_text = labels[0] if labels else f"Polygon {j+1}"
            cv2.putText(img_annotated, label_text, tuple(scaled_points[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        
        # Create output filename
        output_filename = f"sample_{i+1:02d}_{image_filename}"
        output_path_full = output_path / output_filename
        
        # Save the annotated image
        success = cv2.imwrite(str(output_path_full), img_annotated)
        
        if success:
            saved_count += 1
            print(f"  ✓ Saved: {output_filename}")
        else:
            print(f"  ✗ Failed to save: {output_filename}")
    
    print(f"\nCompleted! Saved {saved_count} annotated images to '{output_dir}' folder.")
    print(f"You can now view the images to verify the annotations before creating the YOLO dataset.")

if __name__ == "__main__":
    # Save 10 sample annotations to visualized_samples folder
    save_sample_annotations(num_samples=10, output_dir="visualized_samples") 