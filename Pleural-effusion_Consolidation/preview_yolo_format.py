#!/usr/bin/env python3
"""
Preview YOLO Format Conversion

This script shows how the CSV annotations will be converted to YOLO format
without actually creating the full dataset.
"""

import pandas as pd
import json
import ast

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

def normalize_coordinates(points):
    """Normalize coordinates from 0-100 scale to 0-1 scale."""
    normalized_points = []
    for point in points:
        x, y = point
        norm_x = x / 100.0
        norm_y = y / 100.0
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        normalized_points.append([norm_x, norm_y])
    return normalized_points

def create_yolo_annotation(annotations):
    """Create YOLO format annotation string."""
    yolo_lines = []
    
    for annotation in annotations:
        if not annotation.get('closed', False):
            continue
            
        points = annotation.get('points', [])
        labels = annotation.get('polygonlabels', [])
        
        if not points or not labels:
            continue
        
        # Class ID (0 for Pleural_Effusion)
        class_id = 0
        
        # Normalize coordinates
        normalized_points = normalize_coordinates(points)
        
        # Create YOLO segmentation line
        coords_flat = []
        for point in normalized_points:
            coords_flat.extend([f"{point[0]:.6f}", f"{point[1]:.6f}"])
        
        yolo_line = f"{class_id} " + " ".join(coords_flat)
        yolo_lines.append(yolo_line)
    
    return "\n".join(yolo_lines)

def convert_path_format(study_path: str) -> str:
    """Convert study path to image filename format."""
    parts = study_path.split('/')
    return '_'.join(parts)

def preview_yolo_conversion():
    """Preview YOLO format conversion for first few samples."""
    csv_path = "/home/ai-user/RELA PROJECT/Pleural-effusion_Consolidation/filtered_rows.csv"
    
    print("Loading CSV data...")
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df)}")
    print("\nPreviewing YOLO format conversion for first 3 samples:\n")
    
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        study_path = row['study_path']
        annotation_str = row['filtered_rows']
        
        # Convert path format
        image_filename = convert_path_format(study_path)
        
        print(f"{'='*60}")
        print(f"Sample {i+1}")
        print(f"{'='*60}")
        print(f"Original study path: {study_path}")
        print(f"Image filename: {image_filename}")
        
        # Parse annotations
        annotations = parse_filtered_rows(annotation_str)
        print(f"Number of annotations: {len(annotations)}")
        
        if annotations:
            print(f"\nOriginal annotation (first polygon):")
            first_annotation = annotations[0]
            points = first_annotation.get('points', [])
            labels = first_annotation.get('polygonlabels', [])
            print(f"  Label: {labels}")
            print(f"  Points (0-100 scale): {len(points)} points")
            print(f"  First 3 points: {points[:3]}")
            
            # Show coordinate ranges
            if points:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                print(f"  X range: {min(x_coords):.2f} - {max(x_coords):.2f}")
                print(f"  Y range: {min(y_coords):.2f} - {max(y_coords):.2f}")
            
            # Convert to YOLO format
            yolo_annotation = create_yolo_annotation(annotations)
            
            print(f"\nYOLO format annotation:")
            yolo_lines = yolo_annotation.split('\n')
            for j, line in enumerate(yolo_lines):
                parts = line.split()
                class_id = parts[0]
                coords = parts[1:]
                print(f"  Polygon {j+1}: class_id={class_id}, {len(coords)//2} points")
                print(f"    First 6 coordinates: {' '.join(coords[:6])}")
                if len(coords) > 6:
                    print(f"    ... ({len(coords)-6} more coordinates)")
            
            print(f"\nFull YOLO annotation line:")
            print(f"  {yolo_annotation}")
            
        else:
            print("No valid annotations found")
        
        print(f"\n")
    
    print(f"{'='*60}")
    print("YOLO Dataset Structure Preview")
    print(f"{'='*60}")
    print("Directory structure that will be created:")
    print("yolo_dataset/")
    print("├── images/")
    print("│   ├── train/")
    print("│   ├── val/")
    print("│   └── test/")
    print("├── labels/")
    print("│   ├── train/")
    print("│   ├── val/")
    print("│   └── test/")
    print("└── dataset.yaml")
    print("")
    print("Each image will have a corresponding .txt file with YOLO annotations.")
    print("Format: class_id x1 y1 x2 y2 x3 y3 ... (normalized 0-1 coordinates)")
    print("")
    print("Example files:")
    sample_filename = convert_path_format(df.iloc[0]['study_path'])
    base_name = sample_filename.rsplit('.', 1)[0]
    print(f"  images/train/{sample_filename}")
    print(f"  labels/train/{base_name}.txt")

if __name__ == "__main__":
    preview_yolo_conversion() 