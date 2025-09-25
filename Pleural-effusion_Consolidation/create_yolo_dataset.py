#!/usr/bin/env python3
"""
YOLO Dataset Creator for Pleural Effusion Segmentation

This script converts the CSV-based annotation data to YOLO format for segmentation tasks.
It handles coordinate normalization from 0-100 scale to 0-1 scale and creates the proper
directory structure for YOLO training.

Key Features:
- Converts polygon coordinates from 0-100 scale to 0-1 scale (YOLO format)
- Creates train/val/test splits
- Generates YOLO-compatible annotation files
- Creates dataset.yaml configuration file
- Handles multiple polygons per image
- Provides comprehensive logging and error handling

Usage:
    python create_yolo_dataset.py
"""

import os
import cv2
import json
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any
import yaml
from sklearn.model_selection import train_test_split
import logging
import ast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_dataset_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YOLODatasetCreator:
    """
    Creates YOLO-format dataset from CSV annotations for segmentation tasks.
    
    This class handles the conversion of polygon annotations from a CSV file
    to YOLO segmentation format, including coordinate normalization and
    dataset splitting.
    """
    
    def __init__(self, 
                 csv_path: str,
                 images_dir: str,
                 output_dir: str,
                 class_names: List[str] = None):
        """
        Initialize the YOLO dataset creator.
        
        Args:
            csv_path: Path to the CSV file containing annotations
            images_dir: Directory containing the images
            output_dir: Directory where YOLO dataset will be created
            class_names: List of class names (defaults to ['Pleural_Effusion'])
        """
        self.csv_path = csv_path
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.class_names = class_names or ['Pleural_Effusion']
        self.class_map = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Create output directories
        self.create_directory_structure()
        
        logger.info(f"Initialized YOLO Dataset Creator")
        logger.info(f"CSV path: {csv_path}")
        logger.info(f"Images directory: {images_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Classes: {self.class_names}")
    
    def create_directory_structure(self):
        """Create the YOLO dataset directory structure."""
        directories = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val", 
            self.output_dir / "images" / "test",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val",
            self.output_dir / "labels" / "test"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def normalize_coordinates(self, points: List[List[float]]) -> List[List[float]]:
        """
        Normalize coordinates from 0-100 scale to 0-1 scale (YOLO format).
        
        Args:
            points: List of [x, y] coordinates in 0-100 scale
            
        Returns:
            List of [x, y] coordinates in 0-1 scale
        """
        normalized_points = []
        for point in points:
            x, y = point
            # Convert from 0-100 scale to 0-1 scale
            norm_x = x / 100.0
            norm_y = y / 100.0
            
            # Clamp values to [0, 1] range
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            
            normalized_points.append([norm_x, norm_y])
        
        return normalized_points
    
    def parse_annotation(self, annotation_str: str) -> List[Dict[str, Any]]:
        """
        Parse the annotation string from CSV into structured data.
        
        Args:
            annotation_str: JSON string containing polygon annotations
            
        Returns:
            List of annotation dictionaries
        """
        try:
            # Try to parse as JSON first
            annotations = json.loads(annotation_str.replace("'", '"'))
            return annotations
        except json.JSONDecodeError:
            try:
                # Try to evaluate as Python literal
                annotations = ast.literal_eval(annotation_str)
                return annotations
            except Exception as e:
                logger.error(f"Failed to parse annotation: {annotation_str[:100]}...")
                logger.error(f"Error: {e}")
                return []
    
    def create_yolo_annotation(self, annotations: List[Dict[str, Any]]) -> str:
        """
        Create YOLO format annotation string from parsed annotations.
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            YOLO format annotation string
        """
        yolo_lines = []
        
        for annotation in annotations:
            if not annotation.get('closed', False):
                continue
                
            points = annotation.get('points', [])
            labels = annotation.get('polygonlabels', [])
            
            if not points or not labels:
                continue
            
            # Get class ID
            label = labels[0] if labels else 'Pleural Effusion'
            # Normalize label name (replace spaces with underscores)
            label_normalized = label.replace(' ', '_')
            class_id = self.class_map.get(label_normalized, 0)
            
            # Normalize coordinates
            normalized_points = self.normalize_coordinates(points)
            
            # Create YOLO segmentation line
            # Format: class_id x1 y1 x2 y2 x3 y3 ...
            coords_flat = []
            for point in normalized_points:
                coords_flat.extend([f"{point[0]:.6f}", f"{point[1]:.6f}"])
            
            yolo_line = f"{class_id} " + " ".join(coords_flat)
            yolo_lines.append(yolo_line)
        
        return "\n".join(yolo_lines)
    
    def get_image_path(self, study_path: str) -> Path:
        """
        Convert study path to actual image file path.
        
        Args:
            study_path: Study path from CSV (e.g., "2024/09/16/6E2797CC/E27AA09B/EDA46F3D.jpeg")
            
        Returns:
            Path to the actual image file
        """
        # Convert path format: 2024/09/16/6E2797CC/E27AA09B/EDA46F3D.jpeg
        # to filename format: 2024_09_16_6E2797CC_E27AA09B_EDA46F3D.jpeg
        parts = study_path.split('/')
        filename = '_'.join(parts)
        return self.images_dir / filename
    
    def verify_image_exists(self, image_path: Path) -> bool:
        """
        Verify that the image file exists and can be read.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image exists and can be read, False otherwise
        """
        if not image_path.exists():
            return False
        
        try:
            img = cv2.imread(str(image_path))
            return img is not None
        except Exception:
            return False
    
    def copy_image_to_split(self, source_path: Path, split: str, filename: str):
        """
        Copy image to the appropriate split directory.
        
        Args:
            source_path: Source image path
            split: Dataset split ('train', 'val', 'test')
            filename: Target filename
        """
        dest_path = self.output_dir / "images" / split / filename
        shutil.copy2(source_path, dest_path)
    
    def save_annotation(self, annotation_content: str, split: str, filename: str):
        """
        Save YOLO annotation to the appropriate split directory.
        
        Args:
            annotation_content: YOLO format annotation content
            split: Dataset split ('train', 'val', 'test')
            filename: Target filename (without extension)
        """
        label_path = self.output_dir / "labels" / split / f"{filename}.txt"
        with open(label_path, 'w') as f:
            f.write(annotation_content)
    
    def create_dataset_yaml(self, train_count: int, val_count: int, test_count: int):
        """
        Create dataset.yaml file for YOLO training.
        
        Args:
            train_count: Number of training images
            val_count: Number of validation images  
            test_count: Number of test images
        """
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': self.class_names,
            
            # Additional metadata
            'task': 'segment',  # Specify this is for segmentation
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count,
            'total_images': train_count + val_count + test_count,
            'coordinate_format': 'normalized_0_1',
            'original_coordinate_scale': '0_100'
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created dataset.yaml at {yaml_path}")
        return yaml_path
    
    def process_dataset(self, 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.2,
                       test_ratio: float = 0.1,
                       random_state: int = 42) -> Dict[str, int]:
        """
        Process the entire dataset and create YOLO format files.
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            random_state: Random state for reproducible splits
            
        Returns:
            Dictionary with counts for each split
        """
        logger.info("Starting dataset processing...")
        
        # Validate split ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Load CSV data
        logger.info(f"Loading CSV data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Filter out rows with missing or invalid data
        valid_rows = []
        skipped_rows = 0
        
        for idx, row in df.iterrows():
            study_path = row['study_path']
            annotation_str = row['filtered_rows']
            
            # Get image path
            image_path = self.get_image_path(study_path)
            
            # Verify image exists
            if not self.verify_image_exists(image_path):
                logger.warning(f"Image not found or unreadable: {image_path}")
                skipped_rows += 1
                continue
            
            # Parse annotations
            annotations = self.parse_annotation(annotation_str)
            if not annotations:
                logger.warning(f"No valid annotations for {study_path}")
                skipped_rows += 1
                continue
            
            valid_rows.append({
                'study_path': study_path,
                'image_path': image_path,
                'annotations': annotations,
                'filename': image_path.name
            })
        
        logger.info(f"Valid rows: {len(valid_rows)}, Skipped: {skipped_rows}")
        
        if not valid_rows:
            raise ValueError("No valid data found for processing")
        
        # Create dataset splits
        logger.info("Creating dataset splits...")
        
        # First split: train vs (val + test)
        train_data, temp_data = train_test_split(
            valid_rows, 
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Second split: val vs test
        if test_ratio > 0:
            val_size = val_ratio / (val_ratio + test_ratio)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_size),
                random_state=random_state
            )
        else:
            val_data = temp_data
            test_data = []
        
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        logger.info(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Process each split
        counts = {}
        for split_name, split_data in splits.items():
            logger.info(f"Processing {split_name} split...")
            
            processed_count = 0
            for item in split_data:
                try:
                    # Create YOLO annotation
                    yolo_annotation = self.create_yolo_annotation(item['annotations'])
                    
                    if not yolo_annotation.strip():
                        logger.warning(f"Empty annotation for {item['filename']}")
                        continue
                    
                    # Copy image
                    self.copy_image_to_split(
                        item['image_path'], 
                        split_name, 
                        item['filename']
                    )
                    
                    # Save annotation
                    annotation_filename = item['filename'].rsplit('.', 1)[0]
                    self.save_annotation(
                        yolo_annotation,
                        split_name,
                        annotation_filename
                    )
                    
                    processed_count += 1
                    
                    # Progress update every 50 files
                    if processed_count % 50 == 0:
                        logger.info(f"  Processed {processed_count} files for {split_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {item['filename']}: {e}")
                    continue
            
            counts[split_name] = processed_count
            logger.info(f"Successfully processed {processed_count} images for {split_name}")
        
        # Create dataset.yaml
        self.create_dataset_yaml(
            counts.get('train', 0),
            counts.get('val', 0), 
            counts.get('test', 0)
        )
        
        # Log final statistics
        total_processed = sum(counts.values())
        logger.info(f"Dataset creation completed!")
        logger.info(f"Total images processed: {total_processed}")
        logger.info(f"Train: {counts.get('train', 0)}")
        logger.info(f"Val: {counts.get('val', 0)}")
        logger.info(f"Test: {counts.get('test', 0)}")
        logger.info(f"Output directory: {self.output_dir}")
        
        return counts
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate the created dataset by checking file counts and format.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating created dataset...")
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'counts': {}
        }
        
        for split in ['train', 'val', 'test']:
            images_dir = self.output_dir / "images" / split
            labels_dir = self.output_dir / "labels" / split
            
            image_files = list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            label_files = list(labels_dir.glob("*.txt"))
            
            results['counts'][split] = {
                'images': len(image_files),
                'labels': len(label_files)
            }
            
            # Check if image and label counts match
            if len(image_files) != len(label_files):
                error_msg = f"{split}: Image count ({len(image_files)}) != Label count ({len(label_files)})"
                results['errors'].append(error_msg)
                results['valid'] = False
            
            # Check for orphaned files
            image_stems = {f.stem for f in image_files}
            label_stems = {f.stem for f in label_files}
            
            orphaned_images = image_stems - label_stems
            orphaned_labels = label_stems - image_stems
            
            if orphaned_images:
                results['warnings'].append(f"{split}: Orphaned images: {list(orphaned_images)[:5]}{'...' if len(orphaned_images) > 5 else ''}")
            
            if orphaned_labels:
                results['warnings'].append(f"{split}: Orphaned labels: {list(orphaned_labels)[:5]}{'...' if len(orphaned_labels) > 5 else ''}")
        
        # Check dataset.yaml
        yaml_path = self.output_dir / "dataset.yaml"
        if not yaml_path.exists():
            results['errors'].append("dataset.yaml file missing")
            results['valid'] = False
        
        # Log validation results
        if results['valid']:
            logger.info("Dataset validation passed!")
        else:
            logger.error("Dataset validation failed!")
            for error in results['errors']:
                logger.error(f"  Error: {error}")
        
        for warning in results['warnings']:
            logger.warning(f"  Warning: {warning}")
        
        return results

def main():
    """Main function to create the YOLO dataset."""
    
    # Configuration
    csv_path = "/home/ai-user/RELA PROJECT/Pleural-effusion_Consolidation/filtered_rows.csv"
    images_dir = "/home/ai-user/RELA PROJECT/Pleural-effusion_Consolidation/data"
    output_dir = "/home/ai-user/RELA PROJECT/Pleural-effusion_Consolidation/yolo_dataset"
    class_names = ["Pleural_Effusion", "Consolidation"]
    
    try:
        # Create YOLO dataset
        creator = YOLODatasetCreator(
            csv_path=csv_path,
            images_dir=images_dir,
            output_dir=output_dir,
            class_names=class_names
        )
        
        # Process the dataset
        counts = creator.process_dataset(
            train_ratio=0.7,
            val_ratio=0.2, 
            test_ratio=0.1,
            random_state=42
        )
        
        # Validate the dataset
        validation_results = creator.validate_dataset()
        
        # Print summary
        print("\n" + "="*60)
        print("YOLO DATASET CREATION SUMMARY")
        print("="*60)
        print(f"Dataset location: {output_dir}")
        print(f"Train images: {counts.get('train', 0)}")
        print(f"Validation images: {counts.get('val', 0)}")
        print(f"Test images: {counts.get('test', 0)}")
        print(f"Total images: {sum(counts.values())}")
        print(f"Validation status: {'PASSED' if validation_results['valid'] else 'FAILED'}")
        
        if validation_results['errors']:
            print("\nErrors:")
            for error in validation_results['errors']:
                print(f"  - {error}")
        
        if validation_results['warnings']:
            print("\nWarnings:")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")
        
        print("\nDataset ready for YOLO training!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Failed to create YOLO dataset: {e}")
        raise

if __name__ == "__main__":
    main() 