#!/usr/bin/env python3
"""
Image-Level Metrics Calculator for YOLO Segmentation

This script calculates image-level Precision, Recall, and F1 score for YOLO segmentation results.
It compares ground truth labels with model predictions and determines whether each image
contains the target classes (pleural effusion and consolidation).

Image-level metrics:
- True Positive (TP): Image has ground truth annotations AND model detects the class
- False Positive (FP): Image has NO ground truth annotations BUT model detects the class  
- False Negative (FN): Image has ground truth annotations BUT model does NOT detect the class
- True Negative (TN): Image has NO ground truth annotations AND model does NOT detect the class

Metrics:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN) 
- F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
"""

import os
import glob
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
from collections import defaultdict
import argparse


class ImageLevelMetrics:
    def __init__(self, model_path, test_images_dir, test_labels_dir, confidence_threshold=0.5):
        """
        Initialize the metrics calculator
        
        Args:
            model_path: Path to the trained YOLO model (.pt file)
            test_images_dir: Directory containing test images
            test_labels_dir: Directory containing test labels
            confidence_threshold: Confidence threshold for predictions
        """
        self.model_path = model_path
        self.test_images_dir = Path(test_images_dir)
        self.test_labels_dir = Path(test_labels_dir)
        self.confidence_threshold = confidence_threshold
        
        # Load the model
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Class names (assuming 0=consolidation, 1=pleural_effusion based on your label file)
        self.class_names = {0: 'consolidation', 1: 'pleural_effusion'}
        
    def get_ground_truth_classes(self, image_name):
        """
        Get ground truth classes present in an image
        
        Args:
            image_name: Name of the image file (without extension)
            
        Returns:
            set: Set of class IDs present in the ground truth
        """
        label_file = self.test_labels_dir / f"{image_name}.txt"
        classes = set()
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        classes.add(class_id)
        
        return classes
    
    def get_predicted_classes(self, image_path):
        """
        Get predicted classes from model inference
        
        Args:
            image_path: Path to the image file
            
        Returns:
            set: Set of class IDs predicted by the model above confidence threshold
        """
        results = self.model(image_path, verbose=False)
        classes = set()
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                # For detection/segmentation, check confidence scores
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for conf, cls_id in zip(confidences, class_ids):
                    if conf >= self.confidence_threshold:
                        classes.add(cls_id)
        
        return classes
    
    def calculate_metrics(self):
        """
        Calculate image-level precision, recall, and F1 score for each class
        
        Returns:
            dict: Dictionary containing metrics for each class
        """
        # Get all test images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(self.test_images_dir / ext)))
        
        print(f"Found {len(image_files)} test images")
        
        # Initialize counters for each class
        metrics = {}
        for class_id, class_name in self.class_names.items():
            metrics[class_name] = {
                'TP': 0,  # True Positives
                'FP': 0,  # False Positives  
                'FN': 0,  # False Negatives
                'TN': 0   # True Negatives
            }
        
        # Process each image
        for i, image_path in enumerate(image_files):
            image_name = Path(image_path).stem
            print(f"Processing {i+1}/{len(image_files)}: {image_name}")
            
            # Get ground truth and predicted classes
            gt_classes = self.get_ground_truth_classes(image_name)
            pred_classes = self.get_predicted_classes(image_path)
            
            # Calculate metrics for each class
            for class_id, class_name in self.class_names.items():
                gt_has_class = class_id in gt_classes
                pred_has_class = class_id in pred_classes
                
                if gt_has_class and pred_has_class:
                    metrics[class_name]['TP'] += 1
                elif not gt_has_class and pred_has_class:
                    metrics[class_name]['FP'] += 1
                elif gt_has_class and not pred_has_class:
                    metrics[class_name]['FN'] += 1
                else:  # not gt_has_class and not pred_has_class
                    metrics[class_name]['TN'] += 1
        
        # Calculate final metrics
        results = {}
        overall_tp = overall_fp = overall_fn = overall_tn = 0
        
        for class_name, counts in metrics.items():
            tp, fp, fn, tn = counts['TP'], counts['FP'], counts['FN'], counts['TN']
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results[class_name] = {
                'TP': tp,
                'FP': fp, 
                'FN': fn,
                'TN': tn,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }
            
            # Accumulate for overall metrics
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
            overall_tn += tn
        
        # Calculate overall metrics (macro-averaged)
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        results['Overall'] = {
            'TP': overall_tp,
            'FP': overall_fp,
            'FN': overall_fn, 
            'TN': overall_tn,
            'Precision': overall_precision,
            'Recall': overall_recall,
            'F1': overall_f1
        }
        
        return results
    
    def print_results(self, results):
        """
        Print the calculated metrics in a formatted table
        
        Args:
            results: Dictionary containing metrics for each class
        """
        print("\n" + "="*80)
        print("IMAGE-LEVEL METRICS RESULTS")
        print("="*80)
        print(f"Model: {self.model_path}")
        print(f"Test Images: {self.test_images_dir}")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print("="*80)
        
        # Print header
        print(f"{'Class':<15} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<5} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
        print("-"*80)
        
        # Print results for each class
        for class_name, metrics in results.items():
            print(f"{class_name:<15} "
                  f"{metrics['TP']:<5} "
                  f"{metrics['FP']:<5} " 
                  f"{metrics['FN']:<5} "
                  f"{metrics['TN']:<5} "
                  f"{metrics['Precision']:<10.4f} "
                  f"{metrics['Recall']:<10.4f} "
                  f"{metrics['F1']:<10.4f}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Calculate image-level metrics for YOLO segmentation')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained YOLO model (.pt file)')
    parser.add_argument('--images', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--labels', type=str, required=True, help='Directory containing test labels')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Initialize the metrics calculator
    calculator = ImageLevelMetrics(
        model_path=args.model,
        test_images_dir=args.images, 
        test_labels_dir=args.labels,
        confidence_threshold=args.conf
    )
    
    # Calculate metrics
    print("Starting image-level metrics calculation...")
    results = calculator.calculate_metrics()
    
    # Print results
    calculator.print_results(results)


if __name__ == "__main__":
    main() 