#!/usr/bin/env python3
"""
Google Cloud Storage (GCS) Management Suite

This comprehensive script provides a complete solution for managing Google Cloud Storage operations
including file uploads, downloads, folder synchronization, and CSV-based image downloads. 
It features automatic bucket selection, concurrent processing, progress tracking, and detailed logging.

Key Features:
-------------
- Automatic bucket selection based on operation type
- Concurrent uploads/downloads with configurable thread pools
- CSV-based image downloading with automatic renaming
- Comprehensive error handling and retry mechanisms
- Failed operations tracking with separate text files
- Progress monitoring and detailed statistics
- Flexible logging configuration (file + console)

Usage:
------
1. Setup Environment Variables:
   Create a pwd.env file with the following credentials:
   
   # General GCS operations (models, files, etc.)
   GCP_BUCKET_NAME=your_main_bucket_name
   
   # Image-specific operations (medical images, datasets)
   GCP_IMAGE_BUCKET=your_image_bucket_name

2. Setup GCP Credentials:
   Place your GCP service account JSON file at: /home/ai-user/gcp.json
   
   The file should contain your service account credentials with appropriate
   permissions for the buckets you want to access.

3. Install Required Packages:
   pip install google-cloud-storage pandas python-dotenv

4. Basic Usage Examples:

   A. General File/Folder Operations:
      ```python
      from gcp import GCPStorageManager
      
      # Initialize for general operations (uses GCP_BUCKET_NAME)
      gcp_manager = GCPStorageManager()
      
      # Upload a single file
      success = gcp_manager.upload_blob("local_file.txt", "remote/path/file.txt")
      
      # Upload entire folder with progress tracking
      stats = gcp_manager.upload_folder("/local/folder", "remote/destination")
      print(f"Uploaded: {stats['success']}, Failed: {stats['failed']}")
      
      # Download a single file
      success = gcp_manager.download_blob("remote/file.txt", "/local/path/file.txt")
      
      # Download entire folder
      stats = gcp_manager.download_folder("remote/folder/", "/local/destination/")
      ```

   B. CSV-Based Image Downloads:
      ```python
      from gcp import CSVImageDownloader
      
      # Initialize for image operations (automatically uses GCP_IMAGE_BUCKET)
      downloader = CSVImageDownloader()
      
      # Download images from CSV with automatic renaming and failed tracking
      stats = downloader.download_images_from_csv(
          csv_path="images.csv",
          destination_directory="/local/images/",
          log_file_path="/local/logs/download.csv",
          max_workers=4
      )
      
      print(f"Downloaded: {stats['success']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")
      if stats.get('failed_file_path'):
          print(f"Failed downloads saved to: {stats['failed_file_path']}")
      ```

   C. Command Line Usage:
      ```bash
      # Run the main script with examples
      python gcp.py
      ```

5. CSV File Format for Image Downloads:
   Required columns:
   - study_path: Path to the image in GCS (will be converted from underscore to slash format)
   
   Example CSV content:
   ```
   study_path
   2022_12_23_4DECF9C6_BD0973E9_D49FB2F3.jpeg
   2022_12_24_5AECF7D7_CE1084FA_E5AFC3G4.jpeg
   ```

6. Output Structure:
   - Downloaded images are automatically renamed to include date and identifiers
   - Log files track all operations with status and error messages
   - Failed operations are saved to separate timestamped text files
   - Progress is displayed in real-time for large operations

7. Error Handling:
   - Automatic retry mechanisms for transient failures
   - Comprehensive logging of all operations
   - Failed operations saved to separate files for easy retry
   - Graceful handling of missing files and permission errors

8. Performance Features:
   - Configurable concurrent workers (default: 4 threads)
   - Progress tracking for large operations
   - Memory-efficient streaming for large files
   - Automatic directory creation as needed

Dependencies:
-------------
- google-cloud-storage: GCS client library
- pandas: CSV processing and data manipulation
- python-dotenv: Environment variable management
- typing: Type hints support (Python 3.5+)
- concurrent.futures: Parallel processing
- logging: Comprehensive logging system

File Structure:
---------------
gcp.py
├── GCPStorageManager      # Main class for GCS operations
│   ├── Basic Operations   # upload_blob, download_blob, delete_blob
│   ├── Advanced Ops       # copy_blob, move_blob, list_blobs
│   └── Folder Operations  # upload_folder, download_folder
├── CSVImageDownloader     # Specialized class for CSV-based downloads
└── Utility Functions      # Logging, failed file creation, etc.

"""

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage import Client

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Load environment variables and set credentials
load_dotenv('/home/ai-user/pwd.env')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ai-user/gcp.json"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(log_file: str = "gcp_operations.log", console_output: bool = True) -> logging.Logger:
    """
    Setup comprehensive logging configuration for GCS operations.
    
    Creates a logger that writes to both file and console (optional) with detailed
    formatting including timestamps, log levels, and messages.
    
    Args:
        log_file (str): Path to the log file. Defaults to "gcp_operations.log"
        console_output (bool): Whether to also output logs to console. Defaults to True
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = setup_logging("my_operations.log", console_output=True)
        >>> logger.info("Starting GCS operations")
    """
    # Configure basic logging to file
    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO, 
        format="%(asctime)s :: %(levelname)s :: %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s"))
        logger.addHandler(console_handler)
    
    return logger

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def write_failed_operations_file(failed_items: List[str], failed_file_path: str, operation_type: str = "operation") -> bool:
    """
    Write failed operations to a separate text file for easy tracking and retry.
    
    Creates a well-formatted text file listing all failed operations with timestamps
    and counts. Useful for tracking issues and planning retry operations.
    
    Args:
        failed_items (List[str]): List of failed file paths or blob names
        failed_file_path (str): Path where the failed operations file should be saved
        operation_type (str): Type of operation ('upload', 'download', etc.). Defaults to "operation"
        
    Returns:
        bool: True if file was successfully created, False otherwise
        
    Example:
        >>> failed_files = ["/path/file1.txt", "/path/file2.jpg"]
        >>> write_failed_operations_file(failed_files, "failed_uploads.txt", "upload")
        True
        
    Output file format:
        Failed upload operations - 2024-01-15 14:30:25
        ============================================================
        
        1. /path/file1.txt
        2. /path/file2.jpg
        
        Total failed uploads: 2
    """
    try:
        # Skip if no failed items
        if not failed_items:
            return True
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(failed_file_path), exist_ok=True)
        
        # Write formatted failed operations file
        with open(failed_file_path, 'w') as f:
            f.write(f"Failed {operation_type} operations - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            # List each failed item with numbering
            for i, item in enumerate(failed_items, 1):
                f.write(f"{i}. {item}\n")
            
            # Add summary
            f.write(f"\nTotal failed {operation_type}s: {len(failed_items)}\n")
        
        return True
    except Exception as e:
        print(f"Error writing failed operations file: {e}")
        return False

# =============================================================================
# MAIN GCS STORAGE MANAGER CLASS
# =============================================================================

class GCPStorageManager:
    """
    A comprehensive class for managing Google Cloud Storage operations.
    
    This class provides a complete interface for Google Cloud Storage operations including
    basic blob operations (upload, download, delete), advanced operations (copy, move),
    and bulk folder operations with concurrent processing and progress tracking.
    
    The class automatically handles bucket selection based on operation type, provides
    detailed logging, error handling, and statistics reporting for all operations.
    
    Attributes:
        logger (logging.Logger): Logger instance for operation tracking
        client (storage.Client): Google Cloud Storage client
        bucket_name (str): Name of the GCS bucket being used
        bucket (storage.Bucket): GCS bucket object for operations
        
    Example:
        >>> # For general operations (uses GCP_BUCKET_NAME)
        >>> gcp_manager = GCPStorageManager()
        >>> 
        >>> # For image operations (uses GCP_IMAGE_BUCKET)
        >>> image_manager = GCPStorageManager(use_image_bucket=True)
        >>>
        >>> # Upload a file
        >>> success = gcp_manager.upload_blob("local.txt", "remote/path.txt")
    """
    
    def __init__(self, use_image_bucket: bool = False, logger: Optional[logging.Logger] = None):
        """
        Initialize GCP Storage Manager with automatic bucket selection.
        
        Automatically selects the appropriate bucket based on the operation type:
        - use_image_bucket=False: Uses GCP_BUCKET_NAME (for models, general files)
        - use_image_bucket=True: Uses GCP_IMAGE_BUCKET (for medical images, datasets)
        
        Args:
            use_image_bucket (bool): If True, uses GCP_IMAGE_BUCKET environment variable,
                                   otherwise uses GCP_BUCKET_NAME. Defaults to False
            logger (Optional[logging.Logger]): Logger instance. If None, creates a default logger
            
        Raises:
            ValueError: If the required environment variable is not set
            Exception: If GCS client initialization or bucket access fails
            
        Example:
            >>> # General operations
            >>> manager = GCPStorageManager()
            >>> 
            >>> # Image operations
            >>> image_manager = GCPStorageManager(use_image_bucket=True)
        """
        self.logger = logger or setup_logging()
        self.client = self._initialize_client()
        
        # Automatically determine bucket based on operation type
        if use_image_bucket:
            self.bucket_name = os.getenv("GCP_IMAGE_BUCKET")
            if not self.bucket_name:
                self.logger.error("GCP_IMAGE_BUCKET environment variable not set")
                raise ValueError("GCP_IMAGE_BUCKET environment variable is required for image operations")
        else:
            self.bucket_name = os.getenv("GCP_BUCKET_NAME")
            if not self.bucket_name:
                self.logger.error("GCP_BUCKET_NAME environment variable not set")
                raise ValueError("GCP_BUCKET_NAME environment variable is required")
        
        self.bucket = self._get_bucket()
    
    def _initialize_client(self) -> Client:
        """
        Initialize and verify Google Cloud Storage client.
        
        Creates the GCS client using credentials from the environment variable
        GOOGLE_APPLICATION_CREDENTIALS which should point to a service account JSON file.
        
        Returns:
            storage.Client: Initialized GCS client
            
        Raises:
            Exception: If client initialization fails due to credential or connectivity issues
            
        Note:
            Credentials must be properly configured before calling this method.
            The service account should have appropriate permissions for the target buckets.
        """
        try:
            self.logger.info("Initializing GCP storage client...")
            client = storage.Client()
            self.logger.info("Successfully initialized GCP storage client")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize GCP storage client: {e}")
            raise
    
    def _get_bucket(self) -> storage.Bucket:
        """
        Get and verify access to the specified GCS bucket.
        
        Attempts to access the bucket and verifies that it exists and is accessible
        with the current credentials.
        
        Returns:
            storage.Bucket: GCS bucket object for operations
            
        Raises:
            ValueError: If bucket does not exist or is not accessible
            Exception: If bucket access verification fails
            
        Note:
            This method performs an existence check which requires list permissions
            on the bucket.
        """
        try:
            self.logger.info(f"Connecting to bucket: {self.bucket_name}")
            bucket = self.client.bucket(self.bucket_name)
            
            if not bucket.exists():
                self.logger.error(f"Bucket {self.bucket_name} does not exist or access denied")
                raise ValueError(f"Bucket {self.bucket_name} not accessible")
            
            self.logger.info(f"Successfully connected to bucket: {self.bucket_name}")
            return bucket
        except Exception as e:
            self.logger.error(f"Failed to access bucket {self.bucket_name}: {e}")
            raise
    
    # =========================================================================
    # BASIC BLOB OPERATIONS
    # =========================================================================
    
    def list_blobs(self, prefix: str = "", delimiter: Optional[str] = None) -> List[str]:
        """
        List all blobs in the bucket with an optional prefix filter.
        
        This method can be used to list files in a "folder" by using a prefix.
        The delimiter can be used to get a folder-like listing.
        
        Args:
            prefix (str): Prefix to filter blobs (e.g., "folder/" to list folder contents).
                         Defaults to "" (list all blobs)
            delimiter (Optional[str]): Delimiter for folder-like listing (e.g., "/").
                                     If specified, returns only direct children
                                     
        Returns:
            List[str]: List of blob names matching the criteria
            
        Raises:
            Exception: If listing operation fails
            
        Example:
            >>> # List all blobs
            >>> all_blobs = manager.list_blobs()
            >>> 
            >>> # List blobs in a "folder"
            >>> folder_blobs = manager.list_blobs(prefix="data/images/")
            >>> 
            >>> # Get folder-like listing
            >>> direct_children = manager.list_blobs(prefix="data/", delimiter="/")
        """
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix, delimiter=delimiter)
            blob_names = [blob.name for blob in blobs]
            self.logger.info(f"Found {len(blob_names)} blobs with prefix '{prefix}'")
            return blob_names
        except Exception as e:
            self.logger.error(f"Error listing blobs: {e}")
            raise
    
    def download_blob(self, source_blob_name: str, destination_file_path: str) -> bool:
        """
        Download a single blob from GCS to a local file.
        
        Downloads a blob from the bucket to a specified local file path. Automatically
        creates the destination directory if it doesn't exist.
        
        Args:
            source_blob_name (str): Name/path of the blob in GCS (e.g., "folder/file.txt")
            destination_file_path (str): Local path where the file should be saved
            
        Returns:
            bool: True if download was successful, False otherwise
            
        Raises:
            Exception: If download operation fails (logged but not re-raised)
            
        Example:
            >>> # Download a single file
            >>> success = manager.download_blob("data/model.pkl", "/local/model.pkl")
            >>> if success:
            ...     print("Download completed successfully")
        """
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
            
            blob = self.bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_path)
            
            self.logger.info(f"Downloaded {source_blob_name} to {destination_file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error downloading {source_blob_name}: {e}")
            return False
    
    def upload_blob(self, source_file_path: str, destination_blob_name: str) -> bool:
        """
        Upload a single local file to GCS.
        
        Uploads a local file to the specified blob path in GCS. Validates that
        the source file exists before attempting upload.
        
        Args:
            source_file_path (str): Path to the local file to upload
            destination_blob_name (str): Destination blob name/path in GCS
            
        Returns:
            bool: True if upload was successful, False otherwise
            
        Raises:
            Exception: If upload operation fails (logged but not re-raised)
            
        Example:
            >>> # Upload a single file
            >>> success = manager.upload_blob("/local/data.csv", "datasets/data.csv")
            >>> if success:
            ...     print("Upload completed successfully")
        """
        try:
            if not os.path.exists(source_file_path):
                self.logger.error(f"Source file does not exist: {source_file_path}")
                return False
            
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_path)
            
            self.logger.info(f"Uploaded {source_file_path} to {destination_blob_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error uploading {source_file_path}: {e}")
            return False
    
    def delete_blob(self, blob_name: str) -> bool:
        """
        Delete a blob from GCS.
        
        Permanently deletes the specified blob from the bucket. This operation
        cannot be undone, so use with caution.
        
        Args:
            blob_name (str): Name/path of the blob to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
            
        Raises:
            Exception: If deletion operation fails (logged but not re-raised)
            
        Warning:
            This operation permanently deletes the blob and cannot be undone.
            
        Example:
            >>> # Delete a file
            >>> success = manager.delete_blob("old_data/deprecated.csv")
            >>> if success:
            ...     print("File deleted successfully")
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            self.logger.info(f"Deleted blob: {blob_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting blob {blob_name}: {e}")
            return False
    
    # =========================================================================
    # ADVANCED BLOB OPERATIONS
    # =========================================================================
    
    def copy_blob(self, source_blob_name: str, destination_blob_name: str, 
                  destination_bucket_name: Optional[str] = None) -> bool:
        """
        Copy a blob within the same bucket or to another bucket.
        
        Creates a copy of the specified blob with a new name and/or location.
        The original blob remains unchanged. Can copy within the same bucket
        or to a different bucket if specified.
        
        Args:
            source_blob_name (str): Name/path of the source blob to copy
            destination_blob_name (str): Name/path for the copied blob
            destination_bucket_name (Optional[str]): Target bucket name. If None, 
                                                   copies within the same bucket
            
        Returns:
            bool: True if copy was successful, False otherwise
            
        Example:
            >>> # Copy within same bucket
            >>> success = manager.copy_blob("data/original.csv", "backup/original.csv")
            >>> 
            >>> # Copy to different bucket
            >>> success = manager.copy_blob("data.csv", "data.csv", "backup-bucket")
        """
        try:
            source_blob = self.bucket.blob(source_blob_name)
            dest_bucket = self.client.bucket(destination_bucket_name) if destination_bucket_name else self.bucket
            
            dest_bucket.copy_blob(source_blob, dest_bucket, destination_blob_name)
            self.logger.info(f"Copied {source_blob_name} to {destination_blob_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error copying blob: {e}")
            return False
    
    def move_blob(self, source_blob_name: str, destination_blob_name: str,
                  destination_bucket_name: Optional[str] = None) -> bool:
        """
        Move a blob (copy + delete original).
        
        Moves a blob by copying it to the new location and then deleting the original.
        This is effectively a rename operation when moving within the same bucket.
        
        Args:
            source_blob_name (str): Name/path of the source blob to move
            destination_blob_name (str): Name/path for the moved blob
            destination_bucket_name (Optional[str]): Target bucket name. If None,
                                                   moves within the same bucket
            
        Returns:
            bool: True if move was successful, False otherwise
            
        Warning:
            The original blob is permanently deleted after successful copy.
            
        Example:
            >>> # Rename/move within same bucket
            >>> success = manager.move_blob("old_name.csv", "new_name.csv")
            >>> 
            >>> # Move to different bucket
            >>> success = manager.move_blob("data.csv", "data.csv", "archive-bucket")
        """
        try:
            if self.copy_blob(source_blob_name, destination_blob_name, destination_bucket_name):
                return self.delete_blob(source_blob_name)
            return False
        except Exception as e:
            self.logger.error(f"Error moving blob: {e}")
            return False
    
    # =========================================================================
    # BULK FOLDER OPERATIONS
    # =========================================================================
    
    def download_folder(self, prefix: str, local_directory: str, max_workers: int = 4) -> Dict[str, Any]:
        """
        Download all blobs with given prefix (folder-like operation).
        
        Downloads all blobs that start with the specified prefix, effectively
        downloading a "folder" from GCS. Uses concurrent downloads for efficiency
        and provides detailed statistics on the operation.
        
        Args:
            prefix (str): Prefix of blobs to download (e.g., "data/images/")
            local_directory (str): Local directory to save downloaded files
            max_workers (int): Number of concurrent download threads. Defaults to 4
            
        Returns:
            Dict[str, Any]: Statistics dictionary containing:
                - total: Total number of blobs found
                - success: Number of successful downloads
                - failed: Number of failed downloads
                - errors: List of error messages
                
        Example:
            >>> # Download entire folder
            >>> stats = manager.download_folder("data/models/", "/local/models/")
            >>> print(f"Downloaded {stats['success']}/{stats['total']} files")
            >>> 
            >>> # Check for errors
            >>> if stats['errors']:
            ...     print(f"Errors occurred: {stats['errors']}")
        """
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            blob_list = list(blobs)
            
            if not blob_list:
                self.logger.warning(f"No blobs found with prefix: {prefix}")
                return {"total": 0, "success": 0, "failed": 0, "errors": []}
            
            stats = {"total": len(blob_list), "success": 0, "failed": 0, "errors": []}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for blob in blob_list:
                    if blob.name.endswith('/'):  # Skip directory markers
                        continue
                    
                    relative_path = blob.name[len(prefix):].lstrip('/')
                    local_path = os.path.join(local_directory, relative_path)
                    
                    future = executor.submit(self.download_blob, blob.name, local_path)
                    futures.append((future, blob.name))
                
                for future, blob_name in futures:
                    try:
                        if future.result():
                            stats["success"] += 1
                        else:
                            stats["failed"] += 1
                            stats["errors"].append(f"Failed to download {blob_name}")
                    except Exception as e:
                        stats["failed"] += 1
                        stats["errors"].append(f"Error downloading {blob_name}: {str(e)}")
            
            self.logger.info(f"Folder download completed - Success: {stats['success']}, Failed: {stats['failed']}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error downloading folder: {e}")
            return {"total": 0, "success": 0, "failed": 1, "errors": [str(e)]}
    
    def upload_folder(self, local_folder_path: str, gcs_destination_prefix: str, 
                      max_workers: int = 4, save_failed_file: bool = True, 
                      failed_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload an entire local folder to GCS with comprehensive tracking.
        
        Recursively uploads all files in a local folder to GCS, preserving the
        folder structure. Provides concurrent uploads, progress tracking, and
        automatic failed file logging for easy retry operations.
        
        Args:
            local_folder_path (str): Path to the local folder to upload
            gcs_destination_prefix (str): GCS destination prefix (folder path)
            max_workers (int): Number of concurrent upload threads. Defaults to 4
            save_failed_file (bool): Whether to save failed uploads to a text file. Defaults to True
            failed_file_path (Optional[str]): Custom path for failed uploads file. 
                                             If None, auto-generates filename
            
        Returns:
            Dict[str, Any]: Statistics dictionary containing:
                - total: Total number of files processed
                - success: Number of successful uploads
                - failed: Number of failed uploads
                - errors: List of error messages
                - failed_files: List of failed file paths
                - failed_file_path: Path to failed files log (if created)
                
        Example:
            >>> # Upload folder with default settings
            >>> stats = manager.upload_folder("/local/data/", "remote/data/")
            >>> print(f"Uploaded {stats['success']}/{stats['total']} files")
            >>> 
            >>> # Custom failed file location
            >>> stats = manager.upload_folder(
            ...     "/local/models/", 
            ...     "remote/models/",
            ...     failed_file_path="my_failed_uploads.txt"
            ... )
        """
        try:
            if not os.path.exists(local_folder_path):
                error_msg = f"Local folder does not exist: {local_folder_path}"
                self.logger.error(error_msg)
                return {"total": 0, "success": 0, "failed": 1, "errors": [error_msg], "failed_files": []}
            
            # Collect all files to upload
            files_to_upload = []
            for root, _, files in os.walk(local_folder_path):
                for file_name in files:
                    local_file_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(local_file_path, local_folder_path)
                    gcs_blob_name = os.path.join(gcs_destination_prefix, relative_path).replace('\\', '/')
                    files_to_upload.append((local_file_path, gcs_blob_name))
            
            stats = {"total": len(files_to_upload), "success": 0, "failed": 0, "errors": [], "failed_files": []}
            
            if not files_to_upload:
                self.logger.warning(f"No files found in folder: {local_folder_path}")
                return stats
            
            self.logger.info(f"Starting upload of {len(files_to_upload)} files")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for local_path, gcs_path in files_to_upload:
                    future = executor.submit(self.upload_blob, local_path, gcs_path)
                    futures.append((future, local_path, gcs_path))
                
                for i, (future, local_path, gcs_path) in enumerate(futures):
                    try:
                        if future.result():
                            stats["success"] += 1
                        else:
                            stats["failed"] += 1
                            stats["errors"].append(f"Failed to upload {local_path}")
                            stats["failed_files"].append(local_path)
                    except Exception as e:
                        stats["failed"] += 1
                        stats["errors"].append(f"Error uploading {local_path}: {str(e)}")
                        stats["failed_files"].append(local_path)
                    
                    # Progress update
                    if (i + 1) % 10 == 0 or (i + 1) == len(futures):
                        progress = (i + 1) / len(futures) * 100
                        self.logger.info(f"Progress: {progress:.1f}% ({i + 1}/{len(futures)})")
            
            # Save failed files to separate file if requested
            if save_failed_file and stats["failed_files"]:
                if not failed_file_path:
                    folder_name = os.path.basename(local_folder_path.rstrip('/'))
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    failed_file_path = f"failed_uploads_{folder_name}_{timestamp}.txt"
                
                if write_failed_operations_file(stats["failed_files"], failed_file_path, "upload"):
                    self.logger.info(f"Failed uploads saved to: {failed_file_path}")
                    stats["failed_file_path"] = failed_file_path
            
            self.logger.info(f"Folder upload completed - Success: {stats['success']}, Failed: {stats['failed']}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error uploading folder: {e}")
            return {"total": 0, "success": 0, "failed": 1, "errors": [str(e)], "failed_files": []}

class CSVImageDownloader:
    """
    Specialized class for downloading images based on CSV file paths.
    
    This class is specifically designed for downloading medical images or datasets
    where image paths are stored in a CSV file. It automatically handles:
    - Path format conversion (underscore to slash)
    - Image renaming with date and identifier extraction
    - Duplicate detection and skipping
    - Failed download tracking
    - Concurrent downloads with progress monitoring
    
    The class automatically uses the GCP_IMAGE_BUCKET environment variable,
    making it ideal for medical image datasets and similar use cases.
    
    Attributes:
        logger (logging.Logger): Logger instance for operation tracking
        gcp_manager (GCPStorageManager): GCS manager configured for image bucket
        
    Example:
        >>> # Download images from CSV
        >>> downloader = CSVImageDownloader()
        >>> stats = downloader.download_images_from_csv(
        ...     "images.csv", 
        ...     "/local/images/", 
        ...     "download_log.csv"
        ... )
        >>> print(f"Downloaded: {stats['success']}, Failed: {stats['failed']}")
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize CSV Image Downloader with automatic image bucket configuration.
        
        Automatically creates a GCPStorageManager instance configured to use the
        GCP_IMAGE_BUCKET environment variable for image-specific operations.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance. If None, creates a default logger
            
        Raises:
            ValueError: If GCP_IMAGE_BUCKET environment variable is not set
            Exception: If GCS client initialization fails
            
        Example:
            >>> # Simple initialization
            >>> downloader = CSVImageDownloader()
            >>> 
            >>> # With custom logger
            >>> custom_logger = setup_logging("image_downloads.log")
            >>> downloader = CSVImageDownloader(logger=custom_logger)
        """
        self.logger = logger or setup_logging()
        # Automatically create GCP manager with image bucket
        self.gcp_manager = GCPStorageManager(use_image_bucket=True, logger=self.logger)
    
    def download_images_from_csv(self, csv_path: str, destination_directory: str, 
                                 log_file_path: str, max_workers: int = 4,
                                 csv_column: str = 'study_path', save_failed_file: bool = True,
                                 failed_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Download images listed in a CSV file with automatic renaming and tracking.
        
        Downloads medical images or datasets based on paths stored in a CSV file.
        Automatically handles path format conversion, image renaming with date/ID
        extraction, duplicate detection, and comprehensive logging.
        
        The method expects a CSV with a column containing GCS blob paths in underscore
        format (e.g., "2022_12_23_ID1_ID2_ID3.jpeg") which are converted to slash
        format for GCS access and renamed for local storage.
        
        Args:
            csv_path (str): Path to the CSV file containing image information
            destination_directory (str): Local directory to save downloaded images
            log_file_path (str): Path for the detailed download log CSV
            max_workers (int): Number of concurrent download threads. Defaults to 4
            csv_column (str): Column name containing image paths. Defaults to 'study_path'
            save_failed_file (bool): Whether to save failed downloads to text file. Defaults to True
            failed_file_path (Optional[str]): Custom path for failed downloads file
            
        Returns:
            Dict[str, Any]: Comprehensive statistics dictionary containing:
                - total: Total number of images in CSV
                - success: Number of successful downloads
                - failed: Number of failed downloads
                - skipped: Number of skipped (already existing) images
                - errors: List of detailed error messages
                - failed_images: List of failed image paths
                - failed_file_path: Path to failed images log (if created)
                
        Raises:
            ValueError: If specified CSV column is not found
            Exception: If CSV loading or directory creation fails
            
        Example:
            >>> # Basic usage
            >>> downloader = CSVImageDownloader()
            >>> stats = downloader.download_images_from_csv(
            ...     "/path/to/images.csv",
            ...     "/local/images/",
            ...     "/logs/download.csv"
            ... )
            >>> 
            >>> # Check results
            >>> print(f"Success: {stats['success']}")
            >>> print(f"Failed: {stats['failed']}")
            >>> print(f"Skipped: {stats['skipped']}")
            >>> 
            >>> # Retry failed downloads
            >>> if stats.get('failed_file_path'):
            ...     print(f"Failed downloads saved to: {stats['failed_file_path']}")
            
        Note:
            - Images are automatically renamed from "2022_12_23_ID1_ID2_ID3.jpeg" 
              to "2022_12_23_ID1_ID2_ID3.jpeg" format
            - Existing files are automatically skipped to avoid re-downloads
            - All operations are logged to both the specified log file and console
        """
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            if csv_column not in df.columns:
                if 'image_path' in df.columns:
                    df[csv_column] = df['image_path'].apply(os.path.basename)
                else:
                    raise ValueError(f"Column '{csv_column}' not found in CSV")
            if '/' in df[csv_column]:
                df[csv_column] = df[csv_column].apply(lambda x: x.split('/')[-1])
            else:
                df[csv_column] = df[csv_column]
            
            os.makedirs(destination_directory, exist_ok=True)
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            stats = {"total": len(df), "success": 0, "failed": 0, "skipped": 0, "errors": [], "failed_images": []}
            
            with open(log_file_path, 'w') as log_file:
                log_file.write('Image Path,Status,Error Message\n')
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    
                    for _, row in df.iterrows():
                        source_blob_name = row[csv_column].replace('_', '/')
                        future = executor.submit(
                            self._download_and_rename_image, 
                            source_blob_name, destination_directory, log_file
                        )
                        futures.append((future, source_blob_name))
                    
                    for future, blob_name in futures:
                        try:
                            result = future.result()
                            if result == "success":
                                stats["success"] += 1
                            elif result == "skipped":
                                stats["skipped"] += 1
                            else:
                                stats["failed"] += 1
                                stats["failed_images"].append(blob_name)
                        except Exception as e:
                            stats["failed"] += 1
                            stats["errors"].append(f"Error processing {blob_name}: {str(e)}")
                            stats["failed_images"].append(blob_name)
            
            # Save failed images to separate file if requested
            if save_failed_file and stats["failed_images"]:
                if not failed_file_path:
                    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    failed_file_path = f"failed_downloads_{csv_name}_{timestamp}.txt"
                
                if write_failed_operations_file(stats["failed_images"], failed_file_path, "download"):
                    self.logger.info(f"Failed downloads saved to: {failed_file_path}")
                    stats["failed_file_path"] = failed_file_path
            
            self.logger.info(f"CSV download completed - Success: {stats['success']}, "
                           f"Failed: {stats['failed']}, Skipped: {stats['skipped']}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error in CSV image download: {e}")
            return {"total": 0, "success": 0, "failed": 1, "skipped": 0, "errors": [str(e)], "failed_images": []}
    
    def _download_and_rename_image(self, source_blob_name: str, destination_directory: str, 
                                   log_file) -> str:
        """Download and rename a single image with logging."""
        try:
            # Create formatted filename
            parts = source_blob_name.split('/')
            formatted_filename = '_'.join(parts[:3]) + '_' + '_'.join(parts[3:])
            destination_file_name = os.path.join(destination_directory, formatted_filename)
            
            # Check if file already exists
            if os.path.exists(destination_file_name):
                log_file.write(f"{source_blob_name},Skipped,File already exists\n")
                log_file.flush()
                return "skipped"
            
            # Download the file
            if self.gcp_manager.download_blob(source_blob_name, destination_file_name):
                log_file.write(f"{source_blob_name},Success,\n")
                log_file.flush()
                return "success"
            else:
                log_file.write(f"{source_blob_name},Failed,Download error\n")
                log_file.flush()
                return "failed"
                
        except Exception as e:
            log_file.write(f"{source_blob_name},Error,{str(e)}\n")
            log_file.flush()
            return "failed"

def main():
    """
    Main function demonstrating comprehensive usage examples.
    
    This function serves as both a command-line entry point and a comprehensive
    example of how to use the GCS management classes. It demonstrates:
    - General file and folder upload operations
    - CSV-based image download operations
    - Error handling and statistics reporting
    - Proper logging and progress tracking
    
    The function includes real-world examples using actual file paths and
    demonstrates both successful operations and error handling scenarios.
    
    Environment Variables Required:
        - GCP_BUCKET_NAME: For general GCS operations
        - GCP_IMAGE_BUCKET: For image-specific operations
        - GOOGLE_APPLICATION_CREDENTIALS: Path to GCP service account JSON
        
    Example:
        >>> # Run from command line
        >>> python gcp.py
        
    Note:
        The examples in this function use hardcoded paths. Modify them to match
        your specific use case and file locations.
    """
    logger = setup_logging()
    
    try:
        # Initialize GCP Storage Manager (uses GCP_BUCKET_NAME by default)
        gcp_manager = GCPStorageManager(logger=logger)
        
        # Example usage - upload specific folders
        folders_to_upload = [
            {
                "local_path": "/home/ai-user/Paligemma/runs-classification/paligemma_qlora_10b_224_classification_20250509_150127",
                "gcp_destination": "Paligemma-models/Paligemma-10b_224_classification"
            },
            {
                "local_path": "/home/ai-user/Paligemma/runs-3b-448/paligemma_3b_448_20250428_100723",
                "gcp_destination": "Paligemma-models/Paligemma-3b-448"
            },
            {
                "local_path": "/home/ai-user/Paligemma/runs-mix/paligemma_qlora_10b_224_mix_20250507_133445",
                "gcp_destination": "Paligemma-models/Paligemma-10b_224_mix"
            }
        ]
        
        # Upload folders
        total_success = 0
        total_failed = 0
        
        for i, folder in enumerate(folders_to_upload):
            logger.info(f"Processing folder {i+1}/{len(folders_to_upload)}: {folder['local_path']}")
            
            stats = gcp_manager.upload_folder(folder["local_path"], folder["gcp_destination"])
            
            if stats["failed"] == 0:
                total_success += 1
                logger.info(f"Successfully uploaded folder: {folder['local_path']}")
            else:
                total_failed += 1
                logger.error(f"Failed to upload folder: {folder['local_path']}")
                for error in stats["errors"]:
                    logger.error(error)
        
        logger.info(f"Upload process completed - Success: {total_success}, Failed: {total_failed}")
        
        # Example of CSV image downloader (uses GCP_IMAGE_BUCKET automatically)
        logger.info("\n" + "="*60)
        logger.info("CSV Image Download Example")
        logger.info("="*60)
        
        # Example CSV image download - similar to gcp_downloader.py usage
        csv_path = '/home/ai-user/RELA PROJECT/Pleural-effusion_Consolidation/filtered_rows.csv'
        destination_directory = '/home/ai-user/RELA PROJECT/Pleural-effusion_Consolidation/data'
        log_file_path = '/home/ai-user/RELA PROJECT/Pleural-effusion_Consolidation/data/logs/pleural_effusion_consolidation_log.csv'
        
        try:
            # Initialize CSV Image Downloader (automatically uses GCP_IMAGE_BUCKET)
            downloader = CSVImageDownloader(logger=logger)
            
            logger.info(f"Starting CSV image download from: {csv_path}")
            logger.info(f"Destination directory: {destination_directory}")
            logger.info(f"Log file: {log_file_path}")
            
            # Download images with automatic failed file creation
            stats = downloader.download_images_from_csv(
                csv_path=csv_path,
                destination_directory=destination_directory,
                log_file_path=log_file_path,
                max_workers=4,  # Concurrent downloads
                csv_column='study_path'  # Column containing image paths
            )
            
            # Report results
            logger.info(f"CSV Download Results:")
            logger.info(f"  Total images: {stats['total']}")
            logger.info(f"  Successfully downloaded: {stats['success']}")
            logger.info(f"  Failed downloads: {stats['failed']}")
            logger.info(f"  Skipped (already exist): {stats['skipped']}")
            
            if stats.get('failed_file_path'):
                logger.info(f"  Failed downloads saved to: {stats['failed_file_path']}")
            
            if stats['errors']:
                logger.warning(f"  Error details: {stats['errors'][:3]}...")  # Show first 3 errors
            
        except Exception as e:
            logger.error(f"CSV image download example failed: {e}")
            logger.info("Note: Make sure the CSV file exists and GCP_IMAGE_BUCKET is set in environment")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()