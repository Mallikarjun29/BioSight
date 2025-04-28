#!/usr/bin/env python3

"""
Fetch drifted images from MongoDB database.

This script connects to the MongoDB database and retrieves images 
that have been detected as having drift but haven't been used in training yet.
"""

import logging
import argparse
from database import DriftDatabase
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def fetch_drifted_images(limit=100, display_details=True):
    """
    Connect to database and fetch images with detected drift.
    
    Args:
        limit (int): Maximum number of images to fetch
        display_details (bool): Whether to print details of each image
    
    Returns:
        list: List of drifted image records
    """
    db = DriftDatabase()
    drifted_images = []
    
    try:
        # Connect to the database
        logger.info("Connecting to the database...")
        if not db.connect():
            logger.error("Failed to connect to the database.")
            return []
            
        # Fetch drifted images
        logger.info(f"Fetching up to {limit} drifted images...")
        drifted_images = db.get_drifted_images(limit=limit)
        
        # Display results
        if display_details:
            print(f"\n{'-'*50}")
            print(f"Found {len(drifted_images)} drifted images")
            print(f"{'-'*50}")
            
            for i, image in enumerate(drifted_images, 1):
                print(f"{i}. File: {image['original_filename']}")
                print(f"   Class: {image['predicted_class']}")
                if 'drift_checked_at' in image:
                    drift_time = image['drift_checked_at'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(image['drift_checked_at'], datetime) else image['drift_checked_at']
                    print(f"   Detected at: {drift_time}")
                print(f"   Used in training: {image.get('used_in_training', False)}")
                print()
                
        logger.info(f"Successfully retrieved {len(drifted_images)} images with drift.")
        
    except Exception as e:
        logger.error(f"Error fetching drifted images: {str(e)}")
    finally:
        # Close the database connection
        logger.info("Closing database connection...")
        db.close()
    
    return drifted_images


def main():
    """Main function to parse arguments and fetch drifted images."""
    parser = argparse.ArgumentParser(description='Fetch images with detected drift from the database')
    parser.add_argument('-l', '--limit', type=int, default=100,
                        help='Maximum number of images to fetch (default: 100)')
    parser.add_argument('--no-details', action='store_true',
                        help='Do not display detailed information for each image')
    
    args = parser.parse_args()
    
    # Fetch and display images
    drifted_images = fetch_drifted_images(
        limit=args.limit,
        display_details=not args.no_details
    )
    
    return len(drifted_images)


if __name__ == "__main__":
    main()