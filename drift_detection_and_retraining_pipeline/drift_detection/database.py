from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from datetime import datetime, timezone
import logging
import time

logger = logging.getLogger(__name__)

class DriftDatabase:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None  # Use existing images collection
        self._connected = False
        self.MONGO_URI = "mongodb://localhost:27018/"
        self.DB_NAME = "biosight"
        self.COLLECTION_NAME = "images"  # Use existing images collection

    def connect(self, max_retries=3, retry_delay=5):
        """Connect to MongoDB with retries"""
        if self._connected:
            return True

        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to MongoDB at {self.MONGO_URI}...")
                self.client = MongoClient(
                    self.MONGO_URI,
                    serverSelectionTimeoutMS=20000,
                    connectTimeoutMS=20000,
                    retryWrites=True
                )
                self.client.admin.command('ping')
                self.db = self.client[self.DB_NAME]
                self.collection = self.db[self.COLLECTION_NAME]
                self._connected = True
                logger.info("Successfully connected to MongoDB")
                return True
            except (PyMongoError, ServerSelectionTimeoutError) as e:
                logger.error(f"Error connecting to MongoDB: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Could not connect to MongoDB.")
                    return False

    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            self._connected = False

    def update_drift_status(self, filename, drift_detected):
        """Update drift detection status for an image."""
        try:
            result = self.collection.update_one(
                {'original_filename': filename},
                {
                    '$set': {
                        'drift_detected': drift_detected,
                        'drift_checked_at': datetime.now(timezone.utc)
                    }
                }
            )
            if result.modified_count > 0:
                logger.info(f"Updated drift status for image: {filename}")
                return True
            else:
                logger.warning(f"No image found with filename: {filename}")
                return False
        except Exception as e:
            logger.error(f"Error updating drift status: {e}")
            return False

    def get_drifted_images(self, limit=100):
        """Get list of images with detected drift that haven't been used in training."""
        try:
            results = list(self.collection.find(
                {
                    'drift_detected': True,
                    'used_in_training': False  # Added this condition
                },
                {
                    'original_filename': 1,
                    'predicted_class': 1,
                    'drift_checked_at': 1,
                    'used_in_training': 1,  # Added this field to return
                    '_id': 0
                }
            ).sort('drift_checked_at', -1).limit(limit))
            return results
        except Exception as e:
            logger.error(f"Error fetching drifted images: {e}")
            return []

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    db = DriftDatabase()
    if db.connect():
        # Example: Update drift status
        db.update_drift_status(
            filename="example.jpg",
            drift_detected=True
        )
        
        # Example: Get drifted images
        drifted = db.get_drifted_images(limit=5)
        for img in drifted:
            print(f"Drift detected in: {img['original_filename']}")
        
        db.close()