"""MongoDB database operations."""
from pymongo import MongoClient
from datetime import datetime, timezone
import logging
from .config import MONGO_URI, DB_NAME, COLLECTION_NAME

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.connect()

    def connect(self):
        """Establish connection to MongoDB."""
        try:
            print(f"Connecting to MongoDB at {MONGO_URI}...")
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            # Test connection
            self.client.admin.command('ping')
            print("MongoDB connection successful")
            return True
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            return False

    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None

    def save_metadata(self, metadata):
        """Save image metadata to MongoDB."""
        try:
            self.collection.insert_one(metadata)
            return True
        except Exception as e:
            logger.error(f"Error saving metadata to database: {e}")
            return False

    def check_health(self):
        """Check MongoDB connection health."""
        try:
            self.client.admin.command('ping')
            return "ok"
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return "error"

# Initialize database connection
db = Database()