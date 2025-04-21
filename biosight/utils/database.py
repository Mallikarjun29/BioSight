"""MongoDB database operations."""
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from datetime import datetime, timezone
import logging
import time

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.users_collection = None
        self._connected = False
        self.MONGO_URI = "mongodb://mongodb:27017/"
        self.DB_NAME = "biosight"
        self.COLLECTION_NAME = "images"
        self.USERS_COLLECTION_NAME = "users"

    def connect(self, max_retries=3, retry_delay=5):
        """Connect to MongoDB with retries"""
        if self._connected:
            return True

        for attempt in range(max_retries):
            try:
                print(f"Connecting to MongoDB at {self.MONGO_URI}...")
                self.client = MongoClient(
                    self.MONGO_URI,
                    serverSelectionTimeoutMS=20000,
                    connectTimeoutMS=20000,
                    retryWrites=True
                )
                # Test the connection
                self.client.admin.command('ping')
                self.db = self.client[self.DB_NAME]
                # Initialize both collections
                self.collection = self.db[self.COLLECTION_NAME]
                self.users_collection = self.db[self.USERS_COLLECTION_NAME]
                self._connected = True
                print("Successfully connected to MongoDB")
                return True
            except (PyMongoError, ServerSelectionTimeoutError) as e:
                print(f"Error connecting to MongoDB: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Could not connect to MongoDB.")
                    return False

    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            self.users_collection = None
            self._connected = False

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
            if not self._connected:
                return "error: not connected"
            self.client.admin.command('ping')
            return "ok"
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return f"error: {str(e)}"

# Initialize database connection
db = Database()