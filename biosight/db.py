"""MongoDB database operations."""
import os
from pymongo import MongoClient
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.users_collection = None
        # Initialize connection immediately
        self.connect()

    def connect(self):
        """Establish connection to MongoDB."""
        try:
            # Get connection details from environment or use defaults
            mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")
            db_name = os.getenv('MONGO_DB', "biosight")
            collection_name = os.getenv('MONGO_COLLECTION', "images")
            users_collection_name = "users"
            
            print(f"Connecting to MongoDB at {mongo_uri}...")
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.users_collection = self.db[users_collection_name]
            
            self.client.admin.command('ping')
            print("MongoDB connection successful")
            return True
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            return False

    def save_metadata(self, metadata):
        """Save image metadata to MongoDB."""
        try:
            # Ensure connection is established
            if self.collection is None:
                self.connect()
                
            self.collection.insert_one(metadata)
            return True
        except Exception as e:
            logger.error(f"Error saving metadata to database: {e}")
            return False

    def check_health(self):
        """Check MongoDB connection health."""
        try:
            if self.client is None:
                return "not connected"
            self.client.admin.command('ping')
            return "ok"
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return "error"

    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            self.users_collection = None

# Initialize database
db = Database()