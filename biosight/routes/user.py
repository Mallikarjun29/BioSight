from pydantic import BaseModel, EmailStr, Field
from typing import Optional
import bcrypt
from datetime import datetime
from ..utils.database import db

class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    hashed_password: str
    created_at: datetime = datetime.utcnow()
    is_active: bool = True
    id: Optional[str] = None

    @classmethod
    def create(cls, email: str, password: str, name: str):
        # Generate salt and hash password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        return cls(
            email=email,
            name=name,
            hashed_password=hashed_password
        )

    def verify_password(self, password: str) -> bool:
        # Verify password
        return bcrypt.checkpw(
            password.encode('utf-8'),
            self.hashed_password.encode('utf-8')
        )

    @staticmethod
    def delete_all_users():
        """Delete all users from the database."""
        try:
            result = db.collection.delete_many({})
            return result.deleted_count
        except Exception as e:
            print(f"Error deleting users: {e}")
            return 0
            
    class Config:
        from_attributes = True
        populate_by_name = True