from fastapi import APIRouter, HTTPException, Depends, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, EmailStr
from .user import User, UserCreate
from ..utils.database import db  # Fixed import path
from ..utils.security import verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_password_hash

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = "your-secret-key"  # Change this to a secure secret key
ALGORITHM = "HS256"

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

@router.post("/api/register", response_model=User)
async def register(user_data: UserCreate):
    # Ensure database connection is active
    if db is None or not hasattr(db, 'users_collection') or db.users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is not available"
        )
        
    existing_user = db.users_collection.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    hashed_password = get_password_hash(user_data.password)
    user_dict = user_data.model_dump()
    user_dict["hashed_password"] = hashed_password
    del user_dict["password"]  # Don't store plain password

    # Add created_at and is_active fields
    user_dict["created_at"] = datetime.utcnow()
    user_dict["is_active"] = True

    try:
        # Insert into database
        inserted_result = db.users_collection.insert_one(user_dict)
        created_user = db.users_collection.find_one({"_id": inserted_result.inserted_id})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error during user registration: {str(e)}"
        )

    # Convert ObjectId to string if necessary for the User model
    if created_user and "_id" in created_user:
        created_user["id"] = str(created_user["_id"])

    if not created_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create user"
        )

    return User(**created_user)

@router.post("/api/login")
async def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    # Ensure database connection is active
    if db is None or not hasattr(db, 'users_collection') or db.users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is not available"
        )
        
    try:
        user_dict = db.users_collection.find_one({"email": form_data.username})  # Find user by email (passed as username)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error during login: {str(e)}"
        )
        
    if not user_dict:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    if not verify_password(form_data.password, user_dict.get("hashed_password")):  # Use verify_password util
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_dict["email"]},  # Use email from found user dict
        expires_delta=access_token_expires
    )

    # Set cookie in the response
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",  # Include "Bearer " prefix
        httponly=True,  # Crucial for security
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # In seconds
        expires=access_token_expires,  # Can set specific expiry time
        samesite="Lax",  # Good default, consider "Strict"
        path="/"  # Ensure cookie is valid for all paths
    )

    # Return a success message instead of the token
    return {"message": "Login successful"}

@router.post("/api/logout")
async def logout(response: Response):
    # Clear the cookie by setting it with an expired date/max_age 0
    response.set_cookie(
        key="access_token",
        value="",
        httponly=True,
        max_age=0,
        expires=0,
        samesite="Lax",
        path="/"
    )
    return {"message": "Logout successful"}