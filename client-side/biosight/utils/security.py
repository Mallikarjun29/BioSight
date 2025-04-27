from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import OAuth2, OAuth2PasswordBearer
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from biosight.utils.database import db  # Fixed import path
from ..routes.user import User

# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:  # Handle case where user might not have a password hash
        return False
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# JWT Token handling
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Cookie authentication scheme
class OAuth2PasswordBearerCookie(OAuth2):
    def __init__(
        self,
        tokenUrl: str,
        scheme_name: str = "OAuth2PasswordBearerCookie",
        scopes: Optional[Dict[str, str]] = None,
        auto_error: bool = True, # Keep auto_error=True for the main scheme
    ):
        if not scopes:
            scopes = {}
        flows = OAuthFlowsModel(password={"tokenUrl": tokenUrl, "scopes": scopes})
        super().__init__(flows=flows, scheme_name=scheme_name, auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        token = request.cookies.get("access_token")  # Read from cookie
        
        if not token:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                return None # Return None if auto_error is False and token is missing

        # Extract token from "Bearer {token}"
        parts = token.split()
        token_value: Optional[str] = None
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token_value = parts[1]
        elif len(parts) == 1:
            # Allow plain token if no Bearer prefix
            token_value = parts[0]
        
        if not token_value: # Check if token_value was successfully extracted
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token format in cookie",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                return None # Return None if auto_error is False and format is invalid
        
        return token_value

# Create instance of our cookie scheme for required authentication
# Ensure the tokenUrl matches your actual token endpoint (e.g., /api/login or /api/token)
oauth2_cookie_scheme_required = OAuth2PasswordBearerCookie(tokenUrl="/api/login", auto_error=True)

# Create instance of our cookie scheme for optional authentication
oauth2_cookie_scheme_optional = OAuth2PasswordBearerCookie(tokenUrl="/api/login", auto_error=False)


# User dependency (strict - raises 401 if not authenticated)
async def get_current_user(token: str = Depends(oauth2_cookie_scheme_required)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Token presence is already handled by Depends(oauth2_cookie_scheme_required)
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Use email as the subject 'sub' based on your auth logic
        email: Optional[str] = payload.get("sub") 
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Ensure database connection is active
    if db is None or not hasattr(db, 'users_collection') or db.users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is not available"
        )
    
    try:
        user_dict = db.users_collection.find_one({"email": email})
    except Exception as e:
        # Log the error for debugging
        # logger.error(f"Database error during user lookup: {e}") 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error during user authentication."
        )
    
    if user_dict is None:
        raise credentials_exception
    
    # Convert ObjectId to string and prepare for User model
    if "_id" in user_dict:
        user_dict["id"] = str(user_dict["_id"])
        # Ensure all required fields for User model are present
        # You might need to adjust this based on your User model definition
        required_fields = User.__fields__.keys()
        for field in required_fields:
            if field not in user_dict and field != 'id': 
                 # Handle missing fields appropriately, maybe raise error or provide default
                 # For now, let's assume User model can handle potential missing fields or has defaults
                 pass 

    try:
       return User(**user_dict)
    except Exception as e: # Catch potential validation errors if User is a Pydantic model
        # Log the error
        # logger.error(f"Error creating User model instance: {e}")
        raise credentials_exception # Treat model validation error as credential issue


# Optional user dependency (returns None if not authenticated)
async def get_current_user_optional(token: Optional[str] = Depends(oauth2_cookie_scheme_optional)) -> Optional[User]:
    if token is None:
        # No token found by the optional scheme
        return None
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Use email as the subject 'sub' based on your auth logic
        email: Optional[str] = payload.get("sub") 
        if email is None:
            return None # Invalid payload
    except JWTError:
        return None # Invalid token
    
    # Ensure database connection is active
    if db is None or not hasattr(db, 'users_collection') or db.users_collection is None:
        # Log this issue, but don't necessarily block anonymous access if DB is down
        # logger.error("Database connection not available for optional user check")
        return None # Treat DB error as unauthenticated for optional check
    
    try:
        user_dict = db.users_collection.find_one({"email": email})
    except Exception as e:
        # Log the error
        # logger.error(f"Database error during optional user lookup: {e}")
        return None # Treat DB error as unauthenticated
    
    if user_dict is None:
        return None # User not found
    
    # Convert ObjectId to string and prepare for User model
    if "_id" in user_dict:
        user_dict["id"] = str(user_dict["_id"])
        # Ensure all required fields for User model are present
        required_fields = User.__fields__.keys()
        for field in required_fields:
             if field not in user_dict and field != 'id':
                 # Handle missing fields if necessary
                 pass

    try:
       return User(**user_dict)
    except Exception as e: # Catch potential validation errors
        # Log the error
        # logger.error(f"Error creating User model instance for optional user: {e}")
        return None # Treat model validation error as unauthenticated