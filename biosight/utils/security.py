from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import OAuth2, OAuth2PasswordBearer
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from ..db import db
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
        auto_error: bool = True,
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
                return None

        # Extract token from "Bearer {token}"
        parts = token.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token_value = parts[1]
        elif len(parts) == 1:
            # Allow plain token if no Bearer prefix
            token_value = parts[0]
        else:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token format in cookie",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                return None
        
        return token_value

# Create instance of our cookie scheme
oauth2_cookie_scheme = OAuth2PasswordBearerCookie(tokenUrl="/api/login")

# User dependency
async def get_current_user(token: str = Depends(oauth2_cookie_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error during user authentication: {str(e)}"
        )
    
    if user_dict is None:
        raise credentials_exception
    
    # Convert ObjectId to string and prepare for User model
    if "_id" in user_dict:
        user_dict["id"] = str(user_dict["_id"])
    
    return User(**user_dict)