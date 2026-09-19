from jose import jwt,JWTError
from fastapi import Depends,HTTPException,status
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime,timedelta,timezone
from dotenv import load_dotenv
import os

load_dotenv()


oauth2_scheme=OAuth2PasswordBearer(tokenUrl='/auth/login')
SECRET_KEY=os.getenv('SECRET_KEY')
ALGORITHM=os.getenv('ALGORITHM','HS256')


def create_access_token(data:dict)->str:
    payload=data.copy()
    expire=datetime.now(timezone.utc)+timedelta(minutes=30)
    payload['exp']=expire
    token=jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return token



def get_curr_user(token:str=Depends(oauth2_scheme)):
    try:
        payload=jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        username=payload['sub']
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Token"
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def require_admin(user:str=Depends(get_curr_user))->str:
    # Admins are listed in ADMIN_USERNAMES (comma-separated). Unset/empty means
    # nobody is an admin, so /admin/* fails closed.
    admins={u.strip() for u in os.getenv('ADMIN_USERNAMES','').split(',') if u.strip()}
    if user not in admins:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return user
