"""
Quick user registration script for testing.
Run this ONCE to create your user account.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from db.database import SessionLocal, engine, Base
from db.models import User
from db.crud import hash_password

# Ensure tables exist
Base.metadata.create_all(bind=engine)

def register_user(username: str, password: str):
    """Create a new user in the database."""
    db = SessionLocal()
    try:
        # Check if user already exists
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            print(f"❌ User '{username}' already exists!")
            return False
        
        # Create new user
        hashed_pwd = hash_password(password)
        new_user = User(username=username, hashed_password=hashed_pwd)
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        print(f"✅ User '{username}' created successfully!")
        print(f"   ID: {new_user.id}")
        print(f"   Username: {new_user.username}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    # Get credentials from command line or use defaults for testing
    if len(sys.argv) >= 3:
        username = sys.argv[1]
        password = sys.argv[2]
    else:
        print("Usage: python register_user.py <username> <password>")
        print("\nExample: python register_user.py testuser testpass123")
        print("\nOr use defaults below:")
        username = "testuser"
        password = "testpass123"
    
    register_user(username, password)
