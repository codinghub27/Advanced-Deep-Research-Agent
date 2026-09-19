from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,DeclarativeBase
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env")

class Base(DeclarativeBase):
    pass

engine=create_engine(
    url=DATABASE_URL,
    #echo=True ## you'll see SQL statements in your terminal.
)

SessionLocal=sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()


###Engine = Database communication infrastructure

###Session = Your database workspace