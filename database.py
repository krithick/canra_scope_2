from mongo import MongoDB
import os
from dotenv import load_dotenv

load_dotenv(".env")

MONGO_URL = os.getenv("MONGO_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")
db = MongoDB(MONGO_URL,DATABASE_NAME)

async def get_db():
    return db