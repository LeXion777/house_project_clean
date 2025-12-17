# app/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

class Config:
    # 메인 DB
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{(BASE_DIR / 'house.db').as_posix()}"


    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "dev"







