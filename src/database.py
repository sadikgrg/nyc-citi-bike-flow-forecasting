import getpass
import os

from sqlalchemy import URL, create_engine

def get_database_url():
    if database_url := os.getenv("DATABASE_URL"):
        return database_url

    return URL.create(
        "postgresql",
        username=os.getenv("PGUSER", getpass.getuser()),
        password=os.getenv("PGPASSWORD"),
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        database=os.getenv("PGDATABASE", "citibike")
    )

def get_engine():
    return create_engine(get_database_url(), pool_size=5, echo=False)