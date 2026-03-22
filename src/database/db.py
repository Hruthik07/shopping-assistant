"""Database connection and session management."""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.utils.config import settings

_is_sqlite = "sqlite" in settings.database_url.lower()

if _is_sqlite:
    # SQLite – minimal config for local development
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
    )
else:
    # PostgreSQL / any other production DB – full connection-pool config
    engine = create_engine(
        settings.database_url,
        pool_size=5,           # number of persistent connections
        max_overflow=10,       # extra connections allowed above pool_size
        pool_timeout=30,       # seconds to wait for a connection before raising
        pool_recycle=1800,     # recycle connections after 30 min (keeps RDS happy)
        pool_pre_ping=True,    # test every connection before handing it out
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Run Alembic migrations to bring the database schema up to date.

    Using Alembic (instead of ``Base.metadata.create_all``) means every schema
    change is version-controlled and can be rolled back safely on RDS / Aurora.
    Falls back to ``create_all`` only for SQLite in non-production environments
    so that local dev still works without a running Alembic setup.
    """
    from src.utils.config import settings

    if _is_sqlite:
        # Dev shortcut – keep SQLite working without Alembic installed
        Base.metadata.create_all(bind=engine)
        return

    try:
        from alembic.config import Config as AlembicConfig
        from alembic import command as alembic_command
        import os

        # Locate alembic.ini relative to this file (project root)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        alembic_cfg = AlembicConfig(os.path.join(project_root, "alembic.ini"))
        alembic_cfg.set_main_option("sqlalchemy.url", settings.database_url)
        alembic_command.upgrade(alembic_cfg, "head")
    except Exception as exc:
        # Surface the error clearly rather than silently applying create_all
        raise RuntimeError(f"Alembic migration failed: {exc}") from exc
