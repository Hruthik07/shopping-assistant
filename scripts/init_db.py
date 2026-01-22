"""Database initialization script."""
import sys
import io
from pathlib import Path

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.db import init_db, engine
from src.database.models import Base, PriceHistory
from src.analytics.logger import logger
from sqlalchemy import inspect


def check_table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def initialize_database():
    """Initialize database with all tables."""
    try:
        logger.info("Initializing database...")
        
        # Create all tables using SQLAlchemy
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
        
        # Verify price_history table exists
        if check_table_exists("price_history"):
            logger.info("[OK] price_history table exists")
        else:
            logger.warning("[WARN] price_history table not found, creating...")
            # Run migration if needed
            try:
                from src.database.migrations import upgrade
                upgrade()
            except Exception as e:
                logger.error(f"Migration failed: {e}")
                # Fallback: create directly
                Base.metadata.create_all(bind=engine)
        
        # Verify all expected tables
        expected_tables = ["users", "sessions", "conversations", "user_preferences", 
                          "cart_items", "price_history"]
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        logger.info("\nDatabase tables:")
        for table in expected_tables:
            if table in existing_tables:
                logger.info(f"  [OK] {table}")
            else:
                logger.warning(f"  [WARN] {table} (missing)")
        
        logger.info("\n[SUCCESS] Database initialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = initialize_database()
    sys.exit(0 if success else 1)
