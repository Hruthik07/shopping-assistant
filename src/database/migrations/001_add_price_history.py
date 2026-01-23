"""Migration script to add price_history table and indexes."""

from sqlalchemy import text
from src.database.db import engine
from src.analytics.logger import logger


def upgrade():
    """Create price_history table and indexes."""
    with engine.begin() as conn:
        # Check if table already exists
        result = conn.execute(text("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='price_history'
        """))

        if result.fetchone():
            logger.info("price_history table already exists, skipping creation")
            return

        # Create price_history table (SQLite-compatible)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                product_name TEXT,
                retailer TEXT,
                price REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                shipping_cost REAL DEFAULT 0.0,
                total_cost REAL NOT NULL,
                original_price REAL,
                discount_amount REAL DEFAULT 0.0,
                discount_percent REAL DEFAULT 0.0,
                url TEXT,
                in_stock INTEGER DEFAULT 1,
                availability INTEGER DEFAULT 1,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                upc TEXT,
                gtin TEXT,
                ean TEXT,
                sku TEXT,
                product_metadata TEXT
            )
        """))

        # Create indexes for performance (with IF NOT EXISTS for idempotency)
        indexes = [
            ("idx_price_history_product_id", "product_id"),
            ("idx_price_history_timestamp", "timestamp"),
            ("idx_price_history_retailer", "retailer"),
            ("idx_price_history_upc", "upc"),
            ("idx_price_history_gtin", "gtin"),
            ("idx_price_history_ean", "ean"),
            ("idx_price_history_sku", "sku"),
            ("idx_price_history_product_name", "product_name"),
            ("idx_price_history_product_timestamp", "product_id, timestamp"),
        ]

        for idx_name, idx_columns in indexes:
            try:
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name} ON price_history({idx_columns})
                """))
            except Exception as e:
                # Index might already exist, continue
                logger.debug(f"Index {idx_name} creation: {e}")

        logger.info("Successfully created price_history table and indexes")


def downgrade():
    """Drop price_history table."""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS price_history"))
        logger.info("Dropped price_history table")


if __name__ == "__main__":
    upgrade()
