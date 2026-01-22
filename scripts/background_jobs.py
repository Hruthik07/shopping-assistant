"""Background jobs for price tracking and coupon refresh."""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.price_tracker import price_tracker
from src.database.db import get_db
from src.database.models import PriceHistory
from src.analytics.logger import logger


async def cleanup_old_price_history(days: int = 90):
    """Clean up old price history records.
    
    Args:
        days: Retention period in days (default: 90)
    """
    try:
        logger.info(f"Starting price history cleanup (retention: {days} days)...")
        deleted = await price_tracker.cleanup_old_records(days=days)
        logger.info(f"Price history cleanup complete: {deleted} records deleted")
        return deleted
    except Exception as e:
        logger.error(f"Error during price history cleanup: {e}", exc_info=True)
        return 0


async def get_products_needing_tracking():
    """Get list of product IDs that need price tracking updates.
    
    Returns:
        List of product IDs that haven't been tracked recently
    """
    try:
        db = next(get_db())
        try:
            # Get products that haven't been tracked in the last 24 hours
            cutoff = datetime.utcnow() - timedelta(hours=24)
            
            # Get distinct product IDs that need updates
            recent_products = db.query(PriceHistory.product_id).filter(
                PriceHistory.timestamp >= cutoff
            ).distinct().all()
            
            recent_ids = {row[0] for row in recent_products}
            
            # Get all distinct product IDs
            all_products = db.query(PriceHistory.product_id).distinct().all()
            all_ids = {row[0] for row in all_products}
            
            # Products that need tracking (tracked before but not recently)
            needs_tracking = all_ids - recent_ids
            
            logger.info(f"Found {len(needs_tracking)} products needing price tracking updates")
            return list(needs_tracking)
            
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error getting products needing tracking: {e}", exc_info=True)
        return []


async def run_background_jobs():
    """Run all background jobs."""
    logger.info("=" * 60)
    logger.info("Starting background jobs")
    logger.info("=" * 60)
    
    # Job 1: Cleanup old price history
    logger.info("\n[Job 1] Cleaning up old price history...")
    deleted = await cleanup_old_price_history(days=90)
    logger.info(f"[Job 1] Complete: {deleted} records deleted")
    
    # Job 2: Identify products needing tracking
    logger.info("\n[Job 2] Identifying products needing price tracking...")
    products_to_track = await get_products_needing_tracking()
    logger.info(f"[Job 2] Complete: {len(products_to_track)} products need tracking")
    
    logger.info("\n" + "=" * 60)
    logger.info("Background jobs complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(run_background_jobs())
    except KeyboardInterrupt:
        logger.info("\nBackground jobs interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Background jobs failed: {e}", exc_info=True)
        sys.exit(1)
