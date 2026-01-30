"""Scheduler jobs for BabelByte."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import get_settings
from src.delivery.email_sender import EmailSender
from src.fetchers.reddit import RedditFetcher
from src.fetchers.twitter import TwitterFetcher, MockTwitterFetcher
from src.processors.digest_processor import DigestGenerator
from src.storage.database import Database, SyncDatabase
from src.storage.models import SourceType

logger = logging.getLogger(__name__)


class JobRunner:
    """Runner for scheduled jobs."""

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self._db: Optional[SyncDatabase] = None

    def get_db(self) -> SyncDatabase:
        """Get or create database connection."""
        if self._db is None:
            self._db = SyncDatabase()
            self._db.connect()
        return self._db

    def close_db(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None

    def fetch_all_content(self) -> dict:
        """
        Fetch content from all enabled subscriptions.

        Returns:
            Dict with fetch statistics.
        """
        logger.info("Starting content fetch job...")
        db = self.get_db()

        subscriptions = db.list_subscriptions(enabled_only=True)
        if not subscriptions:
            logger.info("No enabled subscriptions found")
            return {"total": 0, "fetched": 0, "errors": 0}

        # Initialize fetchers
        reddit_fetcher = RedditFetcher()
        settings = get_settings()

        if self.use_mock or not settings.twitter.is_configured:
            twitter_fetcher = MockTwitterFetcher()
        else:
            twitter_fetcher = TwitterFetcher()

        stats = {"total": len(subscriptions), "fetched": 0, "errors": 0, "new_items": 0}

        for sub in subscriptions:
            try:
                # Choose appropriate fetcher
                if sub.source_type == SourceType.REDDIT:
                    fetcher = reddit_fetcher
                else:
                    fetcher = twitter_fetcher

                # Run async fetch in sync context
                result = asyncio.get_event_loop().run_until_complete(fetcher.fetch(sub))

                if result.success:
                    # Store new items
                    new_count = 0
                    skipped_url = 0
                    for item in result.items:
                        # Skip if external_id already exists
                        if db.content_exists(item.source_type, item.external_id):
                            continue
                        # Skip if URL already exists (cross-source deduplication)
                        if db.url_exists(item.url):
                            skipped_url += 1
                            logger.debug(f"Skipping duplicate URL: {item.url[:60]}...")
                            continue
                        db.add_content_item(item)
                        new_count += 1
                    if skipped_url > 0:
                        logger.info(f"Skipped {skipped_url} items with duplicate URLs")

                    # Update subscription's last fetched time
                    sub.last_fetched_at = datetime.now()
                    db.update_subscription(sub)

                    stats["fetched"] += 1
                    stats["new_items"] += new_count
                    logger.info(f"Fetched {sub.display_name}: {new_count} new items")
                else:
                    stats["errors"] += 1
                    logger.error(f"Failed to fetch {sub.display_name}: {result.error_message}")

            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Error fetching {sub.display_name}: {e}")

        logger.info(
            f"Fetch job completed: {stats['fetched']}/{stats['total']} successful, "
            f"{stats['new_items']} new items"
        )
        return stats

    def process_content(self, limit: int = 50) -> int:
        """
        Process unprocessed content with AI.

        Args:
            limit: Maximum items to process.

        Returns:
            Number of items processed.
        """
        logger.info("Starting content processing job...")
        db = self.get_db()

        generator = DigestGenerator(db=db, use_mock=self.use_mock)
        processed = generator.process_unprocessed_items(limit=limit)

        logger.info(f"Processing job completed: {processed} items processed")
        return processed

    def send_digest(self, dry_run: bool = False) -> dict:
        """
        Generate and send the daily digest.

        Args:
            dry_run: If True, don't actually send or mark as delivered.

        Returns:
            Dict with digest statistics.
        """
        logger.info("Starting digest job...")
        db = self.get_db()

        generator = DigestGenerator(db=db, use_mock=self.use_mock)

        # First, process any unprocessed items
        self.process_content()

        # Generate digest
        digest = generator.generate_digest()

        stats = {
            "items": len(digest.items),
            "categories": len(digest.by_category),
            "sent": False,
            "dry_run": dry_run,
        }

        if not digest.items:
            logger.info("No items in digest, skipping send")
            return stats

        if dry_run:
            logger.info(f"Dry run: would send {len(digest.items)} items")
            return stats

        # Send email
        sender = EmailSender()
        if sender.is_configured():
            result = sender.send_digest(digest)
            if result.success:
                # Mark items as delivered
                generator.mark_digest_delivered(digest)
                stats["sent"] = True
                logger.info("Digest sent successfully")
            else:
                logger.error(f"Failed to send digest: {result.message}")
        else:
            logger.warning("Email not configured, skipping send")

        return stats


class BabelByteScheduler:
    """Scheduler for BabelByte jobs."""

    def __init__(self, use_mock: bool = False):
        self.scheduler = BackgroundScheduler()
        self.runner = JobRunner(use_mock=use_mock)
        self._running = False

    def setup_jobs(self) -> None:
        """Setup scheduled jobs based on configuration."""
        settings = get_settings().scheduler

        # Content fetch job - runs every N hours
        self.scheduler.add_job(
            self.runner.fetch_all_content,
            trigger=IntervalTrigger(hours=settings.fetch_interval_hours),
            id="fetch_content",
            name="Fetch content from subscriptions",
            replace_existing=True,
        )

        # Digest job - runs daily at configured time
        self.scheduler.add_job(
            self.runner.send_digest,
            trigger=CronTrigger(
                hour=settings.digest_hour,
                minute=settings.digest_minute,
            ),
            id="send_digest",
            name="Generate and send daily digest",
            replace_existing=True,
        )

        logger.info(
            f"Jobs scheduled: fetch every {settings.fetch_interval_hours}h, "
            f"digest at {settings.digest_send_time}"
        )

    def start(self) -> None:
        """Start the scheduler."""
        if not self._running:
            self.setup_jobs()
            self.scheduler.start()
            self._running = True
            logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._running:
            self.scheduler.shutdown()
            self.runner.close_db()
            self._running = False
            logger.info("Scheduler stopped")

    def run_fetch_now(self) -> dict:
        """Run fetch job immediately."""
        return self.runner.fetch_all_content()

    def run_digest_now(self, dry_run: bool = False) -> dict:
        """Run digest job immediately."""
        return self.runner.send_digest(dry_run=dry_run)

    def get_jobs(self) -> list:
        """Get list of scheduled jobs."""
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time,
            }
            for job in self.scheduler.get_jobs()
        ]
