"""Scheduler jobs for BabelByte."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, List, Tuple

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import get_settings
from src.delivery.email_sender import EmailSender
from src.fetchers.base import BaseFetcher, FetchResult
from src.fetchers.reddit import RedditFetcher
from src.fetchers.twitter import TwitterFetcher, TwitterAPIioFetcher, MockTwitterFetcher
from src.processors.digest_processor import DigestGenerator
from src.storage.database import Database, SyncDatabase
from src.storage.models import SourceType, Subscription

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

    def fetch_all_content(self, progress_callback=None) -> dict:
        """
        Fetch content from all enabled subscriptions.

        Optimizations:
        - Reddit subscriptions are fetched fully in parallel
        - Twitter subscriptions use controlled concurrency with rate limiting
        - Shared HTTP client for TwitterAPI.io (connection pooling)

        Args:
            progress_callback: Optional callback(subscription_name, index, total, new_items)
                              called after each subscription is fetched.

        Returns:
            Dict with fetch statistics.
        """
        logger.info("Starting content fetch job...")
        db = self.get_db()

        subscriptions = db.list_subscriptions(enabled_only=True)
        if not subscriptions:
            logger.info("No enabled subscriptions found")
            return {"total": 0, "fetched": 0, "errors": 0, "new_items": 0}

        # Separate subscriptions by type for optimized parallel fetching
        reddit_subs = [s for s in subscriptions if s.source_type == SourceType.REDDIT]
        twitter_subs = [s for s in subscriptions if s.source_type == SourceType.TWITTER]

        stats = {"total": len(subscriptions), "fetched": 0, "errors": 0, "new_items": 0}

        # Run async fetching
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self._fetch_all_async(reddit_subs, twitter_subs, progress_callback)
            )
            loop.close()
        except Exception as e:
            logger.error(f"Error in async fetch: {e}")
            results = []

        # Process results and store items
        for result, sub in results:
            try:
                if result.success:
                    new_count = 0
                    skipped_url = 0
                    for item in result.items:
                        if db.content_exists(item.source_type, item.external_id):
                            continue
                        if db.url_exists(item.url):
                            skipped_url += 1
                            logger.debug(f"Skipping duplicate URL: {item.url[:60]}...")
                            continue
                        db.add_content_item(item)
                        new_count += 1
                    if skipped_url > 0:
                        logger.info(f"Skipped {skipped_url} items with duplicate URLs")

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
                logger.error(f"Error processing {sub.display_name}: {e}")

        logger.info(
            f"Fetch job completed: {stats['fetched']}/{stats['total']} successful, "
            f"{stats['new_items']} new items"
        )
        return stats

    async def _fetch_all_async(
        self,
        reddit_subs: List[Subscription],
        twitter_subs: List[Subscription],
        progress_callback=None,
    ) -> List[Tuple[FetchResult, Subscription]]:
        """
        Fetch all subscriptions asynchronously with optimized parallelism.

        - Reddit: Fully parallel (no rate limits)
        - Twitter: Controlled concurrency with semaphore for rate limiting
        """
        settings = get_settings()
        results = []

        # Initialize fetchers
        reddit_fetcher = RedditFetcher()

        use_twitterapi_io = False
        if self.use_mock:
            twitter_fetcher = MockTwitterFetcher()
        elif settings.twitter.use_twitterapi_io:
            twitter_fetcher = TwitterAPIioFetcher()
            use_twitterapi_io = True
            logger.info("Using TwitterAPI.io for Twitter data (parallel mode)")
        elif settings.twitter.bearer_token:
            twitter_fetcher = TwitterFetcher()
        else:
            twitter_fetcher = MockTwitterFetcher()

        # Fetch Reddit subscriptions fully in parallel
        if reddit_subs:
            logger.info(f"Fetching {len(reddit_subs)} Reddit subscriptions in parallel...")
            reddit_tasks = [reddit_fetcher.fetch(sub) for sub in reddit_subs]
            reddit_results = await asyncio.gather(*reddit_tasks, return_exceptions=True)

            for sub, result in zip(reddit_subs, reddit_results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching {sub.display_name}: {result}")
                    results.append((FetchResult(subscription=sub, success=False, error_message=str(result)), sub))
                else:
                    results.append((result, sub))

        # Fetch Twitter subscriptions with controlled concurrency
        if twitter_subs:
            if use_twitterapi_io:
                # Use shared HTTP client and semaphore for rate limiting
                # TwitterAPI.io: ~1 request per 5 seconds, but we can do 2-3 concurrent
                # with slight overlap since network latency varies
                twitter_results = await self._fetch_twitter_parallel(
                    twitter_fetcher, twitter_subs, max_concurrent=2, delay_between=3.0
                )
            else:
                # Official Twitter API or mock - can run more parallel
                twitter_results = await self._fetch_twitter_parallel(
                    twitter_fetcher, twitter_subs, max_concurrent=5, delay_between=0.5
                )

            results.extend(twitter_results)

        return results

    async def _fetch_twitter_parallel(
        self,
        fetcher: BaseFetcher,
        subscriptions: List[Subscription],
        max_concurrent: int = 2,
        delay_between: float = 3.0,
    ) -> List[Tuple[FetchResult, Subscription]]:
        """
        Fetch Twitter subscriptions with controlled parallelism.

        Uses a semaphore to limit concurrent requests and adds delay between
        request starts to respect rate limits while still being faster than sequential.
        """
        if not subscriptions:
            return []

        logger.info(
            f"Fetching {len(subscriptions)} Twitter subscriptions "
            f"(max_concurrent={max_concurrent}, delay={delay_between}s)"
        )

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        # Use shared HTTP client for connection pooling (TwitterAPIio only)
        shared_client = None
        if isinstance(fetcher, TwitterAPIioFetcher):
            shared_client = httpx.AsyncClient(timeout=fetcher.timeout)
            fetcher.set_shared_client(shared_client)

        async def fetch_with_semaphore(sub: Subscription, index: int) -> Tuple[FetchResult, Subscription]:
            async with semaphore:
                # Add staggered delay to spread out requests
                if index > 0:
                    await asyncio.sleep(delay_between * (index % max_concurrent))

                try:
                    if shared_client and isinstance(fetcher, TwitterAPIioFetcher):
                        result = await fetcher.fetch(sub, client=shared_client)
                    else:
                        result = await fetcher.fetch(sub)
                    return (result, sub)
                except Exception as e:
                    logger.error(f"Error fetching {sub.display_name}: {e}")
                    return (FetchResult(subscription=sub, success=False, error_message=str(e)), sub)

        try:
            # Create tasks with index for staggered delays
            tasks = [fetch_with_semaphore(sub, i) for i, sub in enumerate(subscriptions)]
            results = await asyncio.gather(*tasks)
        finally:
            if shared_client:
                await shared_client.aclose()

        return list(results)

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
