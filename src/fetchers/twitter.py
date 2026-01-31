"""Twitter content fetcher using Twitter API v2 or TwitterAPI.io."""

import logging
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import httpx
import tweepy

from config.settings import get_settings
from src.fetchers.base import BaseFetcher, FetchResult
from src.storage.models import ContentItem, SourceType, Subscription, SubscriptionType

logger = logging.getLogger(__name__)


class TwitterFetcher(BaseFetcher):
    """Fetcher for Twitter content using Twitter API v2.

    Optimized for Twitter's free tier limit of 1,500 tweets/month:
    - Caches user_id to avoid repeated get_user() calls
    - Uses since_id to only fetch new tweets
    - Returns updated subscription with cached values
    """

    source_type = SourceType.TWITTER

    # Free tier limit: 1,500 tweets/month
    # With since_id, we typically get fewer results, so we can afford a higher max
    DEFAULT_MAX_RESULTS = 20

    def __init__(self, bearer_token: Optional[str] = None, max_results: int = DEFAULT_MAX_RESULTS):
        self.bearer_token = bearer_token or get_settings().twitter.bearer_token
        self.max_results = max_results
        self._client: Optional[tweepy.Client] = None

    @property
    def client(self) -> tweepy.Client:
        """Get or create the Tweepy client."""
        if self._client is None:
            if not self.bearer_token:
                raise ValueError(
                    "Twitter Bearer Token not configured. "
                    "Please set TWITTER_BEARER_TOKEN in your .env file."
                )
            self._client = tweepy.Client(bearer_token=self.bearer_token)
        return self._client

    async def _get_user_id(
        self, subscription: Subscription
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Get Twitter user_id, using cache if available.

        Returns:
            Tuple of (user_id, username, error_message)
        """
        # Use cached user_id if available (saves 1 API call per fetch)
        if subscription.twitter_user_id:
            logger.debug(f"Using cached user_id for @{subscription.name}")
            return subscription.twitter_user_id, subscription.name, None

        # Need to fetch user_id (costs 1 API call)
        logger.info(f"Fetching user_id for @{subscription.name} (will be cached)")
        user = self.client.get_user(username=subscription.name)
        if not user.data:
            return None, None, f"Twitter user @{subscription.name} not found"

        return str(user.data.id), user.data.username, None

    async def fetch(self, subscription: Subscription) -> FetchResult:
        """Fetch tweets from a Twitter user.

        Optimizations:
        - Uses cached twitter_user_id to skip get_user() API call
        - Uses since_id to only fetch tweets newer than last fetch
        - Updates subscription with new cache values for caller to persist
        """
        if subscription.subscription_type != SubscriptionType.TWITTER_USER:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"Invalid subscription type: {subscription.subscription_type}",
            )

        try:
            # Get user_id (from cache or API)
            user_id, username, error = await self._get_user_id(subscription)
            if error:
                return FetchResult(
                    subscription=subscription,
                    success=False,
                    error_message=error,
                )

            # Update subscription cache if we fetched a new user_id
            if not subscription.twitter_user_id:
                subscription.twitter_user_id = user_id

            # Build API parameters
            api_params = {
                "id": user_id,
                "max_results": min(self.max_results, 100),
                "tweet_fields": ["created_at", "text", "author_id", "public_metrics"],
                "exclude": ["retweets", "replies"],
            }

            # Use since_id to only fetch new tweets (huge API savings)
            if subscription.last_tweet_id:
                api_params["since_id"] = subscription.last_tweet_id
                logger.debug(f"Fetching tweets since_id={subscription.last_tweet_id}")

            # Fetch tweets
            tweets = self.client.get_users_tweets(**api_params)

            items = []
            newest_tweet_id = subscription.last_tweet_id

            if tweets.data:
                for tweet in tweets.data:
                    item = self._parse_tweet(subscription, tweet, username)
                    if item:
                        items.append(item)
                    # Track the newest tweet_id for next fetch
                    if newest_tweet_id is None or int(tweet.id) > int(newest_tweet_id):
                        newest_tweet_id = str(tweet.id)

                logger.info(f"Fetched {len(items)} new tweets from @{username}")
            else:
                logger.debug(f"No new tweets from @{username}")

            # Update subscription with newest tweet_id for next fetch
            subscription.last_tweet_id = newest_tweet_id

            return FetchResult(
                subscription=subscription,
                items=items,
                success=True,
            )

        except tweepy.TooManyRequests:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message="Twitter API rate limit exceeded. Please wait before trying again.",
            )
        except tweepy.Unauthorized:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message="Twitter API authentication failed. Please check your Bearer Token.",
            )
        except tweepy.NotFound:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"Twitter user @{subscription.name} not found.",
            )
        except tweepy.TweepyException as e:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"Twitter API error: {str(e)}",
            )
        except Exception as e:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    async def validate_subscription(self, subscription: Subscription) -> bool:
        """Validate that a Twitter user exists."""
        if not self.bearer_token:
            return False

        try:
            user = self.client.get_user(username=subscription.name)
            return user.data is not None
        except Exception:
            return False

    def _parse_tweet(
        self, subscription: Subscription, tweet, username: str
    ) -> Optional[ContentItem]:
        """Parse a tweet into a ContentItem."""
        try:
            external_id = str(tweet.id)
            text = tweet.text or ""

            # Use first line or truncated text as title
            title = text.split("\n")[0][:100]
            if len(text) > 100:
                title = title[:97] + "..."

            # Build tweet URL
            url = f"https://twitter.com/{username}/status/{tweet.id}"

            # Parse created_at
            published_at = tweet.created_at if tweet.created_at else datetime.now()
            if isinstance(published_at, str):
                published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))

            return self.create_content_item(
                subscription=subscription,
                external_id=external_id,
                title=title,
                content=text,
                url=url,
                author=f"@{username}",
                published_at=published_at,
            )

        except Exception:
            return None


class TwitterAPIioFetcher(BaseFetcher):
    """Fetcher for Twitter content using TwitterAPI.io (third-party, cheaper).

    Optimizations:
    - Reuses HTTP client via connection pooling for faster requests
    - Supports shared client for parallel fetching across subscriptions
    - TTL memory cache to avoid redundant API calls within short intervals
    """

    source_type = SourceType.TWITTER

    BASE_URL = "https://api.twitterapi.io/twitter/user/last_tweets"
    DEFAULT_MAX_RESULTS = 20
    DEFAULT_CACHE_TTL = 300.0  # 5 minutes

    # Class-level cache shared across instances: {username: (timestamp, FetchResult)}
    _cache: dict[str, tuple[float, "FetchResult"]] = {}
    _cache_ttl: float = float(os.environ.get("TWITTERAPI_CACHE_TTL", DEFAULT_CACHE_TTL))

    def __init__(self, api_key: Optional[str] = None, max_results: int = DEFAULT_MAX_RESULTS):
        self.api_key = api_key or get_settings().twitter.twitterapi_io_key
        self.max_results = max_results
        self.timeout = 15.0  # Reduced from 30s for faster failure detection
        self._shared_client: Optional[httpx.AsyncClient] = None

    def set_shared_client(self, client: httpx.AsyncClient) -> None:
        """Set a shared HTTP client for connection pooling across multiple fetches."""
        self._shared_client = client

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the response cache (useful for testing)."""
        cls._cache.clear()

    @classmethod
    def _get_cached_result(cls, username: str) -> Optional["FetchResult"]:
        """Get cached result if still valid within TTL."""
        cache_key = username.lower()
        if cache_key in cls._cache:
            cached_time, cached_result = cls._cache[cache_key]
            if time.time() - cached_time < cls._cache_ttl:
                return cached_result
            # Expired, remove from cache
            del cls._cache[cache_key]
        return None

    @classmethod
    def _set_cached_result(cls, username: str, result: "FetchResult") -> None:
        """Store result in cache with current timestamp."""
        cache_key = username.lower()
        cls._cache[cache_key] = (time.time(), result)

    async def fetch(
        self, subscription: Subscription, client: Optional[httpx.AsyncClient] = None
    ) -> FetchResult:
        """Fetch tweets from a Twitter user using TwitterAPI.io.

        Args:
            subscription: The subscription to fetch tweets for.
            client: Optional shared HTTP client for connection pooling.

        Uses last_tweet_id for incremental fetching to avoid re-fetching old tweets.
        Results are cached for TTL seconds to avoid redundant API calls.
        """
        if subscription.subscription_type != SubscriptionType.TWITTER_USER:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"Invalid subscription type: {subscription.subscription_type}",
            )

        if not self.api_key:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message="TwitterAPI.io API key not configured. Set TWITTERAPI_IO_KEY in .env",
            )

        try:
            username = subscription.name.lstrip("@")

            # Check cache first to avoid redundant API calls
            cached_result = self._get_cached_result(username)
            if cached_result is not None:
                logger.info(f"Using cached results for @{username} (TTL: {self._cache_ttl}s)")
                return cached_result

            # Use provided client, shared client, or create new one
            http_client = client or self._shared_client
            should_close = http_client is None

            if http_client is None:
                http_client = httpx.AsyncClient(timeout=self.timeout)

            try:
                response = await http_client.get(
                    self.BASE_URL,
                    params={
                        "userName": username,
                        "includeReplies": "false",
                    },
                    headers={"X-API-Key": self.api_key},
                )

                if response.status_code == 401:
                    return FetchResult(
                        subscription=subscription,
                        success=False,
                        error_message="TwitterAPI.io authentication failed. Check your API key.",
                    )
                elif response.status_code == 402:
                    return FetchResult(
                        subscription=subscription,
                        success=False,
                        error_message="TwitterAPI.io credits exhausted. Please add credits.",
                    )
                elif response.status_code == 404:
                    return FetchResult(
                        subscription=subscription,
                        success=False,
                        error_message=f"Twitter user @{username} not found.",
                    )

                response.raise_for_status()
                data = response.json()
            finally:
                if should_close:
                    await http_client.aclose()

            if data.get("status") != "success":
                err_msg = data.get("msg") or data.get("message") or "Unknown TwitterAPI.io error"
                return FetchResult(
                    subscription=subscription,
                    success=False,
                    error_message=err_msg,
                )

            items = []
            # Tweets are nested under data.data.tweets
            tweets = data.get("data", {}).get("tweets", [])
            newest_tweet_id = subscription.last_tweet_id

            for tweet in tweets[: self.max_results]:
                tweet_id = tweet.get("id", "")
                if not tweet_id:
                    continue

                # Skip tweets we've already seen (incremental fetch)
                if subscription.last_tweet_id:
                    try:
                        if int(tweet_id) <= int(subscription.last_tweet_id):
                            continue
                    except (ValueError, TypeError):
                        pass

                item = self._parse_tweet(subscription, tweet)
                if item:
                    items.append(item)

                # Track newest tweet_id for next fetch
                if newest_tweet_id is None:
                    newest_tweet_id = tweet_id
                else:
                    try:
                        if int(tweet_id) > int(newest_tweet_id):
                            newest_tweet_id = tweet_id
                    except (ValueError, TypeError):
                        pass

            # Update subscription with newest tweet_id
            subscription.last_tweet_id = newest_tweet_id

            if items:
                logger.info(f"Fetched {len(items)} new tweets from @{username} via TwitterAPI.io")
            else:
                logger.debug(f"No new tweets from @{username}")

            result = FetchResult(
                subscription=subscription,
                items=items,
                success=True,
            )

            # Cache successful result
            self._set_cached_result(username, result)
            return result

        except httpx.HTTPStatusError as e:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"HTTP error {e.response.status_code}: {e.response.text[:200]}",
            )
        except httpx.RequestError as e:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"Request error: {str(e)}",
            )
        except Exception as e:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    async def validate_subscription(self, subscription: Subscription) -> bool:
        """Validate that a Twitter user exists."""
        if not self.api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self.BASE_URL,
                    params={"userName": subscription.name.lstrip("@")},
                    headers={"X-API-Key": self.api_key},
                )
                data = response.json()
                return data.get("status") == "success"
        except Exception:
            return False

    def _parse_tweet(self, subscription: Subscription, tweet: dict) -> Optional[ContentItem]:
        """Parse a tweet from TwitterAPI.io response into a ContentItem."""
        try:
            external_id = tweet.get("id", "")
            if not external_id:
                return None

            text = tweet.get("text", "")
            author_info = tweet.get("author", {})
            username = author_info.get("userName", subscription.name)

            # Use first line or truncated text as title
            title = text.split("\n")[0][:100]
            if len(text) > 100:
                title = title[:97] + "..."

            # Build tweet URL
            url = tweet.get("url", f"https://twitter.com/{username}/status/{external_id}")

            # Parse created_at (format: "Fri Jan 24 18:30:00 +0000 2025")
            created_at_str = tweet.get("createdAt", "")
            published_at = datetime.now()
            if created_at_str:
                try:
                    published_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
                    published_at = published_at.replace(tzinfo=None)
                except (ValueError, TypeError):
                    pass

            return self.create_content_item(
                subscription=subscription,
                external_id=external_id,
                title=title,
                content=text,
                url=url,
                author=f"@{username}",
                published_at=published_at,
            )

        except Exception:
            return None


class MockTwitterFetcher(BaseFetcher):
    """Mock Twitter fetcher for testing without API access."""

    source_type = SourceType.TWITTER

    async def fetch(self, subscription: Subscription) -> FetchResult:
        """Return mock data for testing."""
        mock_items = [
            self.create_content_item(
                subscription=subscription,
                external_id=f"mock_{subscription.name}_{i}",
                title=f"Mock tweet {i} from @{subscription.name}",
                content=f"This is a mock tweet #{i} for testing purposes. "
                f"Real tweets will appear when Twitter API is configured.",
                url=f"https://twitter.com/{subscription.name}/status/mock{i}",
                author=f"@{subscription.name}",
                published_at=datetime.now(),
            )
            for i in range(3)
        ]

        return FetchResult(
            subscription=subscription,
            items=mock_items,
            success=True,
        )

    async def validate_subscription(self, subscription: Subscription) -> bool:
        """Always return True for mock fetcher."""
        return True
