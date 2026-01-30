"""Twitter content fetcher using Twitter API v2."""

from datetime import datetime
from typing import Optional

import tweepy

from config.settings import get_settings
from src.fetchers.base import BaseFetcher, FetchResult
from src.storage.models import ContentItem, SourceType, Subscription, SubscriptionType


class TwitterFetcher(BaseFetcher):
    """Fetcher for Twitter content using Twitter API v2."""

    source_type = SourceType.TWITTER

    # Free tier limit: 1,500 tweets/month
    DEFAULT_MAX_RESULTS = 10  # Conservative default

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

    async def fetch(self, subscription: Subscription) -> FetchResult:
        """Fetch tweets from a Twitter user."""
        if subscription.subscription_type != SubscriptionType.TWITTER_USER:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message=f"Invalid subscription type: {subscription.subscription_type}",
            )

        try:
            # First, get the user ID from username
            user = self.client.get_user(username=subscription.name)
            if not user.data:
                return FetchResult(
                    subscription=subscription,
                    success=False,
                    error_message=f"Twitter user @{subscription.name} not found",
                )

            user_id = user.data.id
            username = user.data.username

            # Fetch recent tweets
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=min(self.max_results, 100),  # API max is 100
                tweet_fields=["created_at", "text", "author_id", "public_metrics"],
                exclude=["retweets", "replies"],  # Only original tweets
            )

            items = []
            if tweets.data:
                for tweet in tweets.data:
                    item = self._parse_tweet(subscription, tweet, username)
                    if item:
                        items.append(item)

            return FetchResult(
                subscription=subscription,
                items=items,
                success=True,
            )

        except tweepy.TooManyRequests as e:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message="Twitter API rate limit exceeded. Please wait before trying again.",
            )
        except tweepy.Unauthorized as e:
            return FetchResult(
                subscription=subscription,
                success=False,
                error_message="Twitter API authentication failed. Please check your Bearer Token.",
            )
        except tweepy.NotFound as e:
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
