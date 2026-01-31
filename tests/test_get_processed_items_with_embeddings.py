"""Test for get_processed_items_with_embeddings bug fix.

This test verifies that the method doesn't raise IndexError when accessing
row["created_at"] (which was incorrectly used instead of fetched_at).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from datetime import datetime, timedelta

from src.storage.database import SyncDatabase
from src.storage.models import Subscription, SourceType, SubscriptionType, ContentItem


@pytest.fixture
def db():
    """Create a test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = SyncDatabase(db_path)
        db.connect()
        yield db
        db.close()


def test_get_processed_items_with_embeddings_no_created_at_error(db):
    """Test that get_processed_items_with_embeddings doesn't raise IndexError for created_at."""
    # Add a subscription
    sub = Subscription(
        source_type=SourceType.REDDIT,
        subscription_type=SubscriptionType.SUBREDDIT,
        name="test",
        enabled=True,
        created_at=datetime.now(),
    )
    sub = db.add_subscription(sub)

    # Add a content item with processing results
    item = ContentItem(
        subscription_id=sub.id,
        source_type=SourceType.REDDIT,
        external_id="test123",
        title="Test Title",
        content="Test content",
        url="https://example.com",
        author="testuser",
        published_at=datetime.now(),
        fetched_at=datetime.now(),
        summary="Test summary",
        category="Tech",
        importance_score=5,
        processed_at=datetime.now(),
    )
    item = db.add_content_item(item)

    # Add embedding for the item
    embedding = np.random.rand(384).astype(np.float32).tobytes()
    db.save_content_embedding(item.id, embedding, "test-model", 384)

    # This should NOT raise IndexError
    since = datetime.now() - timedelta(days=1)
    results = db.get_processed_items_with_embeddings(since=since, limit=10)

    assert len(results) == 1
    result_item, result_embedding, result_dim = results[0]
    assert result_item.title == "Test Title"
    assert result_item.fetched_at is not None
    assert result_item.source_type == SourceType.REDDIT
    assert result_dim == 384


def test_get_processed_items_with_embeddings_returns_correct_fields(db):
    """Test that all ContentItem fields are correctly populated."""
    # Add a subscription
    sub = Subscription(
        source_type=SourceType.HACKERNEWS,
        subscription_type=SubscriptionType.HN_FRONT,
        name="hn-front",
        enabled=True,
        created_at=datetime.now(),
    )
    sub = db.add_subscription(sub)

    # Add a content item with all fields populated
    now = datetime.now()
    item = ContentItem(
        subscription_id=sub.id,
        source_type=SourceType.HACKERNEWS,
        external_id="hn_12345",
        title="HN Test Article",
        content="Full article content here",
        url="https://news.ycombinator.com/item?id=12345",
        author="hn_user",
        published_at=now - timedelta(hours=2),
        fetched_at=now - timedelta(hours=1),
        summary="A test summary for HN article",
        category="Technology",
        importance_score=8,
        processed_at=now,
        one_liner="Key takeaway in one line",
        key_points='[{"type": "insight", "value": "test point"}]',
        impact_assessment='{"short_term": "high", "long_term": "medium"}',
        actionable_items='[{"type": "research", "description": "look into this"}]',
    )
    item = db.add_content_item(item)

    # Add embedding
    embedding = np.random.rand(768).astype(np.float32).tobytes()
    db.save_content_embedding(item.id, embedding, "openai-ada", 768)

    # Fetch and verify
    since = datetime.now() - timedelta(days=1)
    results = db.get_processed_items_with_embeddings(since=since, limit=10)

    assert len(results) == 1
    result_item, result_embedding, result_dim = results[0]

    # Verify core fields
    assert result_item.id == item.id
    assert result_item.subscription_id == sub.id
    assert result_item.source_type == SourceType.HACKERNEWS
    assert result_item.external_id == "hn_12345"
    assert result_item.title == "HN Test Article"
    assert result_item.content == "Full article content here"
    assert result_item.url == "https://news.ycombinator.com/item?id=12345"
    assert result_item.author == "hn_user"

    # Verify datetime fields (the fix)
    assert result_item.published_at is not None
    assert result_item.fetched_at is not None
    assert result_item.processed_at is not None

    # Verify AI-processed fields (basic fields stored by add_content_item)
    assert result_item.summary == "A test summary for HN article"
    assert result_item.category == "Technology"
    assert result_item.importance_score == 8
    # Note: one_liner, key_points, etc. are set via update_content_item, not add_content_item

    # Verify embedding
    assert result_dim == 768
    assert result_embedding == embedding
