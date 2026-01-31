"""Diagnostic SQL queries for data validation."""


class DiagnosticQueries:
    """SQL queries for diagnosing data integrity issues."""

    # Orphan content items: subscription_id references non-existent subscription
    ORPHAN_CONTENT_ITEMS = """
        SELECT c.id, c.title, c.subscription_id
        FROM content_items c
        LEFT JOIN subscriptions s ON c.subscription_id = s.id
        WHERE s.id IS NULL
    """

    # Duplicate external_id per source_type (should be unique)
    DUPLICATE_EXTERNAL_IDS = """
        SELECT source_type, external_id, COUNT(*) as count
        FROM content_items
        GROUP BY source_type, external_id
        HAVING COUNT(*) > 1
    """

    # Processed items without summary (anomaly)
    PROCESSED_NO_SUMMARY = """
        SELECT id, title, processed_at
        FROM content_items
        WHERE processed_at IS NOT NULL AND (summary IS NULL OR summary = '')
    """

    # Importance score out of range (should be 1-10)
    IMPORTANCE_OUT_OF_RANGE = """
        SELECT id, title, importance_score
        FROM content_items
        WHERE importance_score IS NOT NULL AND (importance_score < 1 OR importance_score > 10)
    """

    # Empty event clusters (no members)
    EMPTY_CLUSTERS = """
        SELECT ec.id, ec.event_title, ec.article_count
        FROM event_clusters ec
        LEFT JOIN event_members em ON ec.id = em.event_cluster_id
        WHERE em.content_item_id IS NULL
    """

    # Cluster article_count mismatch with actual member count
    CLUSTER_COUNT_MISMATCH = """
        SELECT ec.id, ec.event_title, ec.article_count as stored_count,
               COUNT(em.content_item_id) as actual_count
        FROM event_clusters ec
        LEFT JOIN event_members em ON ec.id = em.event_cluster_id
        GROUP BY ec.id
        HAVING ec.article_count != COUNT(em.content_item_id)
    """

    # Orphan event members: cluster_id or content_id references non-existent row
    ORPHAN_EVENT_MEMBERS_CLUSTER = """
        SELECT em.content_item_id, em.event_cluster_id
        FROM event_members em
        LEFT JOIN event_clusters ec ON em.event_cluster_id = ec.id
        WHERE ec.id IS NULL
    """

    ORPHAN_EVENT_MEMBERS_CONTENT = """
        SELECT em.content_item_id, em.event_cluster_id
        FROM event_members em
        LEFT JOIN content_items c ON em.content_item_id = c.id
        WHERE c.id IS NULL
    """

    # FTS index missing entries for processed items
    FTS_MISSING_ENTRIES = """
        SELECT c.id, c.title
        FROM content_items c
        LEFT JOIN content_fts f ON c.id = f.content_id
        WHERE c.processed_at IS NOT NULL AND f.content_id IS NULL
    """

    # Duplicate cluster memberships (should not happen with new schema)
    DUPLICATE_CLUSTER_MEMBERSHIPS = """
        SELECT content_item_id, COUNT(*) as cluster_count
        FROM event_members
        GROUP BY content_item_id
        HAVING COUNT(*) > 1
    """

    # Orphan action items: content_item_id references non-existent content
    ORPHAN_ACTION_ITEMS = """
        SELECT a.id, a.description, a.content_item_id
        FROM action_items a
        LEFT JOIN content_items c ON a.content_item_id = c.id
        WHERE a.content_item_id IS NOT NULL AND c.id IS NULL
    """

    # Orphan topic associations
    ORPHAN_CONTENT_TOPICS = """
        SELECT ct.content_id, ct.topic_id
        FROM content_topics ct
        LEFT JOIN content_items c ON ct.content_id = c.id
        LEFT JOIN topics t ON ct.topic_id = t.id
        WHERE c.id IS NULL OR t.id IS NULL
    """

    # Orphan topic snapshots
    ORPHAN_TOPIC_SNAPSHOTS = """
        SELECT ts.id, ts.topic_id, ts.snapshot_date
        FROM topic_snapshots ts
        LEFT JOIN topics t ON ts.topic_id = t.id
        WHERE t.id IS NULL
    """

    # Expired AI cache entries
    EXPIRED_AI_CACHE = """
        SELECT COUNT(*) as count
        FROM ai_cache
        WHERE expires_at <= datetime('now')
    """

    # AI cache statistics
    AI_CACHE_STATS = """
        SELECT
            COUNT(*) as total_entries,
            SUM(CASE WHEN expires_at > datetime('now') THEN 1 ELSE 0 END) as valid_entries,
            SUM(CASE WHEN expires_at <= datetime('now') THEN 1 ELSE 0 END) as expired_entries,
            MIN(created_at) as oldest_entry,
            MAX(created_at) as newest_entry
        FROM ai_cache
    """

    # Content items statistics
    CONTENT_STATS = """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN processed_at IS NOT NULL THEN 1 ELSE 0 END) as processed,
            SUM(CASE WHEN delivered = 1 THEN 1 ELSE 0 END) as delivered,
            AVG(importance_score) as avg_importance
        FROM content_items
    """

    # Event cluster statistics
    CLUSTER_STATS = """
        SELECT
            COUNT(*) as total_clusters,
            SUM(article_count) as total_members,
            AVG(article_count) as avg_members_per_cluster
        FROM event_clusters
    """

    # Delivered items in last N days
    DELIVERED_ITEMS_RECENT = """
        SELECT date(delivered_at) as date, COUNT(*) as count
        FROM content_items
        WHERE delivered = 1 AND delivered_at IS NOT NULL
          AND delivered_at >= date('now', '-7 days')
        GROUP BY date(delivered_at)
        ORDER BY date DESC
    """

    # Items per category
    ITEMS_BY_CATEGORY = """
        SELECT category, COUNT(*) as count
        FROM content_items
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
    """

    # Recently clustered items (for debugging idempotency)
    RECENTLY_CLUSTERED = """
        SELECT c.id, c.title, ec.event_title, em.similarity_score, em.detection_method
        FROM content_items c
        INNER JOIN event_members em ON c.id = em.content_item_id
        INNER JOIN event_clusters ec ON em.event_cluster_id = ec.id
        WHERE c.fetched_at >= date('now', '-1 day')
        ORDER BY c.fetched_at DESC
        LIMIT 50
    """

    # Cluster attempts tracking
    CLUSTER_ATTEMPTS = """
        SELECT
            SUM(CASE WHEN cluster_attempted_at IS NOT NULL THEN 1 ELSE 0 END) as attempted,
            SUM(CASE WHEN cluster_attempted_at IS NULL THEN 1 ELSE 0 END) as not_attempted
        FROM content_items
        WHERE processed_at IS NOT NULL AND delivered = 0
    """
