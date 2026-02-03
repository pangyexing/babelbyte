# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BabelByte is an AI-powered **personal intelligence product** that fetches content from Reddit, Twitter, Hacker News, and RSS feeds, processes it with AI (Claude CLI, Codex CLI, or Ollama local models), and delivers structured intelligence briefings via email. It supports event tracking with semantic clustering, automatic topic discovery, knowledge base search, and periodic reports.

## Commands

**Installation:**
```bash
pip install -e .              # Basic install
pip install -e ".[dev]"       # With dev dependencies (pytest, black, ruff)
```

**CLI entry point:** `bb` or `python main.py`

**Core commands:**
```bash
bb subscribe reddit <subreddit>     # Subscribe to subreddit
bb subscribe twitter <username>     # Subscribe to Twitter user
bb subscribe hackernews <name> --type front  # Subscribe to HN (front/new/best/ask/show)
bb subscribe rss <name> --url <feed-url>     # Subscribe to RSS/Atom feed
bb list                             # Show subscriptions
bb daily                            # One-shot: fetch → process → digest (recommended)
bb daily --dry-run                  # Preview without sending email
bb daily --skip-fetch               # Skip fetch, process existing content
bb run                              # Start scheduler daemon (for 24/7 servers)
bb --mock <command>                 # Test mode with mock data
```

**Knowledge Base (Phase 4):**
```bash
bb search "query"                   # Full-text search
bb search "AI" --category AI --from 2024-01-01
bb browse --date yesterday          # Browse by date
bb item 123 --open                  # View item details
bb mark 123 saved                   # Mark item state (unread/read/saved/archived/flagged)
bb stats                            # Category and state statistics
bb rebuild-index                    # Rebuild FTS index
```

**Event Stream (Phase 2):**
```bash
bb events --days 7                  # List recent event clusters
bb event 123                        # View event details and members
bb cluster --limit 100              # Run event clustering
```

**Topic Radar (Phase 3):**
```bash
bb topics                           # List all topics
bb topic add "AI" --keywords "GPT,ChatGPT,AI"
bb topic show "AI"                  # View topic details
bb topic delete "AI" --yes
bb topic discover --days 14         # Auto-discover topics from content
bb topic suggestions                # View pending topic suggestions
bb topic review                     # Interactive review of suggestions
```

**Action List (Phase 5):**
```bash
bb actions --status pending         # List action items
bb action 123 done                  # Mark action as done/dismissed
```

**Reports (Phase 6):**
```bash
bb report week                      # Generate weekly report
bb report month --months-ago 1      # Generate monthly report
```

**Embeddings (Semantic Clustering):**
```bash
bb embeddings compute --limit 1000  # Compute content embeddings
bb embeddings stats                 # View embedding statistics
bb embeddings rebuild-centroids     # Rebuild cluster centroid vectors
```

**Diagnostics & Optimization:**
```bash
bb validate [--fix] [--verbose]     # Run 12 data integrity checks
bb token-stats [--reset]            # View token usage and costs
bb cache-stats [--cleanup]          # Cache metrics and cleanup
```

**Parallel Processing:**
```bash
bb cluster --parallel --workers 4   # Parallel event clustering
bb digest --parallel                # Parallel digest generation
```

**Code quality:**
```bash
black .                                    # Format (line length: 100)
ruff check . --select E,F,W,I              # Lint
```

## Architecture

```
                        CLI (Click)
                            ↓
                   Scheduler (APScheduler)
                            ↓
┌───────────────────────────┴───────────────────────────┐
↓                                                       ↓
Fetchers                                         Rule Classifier
(Reddit/Twitter/HN/RSS)                          (40+ 域名预分类)
↓                                                       ↓
└───────────────→ SQLite + FTS5 ←───────────────────────┘
                            ↓
                      Embeddings ←─── (本地免费，启用语义去重)
                 (SentenceTransformers)
                            ↓
                    AI Processors ←─── Token Tracker
              (Claude/Codex/Ollama)     (8类调用追踪)
                 (跳过相似内容:85%阈值)
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
  Event Stream         Topic Radar        Topic Discovery
  (混合聚类:            (关键词匹配         (实体频率/
   40%规则+60%语义       趋势检测)           趋势突增)
   +AI确认)
        ↓                   ↓                   ↓
        └─────────┬─────────┴─────────┬─────────┘
                  ↓                   ↓
    ┌─────────────┼─────────────┬─────┴─────┐
    ↓             ↓             ↓           ↓
 Digest       Actions       Reports    Validation
 Email        Extract      Week/Month  (12项检查)
```

**Key modules:**
- `src/fetchers/` - Abstract `BaseFetcher` with multiple implementations
  - `reddit.py` - Reddit RSS/feedparser
  - `twitter.py` - Twitter (TwitterAPI.io)
  - `hackernews.py` - Hacker News (5 feed types via hnrss.org)
  - `rss.py` - Generic RSS/Atom feeds
- `src/processors/` - AI processing with enhanced JSON output (summary, key_points, impact, actions)
  - `base.py` - `ProcessingResult` with enhanced fields, prompt templates
  - `claude_cli.py` / `openai_cli.py` - CLI wrappers with batch processing
  - `ollama_api.py` - Ollama HTTP API for local LLM processing
  - `digest_processor.py` - Digest generation + action extraction
  - `event_stream.py` - Event clustering (hybrid: rule-based + semantic + AI confirmation)
  - `embeddings.py` - Embedding providers (sentence-transformers, OpenAI) + manager
  - `rule_classifier.py` - Pre-classification with 40+ domain patterns
- `src/analytics/` - Analysis modules
  - `topic_radar.py` - Topic matching, trend detection, snapshots
  - `topic_discovery.py` - Automatic topic discovery (entity frequency, keyword clustering, trend detection)
  - `reports.py` - Weekly/monthly report generation
  - `token_tracker.py` - AI call tracking, cost estimation, cache hit rates
- `src/optimization/` - Performance tuning
  - `cache_optimizer.py` - Cache metrics, efficiency analysis, cleanup
  - `dedup_optimizer.py` - Title/content hash for duplicate detection
- `src/validation/` - Data integrity
  - `data_validator.py` - 12-check validator with auto-fix
  - `diagnostic_queries.py` - SQL diagnostics for data issues
- `src/storage/` - SQLite with FTS5 full-text search
  - `models.py` - All data models (ContentItem, EventCluster, Topic, ActionItem, etc.)
  - `database.py` - Async/sync interfaces with all CRUD operations
- `src/delivery/` - SMTP email with Jinja2 HTML templates
- `src/scheduler/` - APScheduler for periodic fetch and daily digest jobs
- `config/settings.py` - Dataclass-based configuration from environment variables
- `config/ai_models.yaml` - Model tier configuration (heavy/light models)

**Data flow:**
1. Subscriptions → Fetch → Store in SQLite
2. AI Process → Enhanced JSON (summary + key_points + impact + actions)
3. Event clustering → Group related content
4. Topic matching → Associate with defined topics
5. Action extraction → Create actionable items from high-importance content
6. Generate HTML → Send email digest
7. Periodic reports → Weekly/monthly summaries

## Daily Usage

**For manual daily use (recommended):**
```bash
bb daily              # Run full pipeline, send email
bb daily --dry-run    # Preview without sending
bb daily --skip-fetch # Skip fetch if already have content
```

The `bb daily` command runs the complete pipeline in one shot:
```
bb daily
  ├── 1. fetch            # Fetch from all subscriptions
  ├── 2. embeddings       # Local sentence-transformers (FREE, enables dedup)
  ├── 3. process_content  # AI processing (consumes LLM tokens, with dedup)
  ├── 4. topic discovery  # Regex + statistics (FREE)
  ├── 5. clustering       # Event grouping (consumes LLM tokens)
  └── 6. digest + email   # Generate and send
```

**For 24/7 servers (daemon mode):**
```bash
# Configure .env
FETCH_INTERVAL_HOURS=4
DIGEST_SEND_TIME=07:00

# Start daemon
bb run
```

**Token Cost Summary:**
| Task | LLM Token Cost |
|------|----------------|
| `process_content` | Yes (required) |
| `compute_embeddings` | No (local model) |
| `discover_topics` | No (pure statistics) |
| `run_clustering` | Yes (with caching optimization) |

## Data Validation Checks

The `bb validate` command runs 12 integrity checks:
1. Orphan content items (invalid subscription_id)
2. Duplicate external IDs
3. Processed items without summaries
4. Invalid importance scores (must be 1-10)
5. Empty event clusters
6. Cluster article count mismatches
7. Missing FTS index entries
8. Expired cache entries
9. Orphan event members
10. Orphan topic associations
11. Orphan action items
12. Duplicate cluster memberships

Use `--fix` to auto-repair identified issues.

## Performance Optimizations

**Parallel Clustering:**
- ThreadPoolExecutor with configurable workers (default: 4)
- Thread-local database connections to avoid asyncio conflicts
- 3-4x speedup when AI calls are needed
- Thread-safe progress tracking with locks

**Caching Strategy (3-tier):**
1. Module-level LRU caches - Entity/keyword extraction (1024 entries)
2. Class-level TTL cache - Recent clusters (60s default)
3. Persistent database cache - Event confirmations with TTL

**Token Optimization:**
- Rule-based pre-classification (DOMAIN_CATEGORIES with 40+ patterns)
- Light vs heavy model tier selection based on importance score
- Batch processing (short: 12, long: 6 items per call)
- Content truncation (1500 chars content, 200 chars title)

**Database Indexes:**
- `event_members(content_id)` - Fast membership lookups
- `event_members(cluster_id)` - Fast cluster member queries

## Token Tracking

The `TokenTracker` (singleton) tracks 8 AI call types:
- `CONTENT_HEAVY` (825 tokens) - High-importance content
- `CONTENT_LIGHT` (615 tokens) - Low-importance content
- `CONTENT_BATCH` (215 tokens) - Batch processing
- `EVENT_CONFIRM` (220 tokens) - Cluster confirmation
- `EVENT_TITLE` (50 tokens) - Event title generation
- `TIMELINE_SUMMARY` (150 tokens) - Timeline summaries
- `DIGEST_GENERATE` (500 tokens) - Digest generation
- `REPORT` (800 tokens) - Report generation

Cost estimation supports Haiku ($0.25/$1.25), Sonnet ($3/$15), Opus ($15/$75) per 1M tokens.

## Database Schema

**Core tables:**
- `subscriptions` - Source subscriptions (Reddit/Twitter/HackerNews/RSS)
- `content_items` - Fetched content with AI processing results
- `user_profiles` - User preferences
- `content_fts` - FTS5 virtual table for full-text search

**Phase 2 - Event Stream:**
- `event_clusters` - Event groupings
- `event_members` - Content-event associations
- `event_timeline` - Daily event summaries

**Embeddings (Semantic Clustering):**
- `content_embeddings` - Content embedding vectors (content_id, embedding, model, dimension)
- `cluster_embeddings` - Cluster centroid vectors (cluster_id, centroid_embedding, member_count)

**Phase 3 - Topic Radar:**
- `topics` - Topic definitions with keywords
- `content_topics` - Content-topic associations
- `topic_snapshots` - Periodic topic summaries
- `topic_suggestions` - Auto-discovered topic suggestions (name, keywords, frequency, confidence, status)

**Phase 5 - Action List:**
- `action_items` - Extracted actions with priority/status
- `triggers` - User-defined automation rules

**Caching:**
- `ai_cache` - AI processing cache with TTL-based expiration

## Key Patterns

- **Enhanced AI output:** ProcessingResult includes `one_liner`, `key_points`, `impact_assessment`, `actionable_items`
- **Triple AI providers:** `AI_PROVIDER` env var controls Claude CLI, Codex CLI, or Ollama selection
- **Async/sync mix:** Fetchers are async, processors use sync subprocess, database has both interfaces
- **Mock mode:** `--mock` flag enables `MockAIProcessor` and `MockTwitterFetcher` for testing
- **JSON AI responses:** Processors parse JSON from CLI output, handle markdown code blocks, fallback to defaults
- **Token optimization:** Rule-based pre-classification skips AI for low-value content
- **Embedding dedup:** Semantic similarity (>=85%) reuses AI results from similar processed items
- **Batch processing:** Multiple items processed in single AI call for efficiency
- **FTS5 search:** Full-text search with filters (category, date, importance, state)
- **State management:** Content items have state (unread/read/saved/archived/flagged)
- **Model tiering:** Heavy model (importance >= 7) vs light model, with quality protection
- **Parallel clustering:** ThreadPoolExecutor with configurable workers (default: 4)
- **3-tier caching:** Module LRU + class TTL + persistent DB cache for event confirmations
- **Auto-accept threshold:** High-confidence matches (score >= 0.6) skip AI confirmation
- **Duplicate prevention:** UNIQUE constraint on content_item_id in event_members
- **Token tracking:** Per-call-type tracking with cost estimation (Haiku/Sonnet/Opus pricing)
- **Data validation:** 12 integrity checks with auto-fix capability
- **Hybrid similarity:** 40% rule-based + 60% semantic (embedding cosine similarity) for event clustering
- **Embedding providers:** Pluggable providers (sentence-transformers local, OpenAI API) with lazy loading
- **Topic discovery:** Entity frequency analysis, keyword bigram clustering, trend spike detection (3x week-over-week)
- **Automated preprocessing:** Digest job automatically runs embeddings + topic discovery before sending (zero extra LLM cost)
- **Two-stage processing (Ollama):** Optional 8B screening + 32B refinement mode (`OLLAMA_MODEL_SCREEN`) for 20-40% time savings

## Data Models

**ContentItem enhanced fields (Phase 1):**
```python
one_liner: str           # One sentence conclusion
key_points: str          # JSON: [{type, value, impact}]
impact_assessment: str   # JSON: {short_term, long_term, certainty}
actionable_items: str    # JSON: [{type, description, priority}]
state: ItemState         # unread/read/saved/archived/flagged
```

**ProcessingResult (base.py):**
```python
summary: str
category: str
importance_score: int
one_liner: str
key_points: list[KeyPointResult]
impact_assessment: Optional[ImpactResult]
actionable_items: list[ActionResult]
```

## Configuration

Copy `.env.example` to `.env`. Key settings:
- `AI_PROVIDER` - "claude", "codex", "ollama", or "auto"
- `CLAUDE_MODEL_HEAVY` - Claude heavy model (default: sonnet)
- `CLAUDE_MODEL_LIGHT` - Claude light model (default: haiku)
- `CODEX_MODEL_HEAVY` - Codex heavy model (default: gpt-5.2-codex)
- `CODEX_MODEL_LIGHT` - Codex light model (default: gpt-5.1-codex-mini)
- `OLLAMA_BASE_URL` - Ollama API URL (default: http://localhost:11434)
- `OLLAMA_MODEL` - Ollama model name (default: qwen3:32b)
- `OLLAMA_MODEL_SCREEN` - 8B model for two-stage screening (empty = disabled)
- `OLLAMA_TIMEOUT` - Ollama request timeout in seconds (default: 120)
- `MODEL_TIER_ENABLED` - Enable model tier selection (default: true)
- `MODEL_TIER_HEAVY_THRESHOLD` - Importance threshold for heavy model (default: 7)
- `MODEL_TIER_CONFIDENCE_CUTOFF` - Confidence threshold for heavy model (default: 0.5)
- `QUALITY_UNKNOWN_DOMAIN_USE_HEAVY` - Use heavy model for unknown domains (default: true)
- `QUALITY_REPROCESS_HIGH_SCORE` - Reprocess if light model returns high score (default: true)
- `TWITTERAPI_IO_KEY` - TwitterAPI.io API key for Twitter data
- `SMTP_*` - Email delivery configuration
- `DIGEST_SEND_TIME` - Daily digest time (HH:MM)
- `FETCH_INTERVAL_HOURS` - Content fetch frequency
- `AI_CACHE_ENABLED` - Enable AI response caching (default: true)
- `AI_CACHE_TTL` - Cache TTL in seconds (default: 86400)
- `AI_MAX_CONTENT_LENGTH` - Content truncation limit (default: 1500)
- `AI_BATCH_SIZE_SHORT` - Batch size for short content (default: 12)
- `AI_BATCH_SIZE_LONG` - Batch size for long content (default: 6)
- `AI_PROCESS_LIMIT` - Max items per AI processing run (default: 500)
- `CLUSTER_CACHE_TTL` - Cluster cache seconds (default: 60)
- `CLUSTER_RETRY_HOURS` - Retry failed clusters after hours (default: 24)
- `EMBEDDING_PROVIDER` - "sentence-transformers" (default, local) or "openai"
- `EMBEDDING_ENABLED` - Enable semantic similarity (default: true)
- `EMBEDDING_RULE_WEIGHT` - Rule-based similarity weight (default: 0.4)
- `EMBEDDING_SEMANTIC_WEIGHT` - Semantic similarity weight (default: 0.6)
- `RULE_ONLY_ENABLED` - Enable rule-only processing for high-confidence content (default: true)
- `RULE_ONLY_MAX_CONTENT` - Max content length for rule-only processing (default: 1000)
- `RULE_ONLY_MIN_BOOST` - Min keyword boost for rule-only (default: 1)
- `SKIP_ENABLED` - Enable skip processing for spam/jobs/etc (default: true)
- `MINIMAL_PROMPT_ENABLED` - Enable minimal prompt for low importance (default: true)
- `MINIMAL_PROMPT_THRESHOLD` - Importance threshold for minimal prompt (default: 3)
