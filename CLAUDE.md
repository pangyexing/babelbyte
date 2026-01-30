# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BabelByte is an AI-powered **personal intelligence product** that fetches content from Reddit and Twitter, processes it with AI (Claude CLI or Codex CLI), and delivers structured intelligence briefings via email. It supports event tracking, topic radar, knowledge base search, and periodic reports.

## Commands

**Installation:**
```bash
pip install -e .              # Basic install
pip install -e ".[dev]"       # With dev dependencies (pytest, black, ruff)
```

**CLI entry point:** `babelbyte` or `python main.py`

**Core commands:**
```bash
babelbyte subscribe reddit <subreddit>     # Subscribe to subreddit
babelbyte subscribe twitter <username>     # Subscribe to Twitter user
babelbyte list                             # Show subscriptions
babelbyte fetch                            # Fetch content from all sources
babelbyte digest --dry-run                 # Preview digest without sending
babelbyte digest                           # Generate and send email digest
babelbyte run                              # Start scheduler daemon
babelbyte --mock <command>                 # Test mode with mock data
```

**Knowledge Base (Phase 4):**
```bash
babelbyte search "query"                   # Full-text search
babelbyte search "AI" --category AI --from 2024-01-01
babelbyte browse --date yesterday          # Browse by date
babelbyte item 123 --open                  # View item details
babelbyte mark 123 saved                   # Mark item state (unread/read/saved/archived/flagged)
babelbyte stats                            # Category and state statistics
babelbyte rebuild-index                    # Rebuild FTS index
```

**Event Stream (Phase 2):**
```bash
babelbyte events --days 7                  # List recent event clusters
babelbyte event 123                        # View event details and members
babelbyte cluster --limit 100              # Run event clustering
```

**Topic Radar (Phase 3):**
```bash
babelbyte topics                           # List all topics
babelbyte topic add "AI" --keywords "GPT,ChatGPT,AI"
babelbyte topic show "AI"                  # View topic details
babelbyte topic delete "AI" --yes
```

**Action List (Phase 5):**
```bash
babelbyte actions --status pending         # List action items
babelbyte action 123 done                  # Mark action as done/dismissed
```

**Reports (Phase 6):**
```bash
babelbyte report week                      # Generate weekly report
babelbyte report month --months-ago 1      # Generate monthly report
```

**Code quality:**
```bash
black .                                    # Format (line length: 100)
ruff check . --select E,F,W,I              # Lint
```

## Architecture

```
CLI (Click) → Scheduler (APScheduler)
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
Fetchers      Event Stream    Topic Radar
(Reddit/Twitter)    ↓               ↓
    ↓          Clustering      Matching
    └──────→ SQLite + FTS5 ←───────┘
                    ↓
            AI Processors
         (Claude/Codex CLI)
                    ↓
         ┌─────────┼─────────┐
         ↓         ↓         ↓
    Digest    Actions    Reports
    Email     Extract    Weekly/Monthly
```

**Key modules:**
- `src/fetchers/` - Abstract `BaseFetcher` with Reddit (RSS/feedparser) and Twitter (tweepy) implementations
- `src/processors/` - AI processing with enhanced JSON output (summary, key_points, impact, actions)
  - `base.py` - `ProcessingResult` with enhanced fields, prompt templates
  - `claude_cli.py` / `openai_cli.py` - CLI wrappers with batch processing
  - `digest_processor.py` - Digest generation + action extraction
  - `event_stream.py` - Event clustering (rule-based + AI confirmation)
  - `rule_classifier.py` - Pre-classification to save tokens
- `src/analytics/` - Analysis modules
  - `topic_radar.py` - Topic matching, trend detection, snapshots
  - `reports.py` - Weekly/monthly report generation
- `src/storage/` - SQLite with FTS5 full-text search
  - `models.py` - All data models (ContentItem, EventCluster, Topic, ActionItem, etc.)
  - `database.py` - Async/sync interfaces with all CRUD operations
- `src/delivery/` - SMTP email with Jinja2 HTML templates
- `src/scheduler/` - APScheduler for periodic fetch and daily digest jobs
- `config/settings.py` - Dataclass-based configuration from environment variables

**Data flow:**
1. Subscriptions → Fetch → Store in SQLite
2. AI Process → Enhanced JSON (summary + key_points + impact + actions)
3. Event clustering → Group related content
4. Topic matching → Associate with defined topics
5. Action extraction → Create actionable items from high-importance content
6. Generate HTML → Send email digest
7. Periodic reports → Weekly/monthly summaries

## Database Schema

**Core tables:**
- `subscriptions` - Source subscriptions (Reddit/Twitter)
- `content_items` - Fetched content with AI processing results
- `user_profiles` - User preferences
- `content_fts` - FTS5 virtual table for full-text search

**Phase 2 - Event Stream:**
- `event_clusters` - Event groupings
- `event_members` - Content-event associations
- `event_timeline` - Daily event summaries

**Phase 3 - Topic Radar:**
- `topics` - Topic definitions with keywords
- `content_topics` - Content-topic associations
- `topic_snapshots` - Periodic topic summaries

**Phase 5 - Action List:**
- `action_items` - Extracted actions with priority/status
- `triggers` - User-defined automation rules

## Key Patterns

- **Enhanced AI output:** ProcessingResult includes `one_liner`, `key_points`, `impact_assessment`, `actionable_items`
- **Dual AI providers:** `AI_PROVIDER` env var controls Claude vs Codex CLI selection
- **Async/sync mix:** Fetchers are async, processors use sync subprocess, database has both interfaces
- **Mock mode:** `--mock` flag enables `MockAIProcessor` and `MockTwitterFetcher` for testing
- **JSON AI responses:** Processors parse JSON from CLI output, handle markdown code blocks, fallback to defaults
- **Token optimization:** Rule-based pre-classification skips AI for low-value content
- **Batch processing:** Multiple items processed in single AI call for efficiency
- **FTS5 search:** Full-text search with filters (category, date, importance, state)
- **State management:** Content items have state (unread/read/saved/archived/flagged)

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
- `AI_PROVIDER` - "claude", "codex", or "auto"
- `TWITTER_BEARER_TOKEN` - Optional, for Twitter API
- `SMTP_*` - Email delivery configuration
- `DIGEST_SEND_TIME` - Daily digest time (HH:MM)
- `FETCH_INTERVAL_HOURS` - Content fetch frequency
