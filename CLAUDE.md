# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BabelByte is an AI-powered content subscription system that fetches content from Reddit and Twitter, processes it with AI (Claude CLI or Codex CLI), and delivers Chinese-language daily digests via email.

## Commands

**Installation:**
```bash
pip install -e .              # Basic install
pip install -e ".[dev]"       # With dev dependencies (pytest, black, ruff)
```

**CLI entry point:** `babelbyte` or `python main.py`

**Key commands:**
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

**Code quality:**
```bash
black .                                    # Format (line length: 100)
ruff check . --select E,F,W,I              # Lint
```

## Architecture

```
CLI (Click) → Scheduler (APScheduler)
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
Fetchers                      Digest Generator
(Reddit RSS, Twitter API)           ↓
    ↓                         AI Processors
    └──────→ SQLite ←─────── (Claude/Codex CLI)
                                    ↓
                            Email Sender (SMTP)
```

**Key modules:**
- `src/fetchers/` - Abstract `BaseFetcher` with Reddit (RSS/feedparser) and Twitter (tweepy) implementations
- `src/processors/` - AI processing via Claude CLI or Codex CLI subprocess calls, returns JSON with summary/category/importance
- `src/storage/` - SQLite with both async (`Database`) and sync (`SyncDatabase`) interfaces
- `src/delivery/` - SMTP email with Jinja2 HTML templates
- `src/scheduler/` - APScheduler for periodic fetch and daily digest jobs
- `config/settings.py` - Dataclass-based configuration from environment variables

**Data flow:** Subscriptions → Fetch → Store in SQLite → AI Process (JSON response) → Filter by importance → Generate HTML → Send email

## Key Patterns

- **Dual AI providers:** `AI_PROVIDER` env var controls Claude vs Codex CLI selection (`get_ai_processor()` factory in `digest_processor.py`)
- **Async/sync mix:** Fetchers are async, processors use sync subprocess, database has both interfaces
- **Mock mode:** `--mock` flag enables `MockAIProcessor` and `MockTwitterFetcher` for testing without external services
- **JSON AI responses:** Processors parse JSON from CLI output, handle markdown code blocks, fallback to defaults on parse failure

## Configuration

Copy `.env.example` to `.env`. Key settings:
- `AI_PROVIDER` - "claude", "codex", or "auto"
- `TWITTER_BEARER_TOKEN` - Optional, for Twitter API
- `SMTP_*` - Email delivery configuration
- `DIGEST_SEND_TIME` - Daily digest time (HH:MM)
- `FETCH_INTERVAL_HOURS` - Content fetch frequency
