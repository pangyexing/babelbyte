"""CLI commands for BabelByte."""

import sys
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from config.settings import get_settings, reload_settings
from src.delivery.email_sender import EmailSender
from src.processors.digest_processor import DigestGenerator, create_digest_preview
from src.scheduler.jobs import BabelByteScheduler, JobRunner
from src.storage.database import SyncDatabase
from src.storage.models import (
    ActionItem,
    ActionStatus,
    ItemState,
    SourceType,
    Subscription,
    SubscriptionType,
    Topic,
)

console = Console()


def get_db() -> SyncDatabase:
    """Get database connection."""
    db = SyncDatabase()
    db.connect()
    return db


@click.group()
@click.option("--mock", is_flag=True, help="Use mock data for testing")
@click.pass_context
def cli(ctx, mock):
    """BabelByte - AI 内容订阅系统"""
    ctx.ensure_object(dict)
    ctx.obj["mock"] = mock


@cli.command()
@click.argument("source", type=click.Choice(["reddit", "twitter"]))
@click.argument("name")
@click.option("--user", "-u", is_flag=True, help="Subscribe to a user instead of subreddit (Reddit only)")
@click.pass_context
def subscribe(ctx, source, name, user):
    """Subscribe to a content source.

    Examples:
        babelbyte subscribe reddit MachineLearning
        babelbyte subscribe reddit --user spez
        babelbyte subscribe twitter elonmusk
    """
    # Determine subscription type
    if source == "reddit":
        source_type = SourceType.REDDIT
        if user:
            sub_type = SubscriptionType.REDDIT_USER
            # Clean up name (remove u/ prefix if present)
            name = name.removeprefix("u/").removeprefix("/u/")
        else:
            sub_type = SubscriptionType.SUBREDDIT
            # Clean up name (remove r/ prefix if present)
            name = name.removeprefix("r/").removeprefix("/r/")
    else:
        source_type = SourceType.TWITTER
        sub_type = SubscriptionType.TWITTER_USER
        # Clean up name (remove @ prefix if present)
        name = name.lstrip("@")

    db = get_db()
    try:
        # Check if already exists
        existing = db.get_subscription_by_name(source_type, sub_type, name)
        if existing:
            if existing.enabled:
                console.print(f"[yellow]Already subscribed to {existing.display_name}[/yellow]")
            else:
                # Re-enable
                existing.enabled = True
                db.update_subscription(existing)
                console.print(f"[green]Re-enabled subscription: {existing.display_name}[/green]")
            return

        # Create new subscription
        sub = Subscription(
            source_type=source_type,
            subscription_type=sub_type,
            name=name,
            enabled=True,
            created_at=datetime.now(),
        )
        db.add_subscription(sub)
        console.print(f"[green]✓ Subscribed to {sub.display_name}[/green]")

    finally:
        db.close()


@cli.command()
@click.argument("source", type=click.Choice(["reddit", "twitter"]))
@click.argument("name")
@click.option("--user", "-u", is_flag=True, help="Unsubscribe from a user (Reddit only)")
@click.option("--delete", "-d", is_flag=True, help="Permanently delete instead of disable")
@click.pass_context
def unsubscribe(ctx, source, name, user, delete):
    """Unsubscribe from a content source.

    Examples:
        babelbyte unsubscribe reddit MachineLearning
        babelbyte unsubscribe twitter elonmusk
    """
    # Determine subscription type
    if source == "reddit":
        source_type = SourceType.REDDIT
        sub_type = SubscriptionType.REDDIT_USER if user else SubscriptionType.SUBREDDIT
        name = name.removeprefix("r/").removeprefix("/r/").removeprefix("u/").removeprefix("/u/")
    else:
        source_type = SourceType.TWITTER
        sub_type = SubscriptionType.TWITTER_USER
        name = name.lstrip("@")

    db = get_db()
    try:
        sub = db.get_subscription_by_name(source_type, sub_type, name)
        if not sub:
            console.print(f"[yellow]Not subscribed to {name}[/yellow]")
            return

        if delete:
            db.delete_subscription(sub.id)
            console.print(f"[red]✗ Deleted subscription: {sub.display_name}[/red]")
        else:
            sub.enabled = False
            db.update_subscription(sub)
            console.print(f"[yellow]✗ Disabled subscription: {sub.display_name}[/yellow]")

    finally:
        db.close()


@cli.command("list")
@click.option("--all", "-a", "show_all", is_flag=True, help="Show disabled subscriptions too")
@click.pass_context
def list_subs(ctx, show_all):
    """List all subscriptions."""
    db = get_db()
    try:
        subs = db.list_subscriptions(enabled_only=not show_all)

        if not subs:
            console.print("[dim]No subscriptions found. Use 'subscribe' to add some.[/dim]")
            return

        table = Table(title="📚 Subscriptions")
        table.add_column("ID", style="dim")
        table.add_column("Source", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Status")
        table.add_column("Last Fetched", style="dim")

        for sub in subs:
            status = "[green]✓ Enabled[/green]" if sub.enabled else "[red]✗ Disabled[/red]"
            last_fetched = (
                sub.last_fetched_at.strftime("%Y-%m-%d %H:%M")
                if sub.last_fetched_at
                else "Never"
            )
            table.add_row(
                str(sub.id),
                sub.source_type.value.title(),
                sub.display_name,
                status,
                last_fetched,
            )

        console.print(table)

    finally:
        db.close()


@cli.command()
@click.pass_context
def fetch(ctx):
    """Fetch content from all enabled subscriptions."""
    mock = ctx.obj.get("mock", False)

    runner = JobRunner(use_mock=mock)
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[green]{task.fields[new_items]} new[/green]"),
            console=console,
        ) as progress:
            task = progress.add_task("[bold]Fetching...[/bold]", total=None, new_items=0)

            def on_progress(name, current, total, new_items):
                progress.update(task, total=total, completed=current, new_items=new_items)
                progress.update(task, description=f"[bold]{name}[/bold]")

            stats = runner.fetch_all_content(progress_callback=on_progress)

        console.print(
            Panel(
                f"[green]Fetched:[/green] {stats['fetched']}/{stats['total']} subscriptions\n"
                f"[green]New items:[/green] {stats['new_items']}\n"
                f"[red]Errors:[/red] {stats['errors']}",
                title="Fetch Results",
            )
        )

    finally:
        runner.close_db()


@cli.command()
@click.option("--dry-run", is_flag=True, help="Preview digest without sending")
@click.option("--min-importance", "-m", default=5, help="Minimum importance score (1-10)")
@click.option("--max-items", "-n", default=30, help="Maximum items in digest")
@click.option("--provider", "-p", type=click.Choice(["claude", "codex", "auto"]), default=None, help="AI provider to use")
@click.pass_context
def digest(ctx, dry_run, min_importance, max_items, provider):
    """Generate and send the daily digest."""
    mock = ctx.obj.get("mock", False)

    db = get_db()
    try:
        generator = DigestGenerator(db=db, provider=provider, use_mock=mock)

        # First process unprocessed items with progress bar
        console.print("[bold]Processing content with AI...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Initializing...", total=None)

            def update_progress(phase: str, current: int, total: int):
                progress.update(task, description=phase, completed=current, total=total)

            processed = generator.process_unprocessed_items(
                limit=100, progress_callback=update_progress
            )

        console.print(f"[dim]Processed {processed} items[/dim]")

        # Generate digest
        console.print("[bold]Generating digest...[/bold]")
        digest_result = generator.generate_digest(
            min_importance=min_importance,
            max_items=max_items,
        )

        if not digest_result.items:
            console.print("[yellow]No items to include in digest[/yellow]")
            return

        # Show preview
        preview = create_digest_preview(digest_result)
        console.print(preview)

        if dry_run:
            console.print("\n[yellow]Dry run - email not sent[/yellow]")
            return

        # Send email
        sender = EmailSender()
        if not sender.is_configured():
            console.print("[red]Email not configured. Please set SMTP settings in .env[/red]")
            return

        console.print("\n[bold]Sending email...[/bold]")
        result = sender.send_digest(digest_result)

        if result.success:
            generator.mark_digest_delivered(digest_result)
            console.print(f"[green]✓ {result.message}[/green]")
        else:
            console.print(f"[red]✗ {result.message}[/red]")

    finally:
        db.close()


@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration."""
    settings = get_settings()

    console.print(Panel("[bold]BabelByte Configuration[/bold]"))

    # Database
    console.print(f"\n[bold]Database:[/bold]")
    console.print(f"  Path: {settings.database.path}")

    # Twitter
    console.print(f"\n[bold]Twitter:[/bold]")
    if settings.twitter.is_configured:
        console.print("  [green]✓ Configured[/green]")
    else:
        console.print("  [red]✗ Not configured (set TWITTER_BEARER_TOKEN)[/red]")

    # Email
    console.print(f"\n[bold]Email:[/bold]")
    if settings.email.is_configured:
        console.print("  [green]✓ Configured[/green]")
        console.print(f"  Host: {settings.email.host}:{settings.email.port}")
        console.print(f"  From: {settings.email.from_addr}")
        console.print(f"  To: {settings.email.to_addr}")
    else:
        console.print("  [red]✗ Not configured (set SMTP settings in .env)[/red]")

    # Scheduler
    console.print(f"\n[bold]Scheduler:[/bold]")
    console.print(f"  Fetch interval: every {settings.scheduler.fetch_interval_hours} hours")
    console.print(f"  Digest time: {settings.scheduler.digest_send_time}")

    # AI Provider
    console.print(f"\n[bold]AI Provider:[/bold]")
    console.print(f"  Current: {settings.ai.provider}")

    # Claude CLI
    console.print(f"\n[bold]Claude CLI:[/bold]")
    console.print(f"  Path: {settings.claude.cli_path}")

    # Codex CLI
    console.print(f"\n[bold]Codex CLI:[/bold]")
    console.print(f"  Path: {settings.codex.cli_path}")


@cli.command("test-email")
@click.option("--to", "-t", "to_addr", help="Override recipient email")
@click.pass_context
def test_email(ctx, to_addr):
    """Send a test email to verify configuration."""
    sender = EmailSender()

    if not sender.is_configured():
        console.print("[red]Email not configured. Please set SMTP settings in .env[/red]")
        return

    console.print("[bold]Sending test email...[/bold]")
    result = sender.send_test_email(to_addr)

    if result.success:
        console.print(f"[green]✓ {result.message}[/green]")
    else:
        console.print(f"[red]✗ {result.message}[/red]")


@cli.command()
@click.pass_context
def run(ctx):
    """Run the scheduler (foreground process)."""
    mock = ctx.obj.get("mock", False)

    console.print("[bold]Starting BabelByte scheduler...[/bold]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    scheduler = BabelByteScheduler(use_mock=mock)

    try:
        scheduler.start()

        # Show scheduled jobs
        jobs = scheduler.get_jobs()
        table = Table(title="Scheduled Jobs")
        table.add_column("Job")
        table.add_column("Next Run")

        for job in jobs:
            next_run = job["next_run"].strftime("%Y-%m-%d %H:%M") if job["next_run"] else "N/A"
            table.add_row(job["name"], next_run)

        console.print(table)
        console.print("\n[green]Scheduler running...[/green]")

        # Keep running
        import time
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping scheduler...[/yellow]")
        scheduler.stop()
        console.print("[green]Scheduler stopped[/green]")


# ============================================
# Phase 4: Knowledge Base Commands
# ============================================


@cli.command()
@click.argument("query")
@click.option("--category", "-c", help="Filter by category")
@click.option("--from", "from_date", help="Filter from date (YYYY-MM-DD)")
@click.option("--to", "to_date", help="Filter to date (YYYY-MM-DD)")
@click.option("--min-importance", "-m", type=int, help="Minimum importance score")
@click.option("--limit", "-n", default=20, help="Maximum results")
@click.pass_context
def search(ctx, query, category, from_date, to_date, min_importance, limit):
    """Search content in the knowledge base.

    Examples:
        babelbyte search "GPT"
        babelbyte search "AI" --category AI --from 2024-01-01
        babelbyte search "融资" --min-importance 7
    """
    db = get_db()
    try:
        results = db.search_content(
            query=query,
            category=category,
            from_date=from_date,
            to_date=to_date,
            min_importance=min_importance,
            limit=limit,
        )

        if not results:
            console.print(f"[dim]No results found for '{query}'[/dim]")
            return

        console.print(f"[bold]Found {len(results)} results for '{query}':[/bold]\n")

        for item in results:
            importance_color = "red" if item.importance_score and item.importance_score >= 8 else "yellow" if item.importance_score and item.importance_score >= 5 else "dim"
            console.print(f"[{importance_color}][{item.importance_score or '-'}/10][/{importance_color}] ", end="")
            console.print(f"[cyan][{item.category or '未分类'}][/cyan] ", end="")
            console.print(f"[bold]{item.title[:60]}{'...' if len(item.title) > 60 else ''}[/bold]")
            if item.summary:
                console.print(f"    {item.summary[:100]}{'...' if len(item.summary) > 100 else ''}")
            console.print(f"    [dim]ID: {item.id} | {item.published_at.strftime('%Y-%m-%d')} | {item.url[:50]}...[/dim]\n")

    finally:
        db.close()


@cli.command()
@click.option("--date", "-d", default="today", help="Date to browse (YYYY-MM-DD or 'today', 'yesterday')")
@click.option("--category", "-c", help="Filter by category")
@click.option("--limit", "-n", default=50, help="Maximum results")
@click.pass_context
def browse(ctx, date, category, limit):
    """Browse content by date.

    Examples:
        babelbyte browse
        babelbyte browse --date yesterday
        babelbyte browse --date 2024-01-15 --category AI
    """
    from datetime import timedelta

    # Handle date shortcuts
    if date == "today":
        date = datetime.now().strftime("%Y-%m-%d")
    elif date == "yesterday":
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    db = get_db()
    try:
        results = db.browse_by_date(date=date, category=category, limit=limit)

        if not results:
            console.print(f"[dim]No content found for {date}[/dim]")
            return

        console.print(f"[bold]Content from {date}:[/bold]\n")

        # Group by category
        by_category: dict[str, list] = {}
        for item in results:
            cat = item.category or "未分类"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)

        for cat, items in sorted(by_category.items()):
            console.print(f"[cyan][{cat}] ({len(items)} items)[/cyan]")
            for item in items:
                state_icon = {"unread": "", "read": "", "saved": "", "flagged": "", "archived": ""}.get(item.state.value, "")
                console.print(f"  {state_icon} [{item.importance_score or '-'}/10] {item.title[:50]}{'...' if len(item.title) > 50 else ''} [dim](ID: {item.id})[/dim]")
            console.print()

    finally:
        db.close()


@cli.command()
@click.argument("item_id", type=int)
@click.option("--open", "-o", "open_url", is_flag=True, help="Open URL in browser")
@click.pass_context
def item(ctx, item_id, open_url):
    """View a content item details.

    Examples:
        babelbyte item 123
        babelbyte item 123 --open
    """
    db = get_db()
    try:
        content = db.get_content_item(item_id)

        if not content:
            console.print(f"[red]Item {item_id} not found[/red]")
            return

        # Display item details
        console.print(Panel(f"[bold]{content.title}[/bold]", title=f"Item #{item_id}"))

        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        table.add_row("Category", content.category or "未分类")
        table.add_row("Importance", f"{content.importance_score or '-'}/10")
        table.add_row("State", content.state.value)
        table.add_row("Author", content.author)
        table.add_row("Published", content.published_at.strftime("%Y-%m-%d %H:%M"))
        table.add_row("URL", content.url)

        console.print(table)

        if content.summary:
            console.print(f"\n[bold]Summary:[/bold]\n{content.summary}")

        if content.one_liner:
            console.print(f"\n[bold]One-liner:[/bold]\n{content.one_liner}")

        # Show enhanced data
        enhanced = content.get_enhanced_data()
        if enhanced:
            if enhanced.key_points:
                console.print("\n[bold]Key Points:[/bold]")
                for kp in enhanced.key_points:
                    console.print(f"  - [{kp.type}] {kp.value}: {kp.impact}")

            if enhanced.impact_assessment:
                console.print("\n[bold]Impact Assessment:[/bold]")
                if enhanced.impact_assessment.short_term:
                    console.print(f"  Short-term: {enhanced.impact_assessment.short_term}")
                if enhanced.impact_assessment.long_term:
                    console.print(f"  Long-term: {enhanced.impact_assessment.long_term}")

            if enhanced.actionable_items:
                console.print("\n[bold]Action Items:[/bold]")
                for action in enhanced.actionable_items:
                    console.print(f"  - [{action.priority}] {action.type}: {action.description}")

        if open_url and content.url:
            import webbrowser
            webbrowser.open(content.url)
            console.print("\n[green]Opened URL in browser[/green]")

    finally:
        db.close()


@cli.command()
@click.argument("item_id", type=int)
@click.argument("state", type=click.Choice(["unread", "read", "saved", "archived", "flagged"]))
@click.pass_context
def mark(ctx, item_id, state):
    """Mark a content item with a state.

    Examples:
        babelbyte mark 123 saved
        babelbyte mark 123 read
        babelbyte mark 123 flagged
    """
    db = get_db()
    try:
        content = db.get_content_item(item_id)
        if not content:
            console.print(f"[red]Item {item_id} not found[/red]")
            return

        new_state = ItemState(state)
        db.update_item_state(item_id, new_state)
        console.print(f"[green]Marked item {item_id} as {state}[/green]")

    finally:
        db.close()


@cli.command()
@click.pass_context
def stats(ctx):
    """Show knowledge base statistics."""
    db = get_db()
    try:
        category_stats = db.get_category_stats()
        state_stats = db.get_state_stats()

        console.print(Panel("[bold]Knowledge Base Statistics[/bold]"))

        # Category stats
        console.print("\n[bold]By Category:[/bold]")
        table = Table()
        table.add_column("Category")
        table.add_column("Count", justify="right")
        for cat, count in category_stats.items():
            table.add_row(cat, str(count))
        console.print(table)

        # State stats
        console.print("\n[bold]By State:[/bold]")
        table = Table()
        table.add_column("State")
        table.add_column("Count", justify="right")
        for state, count in state_stats.items():
            table.add_row(state, str(count))
        console.print(table)

        total = sum(category_stats.values())
        console.print(f"\n[bold]Total items:[/bold] {total}")

    finally:
        db.close()


@cli.command("rebuild-index")
@click.pass_context
def rebuild_index(ctx):
    """Rebuild the full-text search index."""
    db = get_db()
    try:
        console.print("[bold]Rebuilding FTS index...[/bold]")
        count = db.rebuild_fts_index()
        console.print(f"[green]Indexed {count} items[/green]")

    finally:
        db.close()


# ============================================
# Phase 5: Action List Commands
# ============================================


@cli.command()
@click.option("--status", "-s", type=click.Choice(["pending", "done", "dismissed"]), default="pending", help="Filter by status")
@click.option("--priority", "-p", type=click.Choice(["高", "中", "低"]), help="Filter by priority")
@click.option("--limit", "-n", default=20, help="Maximum results")
@click.pass_context
def actions(ctx, status, priority, limit):
    """List action items.

    Examples:
        babelbyte actions
        babelbyte actions --status pending --priority 高
    """
    db = get_db()
    try:
        action_status = ActionStatus(status) if status else None
        items = db.get_action_items(status=action_status, priority=priority, limit=limit)

        if not items:
            console.print("[dim]No action items found[/dim]")
            return

        console.print(f"[bold]Action Items ({len(items)}):[/bold]\n")

        table = Table()
        table.add_column("ID", style="dim")
        table.add_column("Priority")
        table.add_column("Type")
        table.add_column("Description")
        table.add_column("Status")
        table.add_column("Created", style="dim")

        for item in items:
            priority_color = {"高": "red", "中": "yellow", "低": "dim"}.get(item.priority, "white")
            table.add_row(
                str(item.id),
                f"[{priority_color}]{item.priority}[/{priority_color}]",
                item.type,
                item.description[:40] + ("..." if len(item.description) > 40 else ""),
                item.status.value,
                item.created_at.strftime("%Y-%m-%d"),
            )

        console.print(table)

    finally:
        db.close()


@cli.command("action")
@click.argument("action_id", type=int)
@click.argument("new_status", type=click.Choice(["done", "dismissed"]))
@click.pass_context
def update_action(ctx, action_id, new_status):
    """Update action item status.

    Examples:
        babelbyte action 123 done
        babelbyte action 123 dismissed
    """
    db = get_db()
    try:
        db.update_action_status(action_id, ActionStatus(new_status))
        console.print(f"[green]Action {action_id} marked as {new_status}[/green]")

    finally:
        db.close()


# ============================================
# Phase 3: Topic Commands
# ============================================


@cli.command()
@click.pass_context
def topics(ctx):
    """List all topics."""
    db = get_db()
    try:
        topic_list = db.list_topics()

        if not topic_list:
            console.print("[dim]No topics defined. Use 'topic add' to create one.[/dim]")
            return

        table = Table(title="Topics")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Keywords")
        table.add_column("Created", style="dim")

        for topic in topic_list:
            keywords = ", ".join(topic.get_keywords()[:5])
            if len(topic.get_keywords()) > 5:
                keywords += "..."
            table.add_row(
                str(topic.id),
                topic.name,
                keywords or "-",
                topic.created_at.strftime("%Y-%m-%d"),
            )

        console.print(table)

    finally:
        db.close()


@cli.group()
@click.pass_context
def topic(ctx):
    """Topic management commands."""
    pass


@topic.command("add")
@click.argument("name")
@click.option("--keywords", "-k", help="Comma-separated keywords")
@click.option("--description", "-d", help="Topic description")
@click.pass_context
def topic_add(ctx, name, keywords, description):
    """Add a new topic.

    Examples:
        babelbyte topic add "AI应用" --keywords "GPT,ChatGPT,AI,人工智能"
        babelbyte topic add "OpenAI" --keywords "OpenAI,Sam Altman" --description "OpenAI company news"
    """
    import json

    db = get_db()
    try:
        # Check if exists
        existing = db.get_topic_by_name(name)
        if existing:
            console.print(f"[yellow]Topic '{name}' already exists[/yellow]")
            return

        keyword_list = [k.strip() for k in keywords.split(",")] if keywords else []
        topic = Topic(
            name=name,
            description=description or "",
            keywords=json.dumps(keyword_list, ensure_ascii=False) if keyword_list else None,
            created_at=datetime.now(),
        )
        db.create_topic(topic)
        console.print(f"[green]Created topic: {name}[/green]")

    finally:
        db.close()


@topic.command("show")
@click.argument("name")
@click.option("--limit", "-n", default=10, help="Number of recent items to show")
@click.pass_context
def topic_show(ctx, name, limit):
    """Show topic details and recent content.

    Examples:
        babelbyte topic show "AI应用"
    """
    db = get_db()
    try:
        topic = db.get_topic_by_name(name)
        if not topic:
            console.print(f"[red]Topic '{name}' not found[/red]")
            return

        console.print(Panel(f"[bold]{topic.name}[/bold]", title=f"Topic #{topic.id}"))

        if topic.description:
            console.print(f"Description: {topic.description}")

        keywords = topic.get_keywords()
        if keywords:
            console.print(f"Keywords: {', '.join(keywords)}")

        # Get related content
        content = db.get_topic_content(topic.id, limit=limit)
        if content:
            console.print(f"\n[bold]Recent Content ({len(content)} items):[/bold]")
            for item in content:
                console.print(f"  - [{item.importance_score}/10] {item.title[:50]}...")

        # Get snapshots
        snapshots = db.get_topic_snapshots(topic.id, limit=3)
        if snapshots:
            console.print(f"\n[bold]Recent Snapshots:[/bold]")
            for snap in snapshots:
                console.print(f"  [{snap.snapshot_date}] {snap.trend} - {snap.summary[:50]}...")

    finally:
        db.close()


@topic.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def topic_delete(ctx, name, yes):
    """Delete a topic.

    Examples:
        babelbyte topic delete "AI应用" --yes
    """
    db = get_db()
    try:
        topic = db.get_topic_by_name(name)
        if not topic:
            console.print(f"[red]Topic '{name}' not found[/red]")
            return

        if not yes:
            if not click.confirm(f"Delete topic '{name}'?"):
                return

        db.delete_topic(topic.id)
        console.print(f"[red]Deleted topic: {name}[/red]")

    finally:
        db.close()


# ============================================
# Phase 6: Report Commands
# ============================================


@cli.group()
@click.pass_context
def report(ctx):
    """Generate reports (weekly/monthly summaries)."""
    pass


@report.command("week")
@click.option("--weeks-ago", "-w", default=0, help="Generate report for N weeks ago (0 = current)")
@click.pass_context
def report_week(ctx, weeks_ago):
    """Generate a weekly report.

    Examples:
        babelbyte report week
        babelbyte report week --weeks-ago 1
    """
    mock = ctx.obj.get("mock", False)
    db = get_db()
    try:
        from src.analytics.reports import ReportGenerator

        console.print("[bold]Generating weekly report...[/bold]\n")

        generator = ReportGenerator(db=db, use_mock=mock)
        report_data = generator.generate_weekly_report(weeks_ago=weeks_ago)
        report_text = generator.format_report_text(report_data)

        console.print(report_text)

    finally:
        db.close()


@report.command("month")
@click.option("--months-ago", "-m", default=0, help="Generate report for N months ago (0 = current)")
@click.pass_context
def report_month(ctx, months_ago):
    """Generate a monthly report.

    Examples:
        babelbyte report month
        babelbyte report month --months-ago 1
    """
    mock = ctx.obj.get("mock", False)
    db = get_db()
    try:
        from src.analytics.reports import ReportGenerator

        console.print("[bold]Generating monthly report...[/bold]\n")

        generator = ReportGenerator(db=db, use_mock=mock)
        report_data = generator.generate_monthly_report(months_ago=months_ago)
        report_text = generator.format_report_text(report_data)

        console.print(report_text)

    finally:
        db.close()


# ============================================
# Phase 2: Event Commands
# ============================================


@cli.command()
@click.option("--days", "-d", default=7, help="Look back N days")
@click.option("--category", "-c", help="Filter by category")
@click.option("--limit", "-n", default=20, help="Maximum events to show")
@click.pass_context
def events(ctx, days, category, limit):
    """List recent event clusters.

    Examples:
        babelbyte events
        babelbyte events --days 3 --category AI
    """
    db = get_db()
    try:
        clusters = db.get_recent_event_clusters(days=days, category=category, limit=limit)

        if not clusters:
            console.print("[dim]No events found[/dim]")
            return

        console.print(f"[bold]Recent Events ({len(clusters)}):[/bold]\n")

        table = Table()
        table.add_column("ID", style="dim")
        table.add_column("Event")
        table.add_column("Category")
        table.add_column("Articles", justify="right")
        table.add_column("Last Updated", style="dim")

        for cluster in clusters:
            table.add_row(
                str(cluster.id),
                cluster.event_title[:40] + ("..." if len(cluster.event_title) > 40 else ""),
                cluster.category,
                str(cluster.article_count),
                cluster.last_updated_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

    finally:
        db.close()


@cli.command()
@click.argument("event_id", type=int)
@click.pass_context
def event(ctx, event_id):
    """View event cluster details.

    Examples:
        babelbyte event 123
    """
    db = get_db()
    try:
        cluster = db.get_event_cluster(event_id)
        if not cluster:
            console.print(f"[red]Event {event_id} not found[/red]")
            return

        console.print(Panel(f"[bold]{cluster.event_title}[/bold]", title=f"Event #{event_id}"))

        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        table.add_row("Category", cluster.category)
        table.add_row("Article Count", str(cluster.article_count))
        table.add_row("First Seen", cluster.first_seen_at.strftime("%Y-%m-%d %H:%M"))
        table.add_row("Last Updated", cluster.last_updated_at.strftime("%Y-%m-%d %H:%M"))

        console.print(table)

        # Show member articles
        members = db.get_event_members(event_id)
        if members:
            console.print(f"\n[bold]Related Articles ({len(members)}):[/bold]")
            for item in members[:10]:
                console.print(f"  - [{item.importance_score}/10] {item.title[:50]}...")
                if item.summary:
                    console.print(f"    {item.summary[:80]}...")

    finally:
        db.close()


@cli.command("cluster")
@click.option("--limit", "-n", default=50, help="Maximum items to cluster")
@click.pass_context
def cluster_content(ctx, limit):
    """Run event clustering on recent content.

    Examples:
        babelbyte cluster
        babelbyte cluster --limit 100
    """
    mock = ctx.obj.get("mock", False)
    db = get_db()
    try:
        from src.processors.event_stream import cluster_unprocessed_items

        console.print("[bold]Running event clustering...[/bold]")

        clustered = cluster_unprocessed_items(db=db, use_mock=mock, limit=limit)

        console.print(f"[green]Clustered {clustered} items into events[/green]")

    finally:
        db.close()


if __name__ == "__main__":
    cli()
