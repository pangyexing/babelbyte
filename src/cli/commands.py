"""CLI commands for BabelByte."""

import sys
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import get_settings, reload_settings
from src.delivery.email_sender import EmailSender
from src.processors.digest_processor import DigestGenerator, create_digest_preview
from src.scheduler.jobs import BabelByteScheduler, JobRunner
from src.storage.database import SyncDatabase
from src.storage.models import SourceType, Subscription, SubscriptionType

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

    console.print("[bold]Fetching content...[/bold]")

    runner = JobRunner(use_mock=mock)
    try:
        stats = runner.fetch_all_content()

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

        # First process unprocessed items
        console.print("[bold]Processing content with AI...[/bold]")
        processed = generator.process_unprocessed_items(limit=100)
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


if __name__ == "__main__":
    cli()
