"""Subscription management CLI commands."""

from datetime import datetime

import click
from rich.table import Table

from src.cli.common import console, get_db
from src.storage.models import SourceType, Subscription, SubscriptionType


@click.command()
@click.argument("source", type=click.Choice(["reddit", "twitter"]))
@click.argument("name")
@click.option("--user", "-u", is_flag=True, help="Subscribe to a user (Reddit only)")
@click.pass_context
def subscribe(ctx, source, name, user):
    """Subscribe to a content source.

    Examples:
        babelbyte subscribe reddit MachineLearning
        babelbyte subscribe reddit --user spez
        babelbyte subscribe twitter elonmusk
    """
    if source == "reddit":
        source_type = SourceType.REDDIT
        if user:
            sub_type = SubscriptionType.REDDIT_USER
            name = name.removeprefix("u/").removeprefix("/u/")
        else:
            sub_type = SubscriptionType.SUBREDDIT
            name = name.removeprefix("r/").removeprefix("/r/")
    else:
        source_type = SourceType.TWITTER
        sub_type = SubscriptionType.TWITTER_USER
        name = name.lstrip("@")

    db = get_db()
    try:
        existing = db.get_subscription_by_name(source_type, sub_type, name)
        if existing:
            if existing.enabled:
                console.print(f"[yellow]Already subscribed to {existing.display_name}[/yellow]")
            else:
                existing.enabled = True
                db.update_subscription(existing)
                console.print(f"[green]Re-enabled subscription: {existing.display_name}[/green]")
            return

        sub = Subscription(
            source_type=source_type,
            subscription_type=sub_type,
            name=name,
            enabled=True,
            created_at=datetime.now(),
        )
        db.add_subscription(sub)
        console.print(f"[green]Subscribed to {sub.display_name}[/green]")

    finally:
        db.close()


@click.command()
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
            console.print(f"[red]Deleted subscription: {sub.display_name}[/red]")
        else:
            sub.enabled = False
            db.update_subscription(sub)
            console.print(f"[yellow]Disabled subscription: {sub.display_name}[/yellow]")

    finally:
        db.close()


@click.command("list")
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

        table = Table(title="Subscriptions")
        table.add_column("ID", style="dim")
        table.add_column("Source", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Status")
        table.add_column("Last Fetched", style="dim")

        for sub in subs:
            status = "[green]Enabled[/green]" if sub.enabled else "[red]Disabled[/red]"
            last_fetched = (
                sub.last_fetched_at.strftime("%Y-%m-%d %H:%M") if sub.last_fetched_at else "Never"
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


def register_commands(cli):
    """Register subscription commands with the CLI."""
    cli.add_command(subscribe)
    cli.add_command(unsubscribe)
    cli.add_command(list_subs)
