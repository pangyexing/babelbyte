"""Topic management CLI commands."""

import json
from datetime import datetime

import click
from rich.panel import Panel
from rich.table import Table

from src.cli.common import console, get_db
from src.storage.models import Topic


@click.command()
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


@click.group()
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
        babelbyte topic add "AI Apps" --keywords "GPT,ChatGPT,AI"
        babelbyte topic add "OpenAI" --keywords "OpenAI,Sam Altman" -d "OpenAI news"
    """
    db = get_db()
    try:
        existing = db.get_topic_by_name(name)
        if existing:
            console.print(f"[yellow]Topic '{name}' already exists[/yellow]")
            return

        keyword_list = [k.strip() for k in keywords.split(",")] if keywords else []
        new_topic = Topic(
            name=name,
            description=description or "",
            keywords=json.dumps(keyword_list, ensure_ascii=False) if keyword_list else None,
            created_at=datetime.now(),
        )
        db.create_topic(new_topic)
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
        babelbyte topic show "AI Apps"
    """
    db = get_db()
    try:
        topic_data = db.get_topic_by_name(name)
        if not topic_data:
            console.print(f"[red]Topic '{name}' not found[/red]")
            return

        console.print(Panel(f"[bold]{topic_data.name}[/bold]", title=f"Topic #{topic_data.id}"))

        if topic_data.description:
            console.print(f"Description: {topic_data.description}")

        keywords = topic_data.get_keywords()
        if keywords:
            console.print(f"Keywords: {', '.join(keywords)}")

        content = db.get_topic_content(topic_data.id, limit=limit)
        if content:
            console.print(f"\n[bold]Recent Content ({len(content)} items):[/bold]")
            for item in content:
                console.print(f"  - [{item.importance_score}/10] {item.title[:50]}...")

        snapshots = db.get_topic_snapshots(topic_data.id, limit=3)
        if snapshots:
            console.print("\n[bold]Recent Snapshots:[/bold]")
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
        babelbyte topic delete "AI Apps" --yes
    """
    db = get_db()
    try:
        topic_data = db.get_topic_by_name(name)
        if not topic_data:
            console.print(f"[red]Topic '{name}' not found[/red]")
            return

        if not yes:
            if not click.confirm(f"Delete topic '{name}'?"):
                return

        db.delete_topic(topic_data.id)
        console.print(f"[red]Deleted topic: {name}[/red]")

    finally:
        db.close()


def register_commands(cli):
    """Register topic commands with the CLI."""
    cli.add_command(topics)
    cli.add_command(topic)
