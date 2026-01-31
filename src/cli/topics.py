"""Topic management CLI commands."""

import json
from datetime import datetime

import click
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.cli.common import console, get_db
from src.storage.models import Topic, TopicSuggestion


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


@topic.command("discover")
@click.option("--days", "-d", default=14, help="Days to look back")
@click.option("--min-frequency", "-f", default=5, help="Minimum occurrence frequency")
@click.option("--save/--no-save", default=True, help="Save suggestions to database")
@click.pass_context
def topic_discover(ctx, days, min_frequency, save):
    """Discover potential topics from content analysis.

    Analyzes recent content using three methods:
    1. Entity frequency - High-frequency company/product names
    2. Keyword clustering - Co-occurring phrases
    3. Trend detection - 3x week-over-week increases

    Examples:
        bb topic discover --days 14 --min-frequency 5
        bb topic discover --days 7 --no-save
    """
    from src.analytics.topic_discovery import TopicDiscovery

    db = get_db()
    try:
        discovery = TopicDiscovery(db)
        suggestions = discovery.discover_topics(days=days, min_frequency=min_frequency)

        if not suggestions:
            console.print("[dim]No topics discovered. Try lowering --min-frequency or increasing --days.[/dim]")
            return

        table = Table(title=f"Discovered Topics (last {days} days)")
        table.add_column("Name", style="cyan")
        table.add_column("Source", style="dim")
        table.add_column("Freq", justify="right")
        table.add_column("Conf", justify="right")
        table.add_column("Sample Titles")

        for s in suggestions[:20]:
            keywords = s.get_keywords() if hasattr(s, 'get_keywords') else s.keywords
            samples = s.get_sample_titles() if hasattr(s, 'get_sample_titles') else s.sample_titles
            sample_str = samples[0][:40] + "..." if samples else "-"

            table.add_row(
                s.name,
                s.source,
                str(s.frequency),
                f"{s.confidence:.0%}",
                sample_str,
            )

        console.print(table)

        if save:
            saved = 0
            for s in suggestions:
                # Convert to model if needed
                if not isinstance(s, TopicSuggestion):
                    suggestion = TopicSuggestion(
                        name=s.name,
                        keywords=json.dumps(s.keywords, ensure_ascii=False),
                        frequency=s.frequency,
                        confidence=s.confidence,
                        source=s.source,
                        sample_titles=json.dumps(s.sample_titles, ensure_ascii=False),
                        status="pending",
                        suggested_at=datetime.now(),
                    )
                else:
                    suggestion = s
                    if not isinstance(suggestion.keywords, str):
                        suggestion.keywords = json.dumps(suggestion.keywords, ensure_ascii=False)
                    if not isinstance(suggestion.sample_titles, str):
                        suggestion.sample_titles = json.dumps(suggestion.sample_titles or [], ensure_ascii=False)

                db.create_topic_suggestion(suggestion)
                saved += 1

            console.print(f"\n[green]Saved {saved} suggestions. Use 'bb topic review' to accept/reject.[/green]")
        else:
            console.print(f"\n[dim]Found {len(suggestions)} suggestions (not saved). Use --save to persist.[/dim]")

    finally:
        db.close()


@topic.command("suggestions")
@click.option("--status", "-s", type=click.Choice(["pending", "accepted", "rejected", "merged"]))
@click.option("--limit", "-n", default=20, help="Number to show")
@click.pass_context
def topic_suggestions(ctx, status, limit):
    """List topic suggestions.

    Examples:
        bb topic suggestions
        bb topic suggestions --status pending
    """
    db = get_db()
    try:
        suggestions = db.get_topic_suggestions(status=status, limit=limit)

        if not suggestions:
            console.print("[dim]No topic suggestions found.[/dim]")
            return

        table = Table(title="Topic Suggestions")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Source")
        table.add_column("Freq", justify="right")
        table.add_column("Conf", justify="right")
        table.add_column("Status")
        table.add_column("Suggested", style="dim")

        for s in suggestions:
            status_style = {
                "pending": "yellow",
                "accepted": "green",
                "rejected": "red",
                "merged": "blue",
            }.get(s.status, "dim")

            table.add_row(
                str(s.id),
                s.name,
                s.source,
                str(s.frequency),
                f"{s.confidence:.0%}",
                f"[{status_style}]{s.status}[/{status_style}]",
                s.suggested_at.strftime("%Y-%m-%d"),
            )

        console.print(table)

    finally:
        db.close()


@topic.command("review")
@click.pass_context
def topic_review(ctx):
    """Interactively review pending topic suggestions.

    For each suggestion, you can:
    - Accept (a): Create a new topic from the suggestion
    - Reject (r): Mark as rejected
    - Merge (m): Add keywords to existing topic
    - Skip (s): Leave pending for later

    Examples:
        bb topic review
    """
    db = get_db()
    try:
        suggestions = db.get_topic_suggestions(status="pending", limit=50)

        if not suggestions:
            console.print("[dim]No pending suggestions to review.[/dim]")
            return

        console.print(f"\n[bold]Reviewing {len(suggestions)} pending suggestions...[/bold]\n")
        existing_topics = db.list_topics()

        for i, s in enumerate(suggestions):
            console.print(Panel(
                f"[cyan bold]{s.name}[/cyan bold]\n"
                f"Source: {s.source} | Frequency: {s.frequency} | Confidence: {s.confidence:.0%}\n"
                f"Keywords: {', '.join(s.get_keywords())}\n"
                f"Samples: {'; '.join(s.get_sample_titles()[:2]) or 'N/A'}",
                title=f"Suggestion {i+1}/{len(suggestions)}",
            ))

            action = Prompt.ask(
                "[a]ccept / [r]eject / [m]erge / [s]kip / [q]uit",
                choices=["a", "r", "m", "s", "q"],
                default="s",
            )

            if action == "q":
                console.print("[dim]Review stopped.[/dim]")
                break
            elif action == "a":
                # Create topic
                new_topic = Topic(
                    name=s.name,
                    description=f"Auto-discovered via {s.source} analysis",
                    keywords=json.dumps(s.get_keywords(), ensure_ascii=False),
                    created_at=datetime.now(),
                )
                new_topic = db.create_topic(new_topic)
                db.update_topic_suggestion_status(s.id, "accepted")
                console.print(f"[green]Created topic: {s.name}[/green]")
            elif action == "r":
                db.update_topic_suggestion_status(s.id, "rejected")
                console.print(f"[red]Rejected: {s.name}[/red]")
            elif action == "m":
                if not existing_topics:
                    console.print("[yellow]No existing topics to merge with.[/yellow]")
                    continue

                console.print("\nExisting topics:")
                for j, t in enumerate(existing_topics):
                    console.print(f"  {j+1}. {t.name}")

                choice = Prompt.ask("Merge with topic number (or 'c' to cancel)", default="c")
                if choice.isdigit() and 1 <= int(choice) <= len(existing_topics):
                    target_topic = existing_topics[int(choice) - 1]

                    # Add keywords
                    existing_kw = target_topic.get_keywords()
                    new_kw = list(set(existing_kw + s.get_keywords()))
                    target_topic.keywords = json.dumps(new_kw, ensure_ascii=False)
                    db.update_topic(target_topic)

                    db.update_topic_suggestion_status(s.id, "merged", target_topic.id)
                    console.print(f"[blue]Merged with: {target_topic.name}[/blue]")
            # 's' (skip) does nothing

            console.print()

        console.print("[dim]Review complete.[/dim]")

    finally:
        db.close()


def register_commands(cli):
    """Register topic commands with the CLI."""
    cli.add_command(topics)
    cli.add_command(topic)
