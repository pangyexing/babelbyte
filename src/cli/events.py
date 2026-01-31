"""Event clustering CLI commands."""

import click
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from src.cli.common import console, get_db


@click.command()
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


@click.command()
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

        members = db.get_event_members(event_id)
        if members:
            console.print(f"\n[bold]Related Articles ({len(members)}):[/bold]")
            for item in members[:10]:
                console.print(f"  - [{item.importance_score}/10] {item.title[:50]}...")
                if item.summary:
                    console.print(f"    {item.summary[:80]}...")

    finally:
        db.close()


@click.command("cluster")
@click.option("--limit", "-n", default=50, help="Maximum items to cluster")
@click.option(
    "--parallel/--no-parallel", default=True, help="Use parallel processing (default: enabled)"
)
@click.option("--workers", "-w", default=4, help="Number of parallel workers (default: 4)")
@click.pass_context
def cluster_content(ctx, limit, parallel, workers):
    """Run event clustering on recent content.

    Uses parallel processing by default for faster execution.
    Each item may trigger an AI subprocess call, so parallel processing
    can provide significant speedup (3-4x with 4 workers).

    Examples:
        babelbyte cluster
        babelbyte cluster --limit 100
        babelbyte cluster --limit 100 --workers 8
        babelbyte cluster --no-parallel
    """
    mock = ctx.obj.get("mock", False)
    db = get_db()
    try:
        from src.processors.event_stream import (
            cluster_unprocessed_items,
            cluster_unprocessed_items_parallel,
        )

        mode_str = f"parallel ({workers} workers)" if parallel else "sequential"
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[green]{task.fields[clustered]} clustered[/green]"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[bold]Clustering ({mode_str})...[/bold]", total=None, clustered=0
            )

            def on_progress(current, total, clustered):
                progress.update(task, total=total, completed=current, clustered=clustered)

            if parallel:
                clustered = cluster_unprocessed_items_parallel(
                    db=db,
                    use_mock=mock,
                    limit=limit,
                    max_workers=workers,
                    progress_callback=on_progress,
                )
            else:
                clustered = cluster_unprocessed_items(
                    db=db, use_mock=mock, limit=limit, progress_callback=on_progress
                )

        console.print(f"[green]Clustered {clustered} items into events ({mode_str})[/green]")

    finally:
        db.close()


def register_commands(cli):
    """Register event commands with the CLI."""
    cli.add_command(events)
    cli.add_command(event)
    cli.add_command(cluster_content)
