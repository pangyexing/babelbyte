"""Fetch and digest CLI commands."""

import click
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from src.cli.common import console, get_db
from src.delivery.email_sender import EmailSender
from src.processors.digest_processor import DigestGenerator, create_digest_preview
from src.scheduler.jobs import JobRunner


@click.command()
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


@click.command()
@click.option("--dry-run", is_flag=True, help="Preview digest without sending")
@click.option("--min-importance", "-m", default=5, help="Minimum importance score (1-10)")
@click.option("--max-items", "-n", default=30, help="Maximum items in digest")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["claude", "codex", "auto"]),
    default=None,
    help="AI provider to use",
)
@click.option("--no-cluster", is_flag=True, help="Skip automatic event clustering")
@click.option(
    "--parallel/--no-parallel", default=True, help="Use parallel clustering (default: enabled)"
)
@click.option("--workers", "-w", default=4, help="Number of parallel clustering workers")
@click.pass_context
def digest(ctx, dry_run, min_importance, max_items, provider, no_cluster, parallel, workers):
    """Generate and send the daily digest."""
    mock = ctx.obj.get("mock", False)

    db = get_db()
    try:
        generator = DigestGenerator(db=db, provider=provider, use_mock=mock)

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

        if no_cluster:
            console.print("[bold]Generating digest (clustering disabled)...[/bold]")
            digest_result = generator.generate_digest(
                min_importance=min_importance,
                max_items=max_items,
                run_clustering=False,
            )
        else:
            mode_str = f"parallel, {workers} workers" if parallel else "sequential"
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[green]{task.fields[clustered]} clustered[/green]"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"[bold]Clustering events ({mode_str})...[/bold]", total=None, clustered=0
                )

                def on_cluster_progress(current, total, clustered):
                    progress.update(task, total=total, completed=current, clustered=clustered)

                digest_result = generator.generate_digest(
                    min_importance=min_importance,
                    max_items=max_items,
                    run_clustering=True,
                    clustering_progress_callback=on_cluster_progress,
                    parallel_clustering=parallel,
                    clustering_workers=workers,
                )

        if not digest_result.items and not digest_result.events:
            console.print("[yellow]No items to include in digest[/yellow]")
            return

        if digest_result.events:
            console.print(
                f"[dim]Found {len(digest_result.events)} events "
                f"and {len(digest_result.items)} individual items[/dim]"
            )

        preview = create_digest_preview(digest_result)
        console.print(preview)

        if dry_run:
            console.print("\n[yellow]Dry run - email not sent[/yellow]")
            return

        sender = EmailSender()
        if not sender.is_configured():
            console.print("[red]Email not configured. Please set SMTP settings in .env[/red]")
            return

        console.print("\n[bold]Sending email...[/bold]")
        result = sender.send_digest(digest_result)

        if result.success:
            generator.mark_digest_delivered(digest_result)
            console.print(f"[green]{result.message}[/green]")
        else:
            console.print(f"[red]{result.message}[/red]")

    finally:
        db.close()


@click.command()
@click.option("--dry-run", is_flag=True, help="Preview digest without sending email")
@click.option("--skip-fetch", is_flag=True, help="Skip fetching new content")
@click.pass_context
def daily(ctx, dry_run, skip_fetch):
    """Run the complete daily pipeline in one command.

    Executes: fetch → embeddings → process → topics → cluster → digest

    Perfect for manual daily runs when you can't keep a daemon running.

    Examples:
        bb daily              # Full pipeline, send email
        bb daily --dry-run    # Full pipeline, preview only
        bb daily --skip-fetch # Skip fetch, process existing content
    """
    mock = ctx.obj.get("mock", False)
    runner = JobRunner(use_mock=mock)

    try:
        # Step 1: Fetch
        if not skip_fetch:
            console.print("\n[bold cyan]Step 1/6: Fetching content...[/bold cyan]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Fetching...", total=None)

                def on_fetch_progress(name, current, total, new_items):
                    progress.update(task, total=total, completed=current, description=name)

                fetch_stats = runner.fetch_all_content(progress_callback=on_fetch_progress)

            console.print(
                f"  [green]✓[/green] Fetched {fetch_stats['fetched']}/{fetch_stats['total']} sources, "
                f"{fetch_stats['new_items']} new items"
            )
        else:
            console.print("\n[bold cyan]Step 1/6: Fetch skipped[/bold cyan]")

        # Step 2: Compute embeddings (free, local) - before AI to enable dedup
        console.print("\n[bold cyan]Step 2/6: Computing embeddings...[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Computing embeddings...", total=None)
            emb_stats = runner.compute_embeddings()

        if emb_stats.get("skipped"):
            console.print("  [dim]Embeddings disabled, skipped[/dim]")
        elif emb_stats.get("error"):
            console.print(f"  [yellow]Warning: {emb_stats['error']}[/yellow]")
        else:
            console.print(f"  [green]✓[/green] Computed {emb_stats['computed']} embeddings")

        # Step 3: Process content with AI (can leverage embeddings for dedup)
        console.print("\n[bold cyan]Step 3/6: Processing with AI...[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Processing...", total=None)
            processed = runner.process_content(limit=100)

        console.print(f"  [green]✓[/green] Processed {processed} items")

        # Step 4: Discover topics (free, statistics only)
        console.print("\n[bold cyan]Step 4/6: Discovering topics...[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing topics...", total=None)
            topic_stats = runner.discover_topics()

        if topic_stats.get("error"):
            console.print(f"  [yellow]Warning: {topic_stats['error']}[/yellow]")
        else:
            console.print(
                f"  [green]✓[/green] Discovered {topic_stats['discovered']} topics, "
                f"saved {topic_stats['saved']}"
            )

        # Step 5: Run clustering
        console.print("\n[bold cyan]Step 5/6: Clustering events...[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Clustering...", total=None)
            cluster_stats = runner.run_clustering()

        if cluster_stats.get("error"):
            console.print(f"  [yellow]Warning: {cluster_stats['error']}[/yellow]")
        else:
            console.print(f"  [green]✓[/green] Clustered {cluster_stats['clustered']} items")

        # Step 6: Generate and send digest
        console.print("\n[bold cyan]Step 6/6: Generating digest...[/bold cyan]")

        db = get_db()
        try:
            generator = DigestGenerator(db=db, use_mock=mock)
            digest_result = generator.generate_digest(
                min_importance=5,
                max_items=30,
                run_clustering=False,  # Already clustered above
            )

            if not digest_result.items and not digest_result.events:
                console.print("  [yellow]No items to include in digest[/yellow]")
                return

            item_count = len(digest_result.items)
            event_count = len(digest_result.events) if digest_result.events else 0
            console.print(f"  [green]✓[/green] Generated: {event_count} events, {item_count} items")

            # Show preview
            preview = create_digest_preview(digest_result)
            console.print(preview)

            if dry_run:
                console.print("\n[yellow]Dry run - email not sent[/yellow]")
                return

            # Send email
            sender = EmailSender()
            if not sender.is_configured():
                console.print("[red]Email not configured. Set SMTP settings in .env[/red]")
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

    finally:
        runner.close_db()

    console.print("\n[bold green]Daily pipeline completed![/bold green]")


def register_commands(cli):
    """Register fetch commands with the CLI."""
    cli.add_command(fetch)
    cli.add_command(digest)
    cli.add_command(daily)
