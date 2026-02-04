"""Fetch and digest CLI commands."""

import click
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from config.settings import get_settings
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

            settings = get_settings()
            processed = generator.process_unprocessed_items(
                limit=settings.ai.process_limit, progress_callback=update_progress
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
@click.option("--min-importance", default=7, help="Minimum importance score to include (default: 7)")
@click.pass_context
def daily(ctx, dry_run, skip_fetch, min_importance):
    """Run the complete daily pipeline in one command.

    Executes: fetch → embeddings → process → cluster → digest

    Perfect for manual daily runs when you can't keep a daemon running.
    Topic discovery is available separately via `bb topic discover`.

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
            console.print("\n[bold cyan]Step 1/5: Fetching content...[/bold cyan]")
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
            console.print("\n[bold cyan]Step 1/5: Fetch skipped[/bold cyan]")

        # Step 2: Compute embeddings (free, local) - before AI to enable dedup
        console.print("\n[bold cyan]Step 2/5: Computing embeddings...[/bold cyan]")
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
        console.print("\n[bold cyan]Step 3/5: Processing with AI...[/bold cyan]")
        settings = get_settings()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Processing...", total=None)

            def update_ai_progress(stage: str, current: int, total: int):
                progress.update(task, description=f"{stage}: {current}/{total}", completed=current, total=total)

            processed = runner.process_content(
                limit=settings.ai.process_limit,
                progress_callback=update_ai_progress,
            )

        console.print(f"  [green]✓[/green] Processed {processed} items")

        # Step 4: Run clustering
        console.print("\n[bold cyan]Step 4/5: Clustering events...[/bold cyan]")
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

        # Step 5: Generate and send digest
        console.print("\n[bold cyan]Step 5/5: Generating digest...[/bold cyan]")

        db = get_db()
        try:
            generator = DigestGenerator(db=db, use_mock=mock)
            digest_result = generator.generate_digest(
                min_importance=min_importance,
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


@click.command("prefilter-stats")
@click.option("--limit", default=100, help="Number of unprocessed items to test")
@click.pass_context
def prefilter_stats(ctx, limit: int):
    """
    Test Phase 1 pre-filtering without AI processing.

    Shows statistics on how many items would be:
    - Skipped (spam, short, etc.)
    - Dedup matched (similar to existing processed items)
    - Sent to AI (need actual processing)

    Examples:
        bb prefilter-stats              # Test 100 items (default)
        bb prefilter-stats --limit 200  # Test 200 items
    """
    from src.optimization.dedup_optimizer import find_similar_processed_item
    from src.processors.rule_classifier import should_skip_ai_processing

    db = get_db()

    try:
        items = db.get_unprocessed_items(limit=limit)
        console.print(f"Testing {len(items)} unprocessed items...\n")

        if not items:
            console.print("[yellow]No unprocessed items found.[/yellow]")
            return

        stats = {
            "total": len(items),
            "skipped": 0,
            "dedup": 0,
            "need_ai": 0,
            "skip_reasons": {},
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing...", total=len(items))

            for item in items:
                # 1. Check skip
                should_skip, reason = should_skip_ai_processing(item)
                if should_skip:
                    stats["skipped"] += 1
                    stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                    progress.update(task, advance=1)
                    continue

                # 2. Find similar (embedding-based dedup)
                similar = find_similar_processed_item(item, db)
                if similar:
                    stats["dedup"] += 1
                    progress.update(task, advance=1)
                    continue

                # 3. Need AI
                stats["need_ai"] += 1
                progress.update(task, advance=1)

        # Output results
        total = stats["total"]
        console.print("[bold]=== Phase 1 Pre-filtering Statistics ===[/bold]")
        console.print(f"Total items tested: {total}\n")

        skip_pct = 100 * stats["skipped"] / total
        dedup_pct = 100 * stats["dedup"] / total
        ai_pct = 100 * stats["need_ai"] / total

        console.print(f"[red]Skipped:[/red]   {stats['skipped']:4d} ({skip_pct:.1f}%)")
        console.print(f"[cyan]Dedup:[/cyan]     {stats['dedup']:4d} ({dedup_pct:.1f}%)")
        console.print(f"[green]Need AI:[/green]   {stats['need_ai']:4d} ({ai_pct:.1f}%)")
        console.print("")

        prefilter_total = stats["skipped"] + stats["dedup"]
        console.print(f"[bold]Pre-filter rate: {100*prefilter_total/total:.1f}%[/bold]")

        if stats["skip_reasons"]:
            console.print("\n[bold]Skip reasons:[/bold]")
            for reason, count in sorted(stats["skip_reasons"].items(), key=lambda x: -x[1]):
                console.print(f"  {reason}: {count}")

    finally:
        db.close()


def register_commands(cli):
    """Register fetch commands with the CLI."""
    cli.add_command(fetch)
    cli.add_command(digest)
    cli.add_command(daily)
    cli.add_command(prefilter_stats)
