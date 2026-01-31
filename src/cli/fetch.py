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


def register_commands(cli):
    """Register fetch commands with the CLI."""
    cli.add_command(fetch)
    cli.add_command(digest)
