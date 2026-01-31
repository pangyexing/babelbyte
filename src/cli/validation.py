"""Validation and optimization CLI commands."""

import time
from datetime import datetime

import click
from rich.panel import Panel
from rich.table import Table

from config.settings import get_settings
from src.cli.common import console, get_db
from src.delivery.email_sender import EmailSender
from src.scheduler.jobs import BabelByteScheduler


@click.command()
@click.pass_context
def config(ctx):
    """Show current configuration."""
    settings = get_settings()

    console.print(Panel("[bold]BabelByte Configuration[/bold]"))

    console.print("\n[bold]Database:[/bold]")
    console.print(f"  Path: {settings.database.path}")

    console.print("\n[bold]Twitter:[/bold]")
    if settings.twitter.is_configured:
        console.print("  [green]Configured[/green]")
    else:
        console.print("  [red]Not configured (set TWITTER_BEARER_TOKEN)[/red]")

    console.print("\n[bold]Email:[/bold]")
    if settings.email.is_configured:
        console.print("  [green]Configured[/green]")
        console.print(f"  Host: {settings.email.host}:{settings.email.port}")
        console.print(f"  From: {settings.email.from_addr}")
        console.print(f"  To: {settings.email.to_addr}")
    else:
        console.print("  [red]Not configured (set SMTP settings in .env)[/red]")

    console.print("\n[bold]Scheduler:[/bold]")
    console.print(f"  Fetch interval: every {settings.scheduler.fetch_interval_hours} hours")
    console.print(f"  Digest time: {settings.scheduler.digest_send_time}")

    console.print("\n[bold]AI Provider:[/bold]")
    console.print(f"  Current: {settings.ai.provider}")

    console.print("\n[bold]Claude CLI:[/bold]")
    console.print(f"  Path: {settings.claude.cli_path}")

    console.print("\n[bold]Codex CLI:[/bold]")
    console.print(f"  Path: {settings.codex.cli_path}")


@click.command("test-email")
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
        console.print(f"[green]{result.message}[/green]")
    else:
        console.print(f"[red]{result.message}[/red]")


@click.command()
@click.pass_context
def run(ctx):
    """Run the scheduler (foreground process)."""
    mock = ctx.obj.get("mock", False)

    console.print("[bold]Starting BabelByte scheduler...[/bold]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    scheduler = BabelByteScheduler(use_mock=mock)

    try:
        scheduler.start()

        jobs = scheduler.get_jobs()
        table = Table(title="Scheduled Jobs")
        table.add_column("Job")
        table.add_column("Next Run")

        for job in jobs:
            next_run = job["next_run"].strftime("%Y-%m-%d %H:%M") if job["next_run"] else "N/A"
            table.add_row(job["name"], next_run)

        console.print(table)
        console.print("\n[green]Scheduler running...[/green]")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping scheduler...[/yellow]")
        scheduler.stop()
        console.print("[green]Scheduler stopped[/green]")


@click.command()
@click.option("--fix", is_flag=True, help="Attempt to fix discovered issues")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed results")
@click.pass_context
def validate(ctx, fix, verbose):
    """Validate data integrity.

    Checks for orphan content, duplicates, invalid scores,
    empty clusters, and other data integrity issues.

    Examples:
        bb validate
        bb validate --verbose
        bb validate --fix
    """
    from src.validation import CheckStatus, DataValidator

    db = get_db()
    try:
        validator = DataValidator(db)

        console.print("[bold]Running data validation checks...[/bold]\n")

        result = validator.run_all_checks(verbose=verbose)

        table = Table(title="Validation Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Message")
        table.add_column("Count", justify="right")

        for check in result.checks:
            if check.status == CheckStatus.PASS:
                status = "[green]PASS[/green]"
            elif check.status == CheckStatus.FAIL:
                status = "[red]FAIL[/red]"
            elif check.status == CheckStatus.WARN:
                status = "[yellow]WARN[/yellow]"
            else:
                status = "[dim]SKIP[/dim]"

            table.add_row(
                check.name,
                status,
                check.message[:50] + ("..." if len(check.message) > 50 else ""),
                str(check.count) if check.count > 0 else "-",
            )

            if verbose and check.details:
                for detail in check.details[:5]:
                    console.print(f"    [dim]{detail}[/dim]")

        console.print(table)
        console.print(f"\n[bold]Summary:[/bold] {result.summary}")

        if result.total_issues > 0:
            console.print(f"[red]Total issues: {result.total_issues}[/red]")

            if fix:
                console.print("\n[bold]Attempting to fix issues...[/bold]")
                fixed = validator.fix_issues(result)
                console.print(f"[green]Fixed {fixed} issues[/green]")
        else:
            console.print("[green]No issues found[/green]")

        if verbose:
            console.print("\n[bold]Database Statistics:[/bold]")
            stats = validator.get_stats()
            if "content" in stats:
                console.print(
                    f"  Content: {stats['content']['total']} total, "
                    f"{stats['content']['processed']} processed, "
                    f"{stats['content']['delivered']} delivered"
                )
            if "clusters" in stats:
                console.print(
                    f"  Clusters: {stats['clusters']['total']} total, "
                    f"{stats['clusters']['total_members']} members"
                )
            if "cache" in stats:
                console.print(
                    f"  Cache: {stats['cache']['valid']} valid, "
                    f"{stats['cache']['expired']} expired"
                )

    finally:
        db.close()


@click.command("token-stats")
@click.option("--reset", is_flag=True, help="Reset token tracking (clear all records)")
@click.option("--days", type=int, default=None, help="Show stats for last N days only")
@click.pass_context
def token_stats(ctx, reset, days):
    """Show token usage statistics.

    Displays token consumption by call type, cache hit rates,
    and estimated costs from persistent database records.

    Examples:
        bb token-stats
        bb token-stats --days 7
        bb token-stats --reset
    """
    from datetime import timedelta

    from src.analytics.token_tracker import get_tracker

    tracker = get_tracker()

    # Calculate since date if days is specified
    since = None
    if days:
        since = datetime.now() - timedelta(days=days)

    # Get stats from database
    stats = tracker.get_persistent_stats(since)

    console.print(Panel("[bold]Token Usage Statistics[/bold]"))

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Calls", str(stats.total_calls))
    table.add_row("Actual AI Calls", str(stats.actual_ai_calls))
    table.add_row("Cache Hits", str(stats.cache_hits))
    table.add_row("Cache Hit Rate", f"{stats.cache_hit_rate:.1f}%")
    table.add_row("", "")
    table.add_row("Input Tokens", f"{stats.input_tokens:,}")
    table.add_row("Output Tokens", f"{stats.output_tokens:,}")
    table.add_row("Total Tokens", f"{stats.total_tokens:,}")
    table.add_row("", "")

    # Calculate costs based on persistent stats
    def estimate_cost(model: str) -> float:
        pricing = {
            "haiku": {"input": 0.25, "output": 1.25},
            "sonnet": {"input": 3.00, "output": 15.00},
        }
        rates = pricing.get(model, pricing["haiku"])
        input_cost = (stats.input_tokens / 1_000_000) * rates["input"]
        output_cost = (stats.output_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost

    table.add_row("Est. Cost (Haiku)", f"${estimate_cost('haiku'):.4f}")
    table.add_row("Est. Cost (Sonnet)", f"${estimate_cost('sonnet'):.4f}")

    console.print(table)

    if stats.calls_by_type:
        console.print("\n[bold]By Call Type:[/bold]")
        type_table = Table()
        type_table.add_column("Type")
        type_table.add_column("Total", justify="right")
        type_table.add_column("Cached", justify="right")
        type_table.add_column("Tokens", justify="right")

        for call_type, data in sorted(stats.calls_by_type.items()):
            type_table.add_row(
                call_type,
                str(data["total"]),
                str(data["cached"]),
                f"{data['tokens']:,}",
            )

        console.print(type_table)

    if stats.errors > 0:
        console.print(f"\n[red]Errors: {stats.errors}[/red]")

    if days:
        console.print(f"\n[dim]Showing stats for last {days} days[/dim]")
    else:
        console.print("\n[dim]Showing all-time stats[/dim]")

    if reset:
        deleted = tracker.clear_persistent_stats()
        tracker.reset()
        console.print(f"[yellow]Token tracking reset ({deleted} records deleted)[/yellow]")


@click.command("cache-stats")
@click.option("--cleanup", is_flag=True, help="Clean up expired cache entries")
@click.pass_context
def cache_stats(ctx, cleanup):
    """Show AI cache statistics and optionally clean up.

    Examples:
        bb cache-stats
        bb cache-stats --cleanup
    """
    from src.optimization import CacheOptimizer

    db = get_db()
    try:
        optimizer = CacheOptimizer(db)

        metrics = optimizer.get_cache_metrics()

        console.print(Panel("[bold]AI Cache Status[/bold]"))

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Entries", str(metrics.total_entries))
        table.add_row("Valid Entries", str(metrics.valid_entries))
        table.add_row("Expired Entries", str(metrics.expired_entries))
        table.add_row("Utilization", f"{metrics.utilization_rate:.1f}%")
        table.add_row("Est. Size", f"{metrics.estimated_size_kb:.2f} KB")

        console.print(table)

        if metrics.oldest_entry:
            console.print(f"\n[dim]Oldest entry: {metrics.oldest_entry}[/dim]")
        if metrics.newest_entry:
            console.print(f"[dim]Newest entry: {metrics.newest_entry}[/dim]")

        efficiency = optimizer.analyze_cache_efficiency()
        if efficiency.recommendations:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in efficiency.recommendations:
                console.print(f"  - {rec}")

        if cleanup:
            console.print("\n[bold]Running cleanup...[/bold]")
            result = optimizer.cleanup_and_optimize()
            console.print(f"[green]Removed {result['expired_removed']} expired entries[/green]")
            console.print(f"[green]Freed ~{result['space_freed_kb']:.2f} KB[/green]")

    finally:
        db.close()


def register_commands(cli):
    """Register validation commands with the CLI."""
    cli.add_command(config)
    cli.add_command(test_email)
    cli.add_command(run)
    cli.add_command(validate)
    cli.add_command(token_stats)
    cli.add_command(cache_stats)
