"""Report generation CLI commands."""

import click

from src.cli.common import console, get_db


@click.group()
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
@click.option(
    "--months-ago", "-m", default=0, help="Generate report for N months ago (0 = current)"
)
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


def register_commands(cli):
    """Register report commands with the CLI."""
    cli.add_command(report)
