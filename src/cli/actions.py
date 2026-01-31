"""Action list CLI commands."""

import click
from rich.table import Table

from src.cli.common import console, get_db
from src.storage.models import ActionStatus


@click.command()
@click.option(
    "--status",
    "-s",
    type=click.Choice(["pending", "done", "dismissed"]),
    default="pending",
    help="Filter by status",
)
@click.option("--priority", "-p", type=click.Choice(["high", "medium", "low"]), help="Filter")
@click.option("--limit", "-n", default=20, help="Maximum results")
@click.pass_context
def actions(ctx, status, priority, limit):
    """List action items.

    Examples:
        babelbyte actions
        babelbyte actions --status pending --priority high
    """
    db = get_db()
    try:
        action_status = ActionStatus(status) if status else None
        items = db.get_action_items(status=action_status, priority=priority, limit=limit)

        if not items:
            console.print("[dim]No action items found[/dim]")
            return

        console.print(f"[bold]Action Items ({len(items)}):[/bold]\n")

        table = Table()
        table.add_column("ID", style="dim")
        table.add_column("Priority")
        table.add_column("Type")
        table.add_column("Description")
        table.add_column("Status")
        table.add_column("Created", style="dim")

        for item in items:
            priority_color = {"high": "red", "medium": "yellow", "low": "dim"}.get(
                item.priority.lower() if item.priority else "", "white"
            )
            table.add_row(
                str(item.id),
                f"[{priority_color}]{item.priority}[/{priority_color}]",
                item.type,
                item.description[:40] + ("..." if len(item.description) > 40 else ""),
                item.status.value,
                item.created_at.strftime("%Y-%m-%d"),
            )

        console.print(table)

    finally:
        db.close()


@click.command("action")
@click.argument("action_id", type=int)
@click.argument("new_status", type=click.Choice(["done", "dismissed"]))
@click.pass_context
def update_action(ctx, action_id, new_status):
    """Update action item status.

    Examples:
        babelbyte action 123 done
        babelbyte action 123 dismissed
    """
    db = get_db()
    try:
        db.update_action_status(action_id, ActionStatus(new_status))
        console.print(f"[green]Action {action_id} marked as {new_status}[/green]")

    finally:
        db.close()


def register_commands(cli):
    """Register action commands with the CLI."""
    cli.add_command(actions)
    cli.add_command(update_action)
