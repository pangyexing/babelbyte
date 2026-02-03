"""Database management CLI commands."""

import shutil
from datetime import datetime
from pathlib import Path

import click

from config.settings import get_settings
from src.cli.common import console


def get_snapshot_dir() -> Path:
    """Get the snapshot directory, creating it if needed."""
    settings = get_settings()
    snapshot_dir = settings.database.path.parent / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


def get_db_path() -> Path:
    """Get the database file path."""
    return get_settings().database.path


@click.group("db")
def db_group():
    """Database management commands."""
    pass


@db_group.command("snapshot")
@click.argument("name", required=False)
def snapshot(name: str | None):
    """
    Save a database snapshot for later restoration.

    NAME is an optional snapshot name. If not provided, uses timestamp.

    Examples:
        bb db snapshot              # Auto-named: snapshot_20240101_120000.db
        bb db snapshot before-test  # Named: before-test.db
    """
    db_path = get_db_path()

    if not db_path.exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        return

    snapshot_dir = get_snapshot_dir()

    # Generate snapshot filename
    if name:
        snapshot_name = f"{name}.db"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"snapshot_{timestamp}.db"

    snapshot_path = snapshot_dir / snapshot_name

    # Check if snapshot already exists
    if snapshot_path.exists():
        if not click.confirm(f"Snapshot '{snapshot_name}' exists. Overwrite?"):
            console.print("[yellow]Cancelled[/yellow]")
            return

    # Copy database file
    shutil.copy2(db_path, snapshot_path)

    # Get file size
    size_mb = snapshot_path.stat().st_size / (1024 * 1024)

    console.print(f"[green]Snapshot saved:[/green] {snapshot_path.name} ({size_mb:.1f} MB)")


@db_group.command("restore")
@click.argument("name", required=False)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def restore(name: str | None, yes: bool):
    """
    Restore database from a snapshot.

    NAME is the snapshot name (without .db extension).
    If not provided, shows available snapshots to choose from.

    Examples:
        bb db restore before-test   # Restore specific snapshot
        bb db restore               # Interactive selection
    """
    snapshot_dir = get_snapshot_dir()
    db_path = get_db_path()

    # List available snapshots
    snapshots = sorted(snapshot_dir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not snapshots:
        console.print("[yellow]No snapshots found.[/yellow]")
        console.print("Create one with: bb db snapshot [name]")
        return

    # Find the snapshot to restore
    if name:
        # Try exact match first
        snapshot_path = snapshot_dir / f"{name}.db"
        if not snapshot_path.exists():
            # Try without .db suffix (in case user included it)
            snapshot_path = snapshot_dir / name
            if not snapshot_path.exists():
                console.print(f"[red]Snapshot not found: {name}[/red]")
                console.print("\nAvailable snapshots:")
                for s in snapshots[:5]:
                    console.print(f"  - {s.stem}")
                return
    else:
        # Interactive selection
        console.print("[bold]Available snapshots:[/bold]\n")
        for i, s in enumerate(snapshots[:10], 1):
            mtime = datetime.fromtimestamp(s.stat().st_mtime)
            size_mb = s.stat().st_size / (1024 * 1024)
            console.print(f"  {i}. {s.stem} ({size_mb:.1f} MB, {mtime:%Y-%m-%d %H:%M})")

        console.print("")
        choice = click.prompt("Select snapshot number", type=int, default=1)

        if choice < 1 or choice > len(snapshots[:10]):
            console.print("[red]Invalid selection[/red]")
            return

        snapshot_path = snapshots[choice - 1]

    # Confirm restoration
    if not yes:
        console.print("\n[yellow]This will replace the current database with:[/yellow]")
        console.print(f"  {snapshot_path.name}")
        if not click.confirm("Continue?"):
            console.print("[yellow]Cancelled[/yellow]")
            return

    # Create backup of current database
    if db_path.exists():
        backup_path = snapshot_dir / f"_before_restore_{datetime.now():%Y%m%d_%H%M%S}.db"
        shutil.copy2(db_path, backup_path)
        console.print(f"[dim]Current DB backed up to: {backup_path.name}[/dim]")

    # Restore snapshot
    shutil.copy2(snapshot_path, db_path)
    console.print(f"[green]Database restored from:[/green] {snapshot_path.name}")


@db_group.command("snapshots")
def list_snapshots():
    """List all available database snapshots."""
    snapshot_dir = get_snapshot_dir()
    snapshots = sorted(snapshot_dir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not snapshots:
        console.print("[yellow]No snapshots found.[/yellow]")
        console.print("Create one with: bb db snapshot [name]")
        return

    console.print("[bold]Database snapshots:[/bold]\n")

    total_size = 0
    for s in snapshots:
        mtime = datetime.fromtimestamp(s.stat().st_mtime)
        size_mb = s.stat().st_size / (1024 * 1024)
        total_size += size_mb

        # Highlight auto-backups differently
        if s.stem.startswith("_before_restore"):
            style = "dim"
            label = " (auto-backup)"
        else:
            style = ""
            label = ""

        console.print(f"  [{style}]{s.stem}{label} ({size_mb:.1f} MB, {mtime:%Y-%m-%d %H:%M})[/]")

    console.print(f"\n[dim]Total: {len(snapshots)} snapshots, {total_size:.1f} MB[/dim]")
    console.print(f"[dim]Location: {snapshot_dir}[/dim]")


@db_group.command("delete-snapshot")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def delete_snapshot(name: str, yes: bool):
    """Delete a database snapshot."""
    snapshot_dir = get_snapshot_dir()

    snapshot_path = snapshot_dir / f"{name}.db"
    if not snapshot_path.exists():
        snapshot_path = snapshot_dir / name
        if not snapshot_path.exists():
            console.print(f"[red]Snapshot not found: {name}[/red]")
            return

    if not yes:
        if not click.confirm(f"Delete snapshot '{snapshot_path.stem}'?"):
            console.print("[yellow]Cancelled[/yellow]")
            return

    snapshot_path.unlink()
    console.print(f"[green]Deleted:[/green] {snapshot_path.stem}")


def register_commands(cli):
    """Register database commands with the CLI."""
    cli.add_command(db_group)
