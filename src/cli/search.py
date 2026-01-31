"""Knowledge base search CLI commands."""

from datetime import datetime, timedelta

import click
from rich.panel import Panel
from rich.table import Table

from src.cli.common import console, get_db
from src.storage.models import ItemState


@click.command()
@click.argument("query")
@click.option("--category", "-c", help="Filter by category")
@click.option("--from", "from_date", help="Filter from date (YYYY-MM-DD)")
@click.option("--to", "to_date", help="Filter to date (YYYY-MM-DD)")
@click.option("--min-importance", "-m", type=int, help="Minimum importance score")
@click.option("--limit", "-n", default=20, help="Maximum results")
@click.pass_context
def search(ctx, query, category, from_date, to_date, min_importance, limit):
    """Search content in the knowledge base.

    Examples:
        babelbyte search "GPT"
        babelbyte search "AI" --category AI --from 2024-01-01
        babelbyte search "融资" --min-importance 7
    """
    db = get_db()
    try:
        results = db.search_content(
            query=query,
            category=category,
            from_date=from_date,
            to_date=to_date,
            min_importance=min_importance,
            limit=limit,
        )

        if not results:
            console.print(f"[dim]No results found for '{query}'[/dim]")
            return

        console.print(f"[bold]Found {len(results)} results for '{query}':[/bold]\n")

        for item in results:
            importance_color = (
                "red"
                if item.importance_score and item.importance_score >= 8
                else "yellow" if item.importance_score and item.importance_score >= 5 else "dim"
            )
            console.print(
                f"[{importance_color}][{item.importance_score or '-'}/10][/{importance_color}] ",
                end="",
            )
            console.print(f"[cyan][{item.category or 'Uncategorized'}][/cyan] ", end="")
            console.print(f"[bold]{item.title[:60]}{'...' if len(item.title) > 60 else ''}[/bold]")
            if item.summary:
                console.print(f"    {item.summary[:100]}{'...' if len(item.summary) > 100 else ''}")
            date_str = item.published_at.strftime("%Y-%m-%d")
            console.print(f"    [dim]ID: {item.id} | {date_str} | {item.url[:50]}...[/dim]\n")

    finally:
        db.close()


@click.command()
@click.option(
    "--date", "-d", default="today", help="Date to browse (YYYY-MM-DD or 'today', 'yesterday')"
)
@click.option("--category", "-c", help="Filter by category")
@click.option("--limit", "-n", default=50, help="Maximum results")
@click.pass_context
def browse(ctx, date, category, limit):
    """Browse content by date.

    Examples:
        babelbyte browse
        babelbyte browse --date yesterday
        babelbyte browse --date 2024-01-15 --category AI
    """
    if date == "today":
        date = datetime.now().strftime("%Y-%m-%d")
    elif date == "yesterday":
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    db = get_db()
    try:
        results = db.browse_by_date(date=date, category=category, limit=limit)

        if not results:
            console.print(f"[dim]No content found for {date}[/dim]")
            return

        console.print(f"[bold]Content from {date}:[/bold]\n")

        by_category: dict[str, list] = {}
        for item in results:
            cat = item.category or "Uncategorized"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)

        for cat, items in sorted(by_category.items()):
            console.print(f"[cyan][{cat}] ({len(items)} items)[/cyan]")
            for item in items:
                state_icon = {
                    "unread": "",
                    "read": "",
                    "saved": "",
                    "flagged": "",
                    "archived": "",
                }.get(item.state.value, "")
                title = item.title[:50] + ("..." if len(item.title) > 50 else "")
                score = item.importance_score or "-"
                console.print(f"  {state_icon} [{score}/10] {title} [dim](#{item.id})[/dim]")
            console.print()

    finally:
        db.close()


@click.command()
@click.argument("item_id", type=int)
@click.option("--open", "-o", "open_url", is_flag=True, help="Open URL in browser")
@click.pass_context
def item(ctx, item_id, open_url):
    """View a content item details.

    Examples:
        babelbyte item 123
        babelbyte item 123 --open
    """
    db = get_db()
    try:
        content = db.get_content_item(item_id)

        if not content:
            console.print(f"[red]Item {item_id} not found[/red]")
            return

        console.print(Panel(f"[bold]{content.title}[/bold]", title=f"Item #{item_id}"))

        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        table.add_row("Category", content.category or "Uncategorized")
        table.add_row("Importance", f"{content.importance_score or '-'}/10")
        table.add_row("State", content.state.value)
        table.add_row("Author", content.author)
        table.add_row("Published", content.published_at.strftime("%Y-%m-%d %H:%M"))
        table.add_row("URL", content.url)

        console.print(table)

        if content.summary:
            console.print(f"\n[bold]Summary:[/bold]\n{content.summary}")

        if content.one_liner:
            console.print(f"\n[bold]One-liner:[/bold]\n{content.one_liner}")

        enhanced = content.get_enhanced_data()
        if enhanced:
            if enhanced.key_points:
                console.print("\n[bold]Key Points:[/bold]")
                for kp in enhanced.key_points:
                    console.print(f"  - [{kp.type}] {kp.value}: {kp.impact}")

            if enhanced.impact_assessment:
                console.print("\n[bold]Impact Assessment:[/bold]")
                if enhanced.impact_assessment.short_term:
                    console.print(f"  Short-term: {enhanced.impact_assessment.short_term}")
                if enhanced.impact_assessment.long_term:
                    console.print(f"  Long-term: {enhanced.impact_assessment.long_term}")

            if enhanced.actionable_items:
                console.print("\n[bold]Action Items:[/bold]")
                for action in enhanced.actionable_items:
                    console.print(f"  - [{action.priority}] {action.type}: {action.description}")

        if open_url and content.url:
            import webbrowser

            webbrowser.open(content.url)
            console.print("\n[green]Opened URL in browser[/green]")

    finally:
        db.close()


@click.command()
@click.argument("item_id", type=int)
@click.argument("state", type=click.Choice(["unread", "read", "saved", "archived", "flagged"]))
@click.pass_context
def mark(ctx, item_id, state):
    """Mark a content item with a state.

    Examples:
        babelbyte mark 123 saved
        babelbyte mark 123 read
        babelbyte mark 123 flagged
    """
    db = get_db()
    try:
        content = db.get_content_item(item_id)
        if not content:
            console.print(f"[red]Item {item_id} not found[/red]")
            return

        new_state = ItemState(state)
        db.update_item_state(item_id, new_state)
        console.print(f"[green]Marked item {item_id} as {state}[/green]")

    finally:
        db.close()


@click.command()
@click.pass_context
def stats(ctx):
    """Show knowledge base statistics."""
    db = get_db()
    try:
        category_stats = db.get_category_stats()
        state_stats = db.get_state_stats()

        console.print(Panel("[bold]Knowledge Base Statistics[/bold]"))

        console.print("\n[bold]By Category:[/bold]")
        table = Table()
        table.add_column("Category")
        table.add_column("Count", justify="right")
        for cat, count in category_stats.items():
            table.add_row(cat, str(count))
        console.print(table)

        console.print("\n[bold]By State:[/bold]")
        table = Table()
        table.add_column("State")
        table.add_column("Count", justify="right")
        for state, count in state_stats.items():
            table.add_row(state, str(count))
        console.print(table)

        total = sum(category_stats.values())
        console.print(f"\n[bold]Total items:[/bold] {total}")

    finally:
        db.close()


@click.command("rebuild-index")
@click.pass_context
def rebuild_index(ctx):
    """Rebuild the full-text search index."""
    db = get_db()
    try:
        console.print("[bold]Rebuilding FTS index...[/bold]")
        count = db.rebuild_fts_index()
        console.print(f"[green]Indexed {count} items[/green]")

    finally:
        db.close()


def register_commands(cli):
    """Register search commands with the CLI."""
    cli.add_command(search)
    cli.add_command(browse)
    cli.add_command(item)
    cli.add_command(mark)
    cli.add_command(stats)
    cli.add_command(rebuild_index)
