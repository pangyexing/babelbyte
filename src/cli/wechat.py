"""CLI commands for WeChat Official Account (公众号) publishing."""

from pathlib import Path

import click

from src.cli.common import console, get_db


@click.group()
def wechat():
    """WeChat Official Account (公众号) commands."""
    pass


@wechat.command("publish")
@click.option("--publish", "do_publish", is_flag=True, help="Publish after creating draft (requires API permission)")
@click.option("--dry-run", is_flag=True, help="Preview locally without calling API")
@click.option("--min-importance", default=5, help="Minimum importance score (default: 5)")
@click.option("--max-items", default=30, help="Maximum items in digest (default: 30)")
@click.option("--no-cluster", is_flag=True, help="Skip automatic event clustering")
@click.pass_context
def publish(ctx, do_publish, dry_run, min_importance, max_items, no_cluster):
    """Create WeChat Official Account draft (publish manually from MP backend).

    Examples:
        bb wechat publish              # Create draft
        bb wechat publish --publish    # Create draft + auto-publish
        bb wechat publish --dry-run    # Local preview, no API calls
    """
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    from config.settings import get_settings
    from src.delivery.wechat_mp import WechatMPPublisher
    from src.processors.digest_processor import DigestGenerator, create_digest_preview

    settings = get_settings()

    if not dry_run and not settings.wechat_mp.is_configured:
        console.print(
            "[red]WeChat MP not configured. "
            "Set WECHAT_MP_APPID and WECHAT_MP_APPSECRET in .env[/red]"
        )
        return

    mock = ctx.obj.get("mock", False)
    db = get_db()

    try:
        # Generate digest
        console.print("[bold]Generating digest...[/bold]")
        generator = DigestGenerator(db=db, use_mock=mock)

        if no_cluster:
            digest_result = generator.generate_digest(
                min_importance=min_importance,
                max_items=max_items,
                run_clustering=False,
                include_delivered=True,
            )
        else:
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
                    "[bold]Clustering events...[/bold]", total=None, clustered=0
                )

                def on_cluster_progress(current, total, clustered):
                    progress.update(
                        task, total=total, completed=current, clustered=clustered
                    )

                digest_result = generator.generate_digest(
                    min_importance=min_importance,
                    max_items=max_items,
                    run_clustering=True,
                    clustering_progress_callback=on_cluster_progress,
                    include_delivered=True,
                )

        if not digest_result.items and not digest_result.events:
            console.print("[yellow]No items to include in digest[/yellow]")
            return

        event_count = len(digest_result.events) if digest_result.events else 0
        item_count = len(digest_result.items)
        console.print(
            f"  [green]✓[/green] Generated: {event_count} events, {item_count} items"
        )

        # Show preview
        preview = create_digest_preview(digest_result)
        console.print(preview)

        if dry_run:
            # Render HTML locally for preview (with same size limits as publish)
            publisher = WechatMPPublisher.__new__(WechatMPPublisher)
            from jinja2 import Environment, FileSystemLoader
            from config.settings import PROJECT_ROOT
            publisher.jinja_env = Environment(
                loader=FileSystemLoader(PROJECT_ROOT / "templates"),
                autoescape=False,
            )
            html = publisher._render_for_wechat(digest_result)
            title = publisher._generate_title(digest_result)

            title_bytes = len(title.encode("utf-8"))
            console.print(f"\n[bold]Title:[/bold] {title}")
            console.print(
                f"[bold]Title size:[/bold] {len(title)} chars, {title_bytes} bytes"
            )
            console.print(
                f"[bold]Content size:[/bold] {len(html)} chars, "
                f"{len(html.encode('utf-8'))} bytes"
            )
            console.print("[yellow]Dry run - no API calls made[/yellow]")
            return

        # Publish
        publisher = WechatMPPublisher()
        draft_only = not do_publish
        mode = "publish" if do_publish else "draft"
        console.print(f"\n[bold]Creating {mode}...[/bold]")

        result = publisher.publish_digest(digest_result, draft_only=draft_only)

        if result.success:
            console.print(f"[green]✓ {result.message}[/green]")
            if result.draft_media_id:
                console.print(f"  Draft media_id: {result.draft_media_id}")
            if result.publish_id:
                console.print(f"  Publish ID: {result.publish_id}")
        else:
            console.print(f"[red]✗ {result.message}[/red]")

    finally:
        db.close()


@wechat.command("test")
def test():
    """Test WeChat MP API connectivity.

    Verifies that APPID and APPSECRET are valid by fetching an access_token.
    """
    from config.settings import get_settings
    from src.delivery.wechat_mp import WechatMPPublisher

    settings = get_settings()
    if not settings.wechat_mp.is_configured:
        console.print(
            "[red]WeChat MP not configured. "
            "Set WECHAT_MP_APPID and WECHAT_MP_APPSECRET in .env[/red]"
        )
        return

    console.print("[bold]Testing WeChat MP API connection...[/bold]")
    publisher = WechatMPPublisher()
    result = publisher.test_connection()

    if result.success:
        console.print(f"[green]✓ {result.message}[/green]")
    else:
        console.print(f"[red]✗ {result.message}[/red]")


@wechat.command("preview")
@click.option("--output", "-o", default="./wechat_preview.html", help="Output HTML file path")
@click.option("--min-importance", default=5, help="Minimum importance score (default: 5)")
@click.option("--max-items", default=30, help="Maximum items in digest (default: 30)")
@click.option("--no-cluster", is_flag=True, help="Skip automatic event clustering")
@click.pass_context
def preview(ctx, output, min_importance, max_items, no_cluster):
    """Generate a preview HTML file for browser viewing.

    Examples:
        bb wechat preview
        bb wechat preview --output ./preview.html
    """
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    from config.settings import PROJECT_ROOT
    from src.processors.digest_processor import DigestGenerator, create_digest_preview

    mock = ctx.obj.get("mock", False)
    db = get_db()

    try:
        console.print("[bold]Generating digest for preview...[/bold]")
        generator = DigestGenerator(db=db, use_mock=mock)

        if no_cluster:
            digest_result = generator.generate_digest(
                min_importance=min_importance,
                max_items=max_items,
                run_clustering=False,
                include_delivered=True,
            )
        else:
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
                    "[bold]Clustering events...[/bold]", total=None, clustered=0
                )

                def on_cluster_progress(current, total, clustered):
                    progress.update(
                        task, total=total, completed=current, clustered=clustered
                    )

                digest_result = generator.generate_digest(
                    min_importance=min_importance,
                    max_items=max_items,
                    run_clustering=True,
                    clustering_progress_callback=on_cluster_progress,
                    include_delivered=True,
                )

        if not digest_result.items and not digest_result.events:
            console.print("[yellow]No items to include in digest[/yellow]")
            return

        # Render using the WeChat publisher (same size reduction as publish)
        from src.delivery.wechat_mp import WechatMPPublisher

        publisher = WechatMPPublisher.__new__(WechatMPPublisher)
        from jinja2 import Environment, FileSystemLoader
        publisher.jinja_env = Environment(
            loader=FileSystemLoader(PROJECT_ROOT / "templates"),
            autoescape=False,
        )
        html = publisher._render_for_wechat(digest_result)

        # Wrap in a full HTML document for browser preview
        full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WeChat MP Preview</title>
    <style>
        body {{ background: #f5f5f5; margin: 0; padding: 20px; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_html, encoding="utf-8")

        event_count = len(digest_result.events) if digest_result.events else 0
        item_count = len(digest_result.items)
        console.print(
            f"[green]✓ Preview saved to {output_path}[/green]\n"
            f"  Events: {event_count}, Items: {item_count}\n"
            f"  Content: {len(html)} chars\n"
            f"  Open in browser to check layout."
        )

    finally:
        db.close()
