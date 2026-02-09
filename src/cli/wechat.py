"""CLI commands for WeChat Official Account (公众号) publishing."""

from pathlib import Path

import click

from src.cli.common import console, get_db


@click.group()
def wechat():
    """WeChat Official Account (公众号) commands."""
    pass


@wechat.command("publish")
@click.option("--draft-only", is_flag=True, help="Only create a draft, do not publish")
@click.option("--dry-run", is_flag=True, help="Preview locally without calling API")
@click.option("--min-importance", default=7, help="Minimum importance score (default: 7)")
@click.option("--max-items", default=30, help="Maximum items in digest (default: 30)")
@click.pass_context
def publish(ctx, draft_only, dry_run, min_importance, max_items):
    """Publish daily digest to WeChat Official Account.

    Examples:
        bb wechat publish              # Full publish
        bb wechat publish --draft-only # Create draft only
        bb wechat publish --dry-run    # Local preview, no API calls
    """
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
        digest_result = generator.generate_digest(
            min_importance=min_importance,
            max_items=max_items,
            run_clustering=False,
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
            # Render HTML locally for preview
            publisher = WechatMPPublisher.__new__(WechatMPPublisher)
            from jinja2 import Environment, FileSystemLoader
            from config.settings import PROJECT_ROOT
            publisher.jinja_env = Environment(
                loader=FileSystemLoader(PROJECT_ROOT / "templates"),
                autoescape=False,
            )
            html = publisher._render_digest_html(digest_result)
            html = publisher._sanitize_html(html)
            title = publisher._generate_title(digest_result)

            console.print(f"\n[bold]Title:[/bold] {title}")
            console.print(f"[bold]HTML length:[/bold] {len(html)} chars")
            console.print("[yellow]Dry run - no API calls made[/yellow]")
            return

        # Publish
        publisher = WechatMPPublisher()
        mode = "draft" if draft_only else "publish"
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
@click.option("--min-importance", default=7, help="Minimum importance score (default: 7)")
@click.option("--max-items", default=30, help="Maximum items in digest (default: 30)")
@click.pass_context
def preview(ctx, output, min_importance, max_items):
    """Generate a preview HTML file for browser viewing.

    Examples:
        bb wechat preview
        bb wechat preview --output ./preview.html
    """
    from config.settings import PROJECT_ROOT
    from src.processors.digest_processor import DigestGenerator, create_digest_preview

    mock = ctx.obj.get("mock", False)
    db = get_db()

    try:
        console.print("[bold]Generating digest for preview...[/bold]")
        generator = DigestGenerator(db=db, use_mock=mock)
        digest_result = generator.generate_digest(
            min_importance=min_importance,
            max_items=max_items,
            run_clustering=False,
        )

        if not digest_result.items and not digest_result.events:
            console.print("[yellow]No items to include in digest[/yellow]")
            return

        # Render using the WeChat template
        from jinja2 import Environment, FileSystemLoader
        from src.delivery.wechat_mp import WechatMPPublisher

        env = Environment(
            loader=FileSystemLoader(PROJECT_ROOT / "templates"),
            autoescape=False,
        )
        template = env.get_template("wechat_digest.html")
        html = template.render(
            date=digest_result.generated_at.strftime("%Y年%m月%d日"),
            total_items=digest_result.total_items,
            event_count=len(digest_result.events),
            individual_count=len(digest_result.regular_items),
            paper_count=len(digest_result.papers),
            events=digest_result.events,
            events_by_category=digest_result.events_by_category,
            regular_items_by_category=digest_result.regular_items_by_category,
            papers_by_category=digest_result.papers_by_category,
            items=digest_result.items,
        )

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
            f"  Open in browser to check layout."
        )

    finally:
        db.close()
