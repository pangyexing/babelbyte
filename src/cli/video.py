"""CLI commands for video generation."""

import json
from datetime import datetime
from pathlib import Path

import click

from src.cli.common import console, get_db


@click.group()
def video():
    """Video generation commands."""
    pass


@video.command("generate")
@click.option("--limit", default=5, help="Number of videos to generate")
@click.option(
    "--template",
    type=click.Choice(["news_brief", "key_points", "deep_analysis", "data_card"]),
    default="news_brief",
    help="Video template type (ignored if --auto-template)",
)
@click.option(
    "--auto-template",
    is_flag=True,
    help="Auto-select template based on content characteristics",
)
@click.option(
    "--ai-select",
    is_flag=True,
    help="Use AI to select content suitable for video broadcast",
)
@click.option(
    "--ai-script",
    is_flag=True,
    help="Use AI (32B) to generate polished TTS script",
)
@click.option(
    "--platform",
    type=click.Choice(["douyin", "shipinhao"]),
    default="douyin",
    help="Target platform (affects aspect ratio)",
)
@click.option("--voice", default="yunxi", help="TTS voice (yunxi, xiaoxiao, etc.)")
@click.option("--output-dir", type=click.Path(), default="./videos", help="Output directory")
@click.option("--min-importance", default=5, help="Minimum importance score for initial filter")
@click.option("--min-score", default=7, help="Minimum AI selection score (with --ai-select)")
@click.option("--bg-music", type=click.Path(exists=True), help="Background music file")
def generate(
    limit,
    template,
    auto_template,
    ai_select,
    ai_script,
    platform,
    voice,
    output_dir,
    min_importance,
    min_score,
    bg_music,
):
    """Generate videos from recent content.

    Use --ai-select to let AI choose which content is suitable for video:
    - Evaluates newsworthiness, visual appeal, audience engagement
    - Filters out technical tutorials, recruitment posts, etc.

    Use --auto-template to intelligently select templates based on content:
    - DATA_CARD: Content with numeric data/percentages
    - KEY_POINTS: Content with 3+ key points
    - DEEP_ANALYSIS: Content with impact assessment
    - NEWS_BRIEF: Default for other content

    Use --ai-script to polish the TTS narration with AI for more natural delivery.
    """
    from src.video.generator import VideoConfig, VideoGenerator
    from src.video.templates import TemplateType

    db = get_db()

    try:
        # Query candidates - only items in event clusters
        fetch_limit = limit * 3 if ai_select else limit
        clustered = db.get_undelivered_clustered_items(
            min_importance=min_importance,
            limit=fetch_limit,
        )

        if not clustered:
            console.print(
                "[yellow]No clustered content items found matching criteria.[/yellow]"
            )
            return

        # Flatten clustered items, keep sorted by importance
        items = [item for members in clustered.values() for item in members]
        items.sort(key=lambda x: (x.importance_score or 0), reverse=True)

        console.print(
            f"Found {len(items)} candidate items "
            f"across {len(clustered)} event clusters."
        )

        # AI content selection
        if ai_select:
            from src.video.content_intelligence import ContentSelector

            console.print("[cyan]AI selection: evaluating content suitability...[/cyan]")
            selector = ContentSelector()
            selected = selector.select_batch(items, min_score=min_score, max_items=limit)

            if not selected:
                console.print("[yellow]No content passed AI selection criteria.[/yellow]")
                return

            console.print(f"[green]AI selected {len(selected)} items for video:[/green]")
            for item, score, reason in selected:
                console.print(f"  • [{score}/10] {item.title[:50]}...")
                console.print(f"    [dim]{reason}[/dim]")

            # Extract just the items
            items = [item for item, _, _ in selected]
        else:
            items = items[:limit]
            console.print(f"Processing {len(items)} items.")

        if auto_template:
            console.print("[cyan]Auto-template: enabled[/cyan]")
        if ai_script:
            console.print("[cyan]AI script: enabled[/cyan]")

        # Configure generator
        template_map = {
            "news_brief": TemplateType.NEWS_BRIEF,
            "key_points": TemplateType.KEY_POINTS,
            "deep_analysis": TemplateType.DEEP_ANALYSIS,
            "data_card": TemplateType.DATA_CARD,
        }

        config = VideoConfig(
            output_dir=Path(output_dir),
            template_type=template_map.get(template, TemplateType.NEWS_BRIEF),
            auto_template=auto_template,
            use_ai_script=ai_script,
            platform=platform,
            voice=voice,
            bg_music_path=Path(bg_music) if bg_music else None,
        )

        generator = VideoGenerator(config)

        # Generate videos
        success_count = 0
        for i, item in enumerate(items, 1):
            console.print(f"\n[{i}/{len(items)}] Processing: {item.title[:50]}...")

            result = generator.generate_from_content(item)

            if result.success:
                success_count += 1
                console.print(f"  [green]✓[/green] Generated: {result.video_path}")
                console.print(f"    Duration: {result.duration:.1f}s, Slides: {result.slide_count}")
            else:
                console.print(f"  [red]✗[/red] Failed: {result.error}")

        console.print(f"\nGenerated {success_count}/{len(items)} videos.")
        console.print(f"Output directory: {output_dir}")

    finally:
        db.close()


@video.command("select")
@click.option("--limit", default=20, help="Number of items to evaluate")
@click.option("--min-importance", default=5, help="Minimum importance score for initial filter")
@click.option("--min-score", default=7, help="Minimum AI selection score")
def select(limit, min_importance, min_score):
    """Preview which content AI would select for video generation.

    Evaluates content without generating videos. Useful for testing
    the AI selection criteria before committing to video generation.
    """
    from rich.table import Table

    from src.video.content_intelligence import ContentSelector

    db = get_db()

    try:
        items = db.get_undelivered_items(
            min_importance=min_importance,
            limit=limit,
        )

        if not items:
            console.print("[yellow]No content items found.[/yellow]")
            return

        console.print(f"Evaluating {len(items)} items...")
        selector = ContentSelector()

        # Evaluate all items (not just selected ones)
        results = []
        for item in items:
            suitable, score, reason = selector.evaluate(item)
            results.append((item, suitable, score, reason))

        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)

        # Display as table
        table = Table(title="AI Content Selection Results")
        table.add_column("Score", justify="center", width=6)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Title", width=45)
        table.add_column("Reason", width=30)

        selected_count = 0
        for item, suitable, score, reason in results:
            if suitable and score >= min_score:
                status = "[green]✓ 选中[/green]"
                selected_count += 1
            elif suitable:
                status = "[yellow]○ 备选[/yellow]"
            else:
                status = "[dim]✗ 跳过[/dim]"

            score_color = "green" if score >= 8 else "yellow" if score >= 6 else "red"
            table.add_row(
                f"[{score_color}]{score}[/{score_color}]",
                status,
                item.title[:45] if item.title else "无标题",
                reason[:30] if reason else "",
            )

        console.print(table)
        console.print(f"\n[green]Selected: {selected_count}[/green] / {len(items)} items")
        console.print(f"[dim]Min score: {min_score}, Min importance: {min_importance}[/dim]")

    finally:
        db.close()


@video.command("auto")
@click.option("--limit", default=5, help="Maximum number of videos to generate")
@click.option("--days", default=3, help="Look back N days for events")
@click.option("--min-articles", default=2, help="Minimum articles in cluster")
@click.option("--output-dir", type=click.Path(), default="./videos", help="Output directory")
@click.option("--voice", default="yunxi", help="TTS voice")
@click.option("--ai-script", is_flag=True, help="Use AI to generate polished TTS script")
@click.option("--dry-run", is_flag=True, help="Preview selection without generating videos")
def auto_generate(limit, days, min_articles, output_dir, voice, ai_script, dry_run):
    """Auto-select best events and generate videos.

    Automatically selects the most newsworthy event clusters from recent days
    and generates videos for them. Prioritizes by:
    - Article count (more sources = more significant)
    - Representative item importance score
    - Recency

    Example:
        bb video auto --limit 3 --days 7
        bb video auto --dry-run  # Preview only
    """
    from rich.table import Table

    from src.video.generator import EventVideoGenerator, VideoConfig
    from src.video.templates import TemplateType

    db = get_db()

    try:
        # Get recent event clusters
        clusters = db.get_recent_event_clusters(days=days, limit=limit * 3)

        if not clusters:
            console.print(f"[yellow]No event clusters found in the last {days} days.[/yellow]")
            return

        console.print(f"Found {len(clusters)} event clusters in the last {days} days.")

        # Score and rank clusters
        scored_clusters = []
        for cluster in clusters:
            if cluster.article_count < min_articles:
                continue

            # Get representative member (highest importance)
            members = db.get_event_members(cluster.id)
            if not members:
                continue

            # Find best representative
            best_member = max(members, key=lambda m: m.importance_score or 0)
            importance = best_member.importance_score or 5

            # Calculate score: article_count * 2 + importance + recency_bonus
            hours_ago = (datetime.now() - cluster.last_updated_at).total_seconds() / 3600
            recency_bonus = max(0, 5 - hours_ago / 12)  # Up to 5 points for recent events

            score = cluster.article_count * 2 + importance + recency_bonus
            scored_clusters.append((cluster, best_member, members, score, importance))

        # Sort by score descending
        scored_clusters.sort(key=lambda x: x[3], reverse=True)
        selected = scored_clusters[:limit]

        if not selected:
            console.print("[yellow]No suitable event clusters found.[/yellow]")
            return

        # Display selection
        table = Table(title=f"Selected Events for Video Generation (Top {len(selected)})")
        table.add_column("Score", justify="right", width=6)
        table.add_column("Articles", justify="center", width=8)
        table.add_column("Importance", justify="center", width=10)
        table.add_column("Event Title", width=50)
        table.add_column("Category", width=10)

        for cluster, rep, members, score, importance in selected:
            imp_color = "green" if importance >= 7 else "yellow" if importance >= 5 else "dim"
            table.add_row(
                f"{score:.1f}",
                str(cluster.article_count),
                f"[{imp_color}]{importance}[/{imp_color}]",
                cluster.event_title[:50] if cluster.event_title else "无标题",
                cluster.category or "未分类",
            )

        console.print(table)

        if dry_run:
            console.print("\n[dim]Dry run mode - no videos generated.[/dim]")
            return

        # Generate videos
        console.print(f"\n[cyan]Generating {len(selected)} videos...[/cyan]")

        config = VideoConfig(
            output_dir=Path(output_dir),
            template_type=TemplateType.NEWS_BRIEF,
            auto_template=True,
            use_ai_script=ai_script,
            voice=voice,
        )

        generator = EventVideoGenerator(config)
        success_count = 0

        for i, (cluster, rep, members, score, importance) in enumerate(selected, 1):
            console.print(f"\n[{i}/{len(selected)}] {cluster.event_title[:50]}...")

            result = generator.generate_from_event(cluster, members)

            if result.success:
                success_count += 1
                console.print(f"  [green]✓[/green] Generated: {result.video_path}")
                console.print(f"    Duration: {result.duration:.1f}s, Slides: {result.slide_count}")
            else:
                console.print(f"  [red]✗[/red] Failed: {result.error}")

        console.print(f"\n[green]Generated {success_count}/{len(selected)} videos.[/green]")
        console.print(f"Output directory: {output_dir}")

    finally:
        db.close()


@video.command("bulletin")
@click.option(
    "--platform",
    type=click.Choice(["douyin", "shipinhao"]),
    default="douyin",
    help="Target platform (affects aspect ratio)",
)
@click.option("--voice", default="yunxi", help="TTS voice (yunxi, xiaoxiao, etc.)")
@click.option("--output-dir", type=click.Path(), default="./videos", help="Output directory")
@click.option("--dry-run", is_flag=True, help="Preview selection without generating video")
@click.option("--min-articles", default=2, help="Minimum articles in cluster to include")
@click.option(
    "--days", default=1, help="Include clusters from last N days (default: 1 = today only)"
)
@click.option("--limit", default=10, help="Maximum number of events in bulletin (default: 10)")
@click.option(
    "--ai-bg",
    is_flag=True,
    help="Use AI-generated backgrounds (requires LocalAI with IMAGE_GEN_ENABLED=true)",
)
@click.option(
    "--all-clusters",
    is_flag=True,
    help="Use all recent clusters instead of only those from today's email digest",
)
def bulletin(platform, voice, output_dir, dry_run, min_articles, days, limit, ai_bg, all_clusters):
    """Generate daily news bulletin from today's email digest events.

    By default, uses the same event clusters that were included in today's
    email digest (items marked as delivered today). Use --all-clusters to
    include all recent clusters regardless of delivery status.

    Visual style: 抖音科技风格 (Douyin Tech Style) with neon gradients,
    glowing borders, and large typography optimized for mobile viewing.

    Example:
        bb video bulletin                    # From today's digest events
        bb video bulletin --dry-run          # Preview without generating
        bb video bulletin --all-clusters     # All recent clusters
        bb video bulletin --days 3 --all-clusters  # Last 3 days, all clusters
        bb video bulletin --platform shipinhao  # 6:7 aspect ratio
    """
    from datetime import datetime, timedelta

    from rich.table import Table

    from src.video.bulletin import BulletinGenerator
    from src.video.generator import BulletinVideoGenerator, VideoConfig

    db = get_db()

    try:
        if all_clusters:
            # Legacy mode: get all recent clusters by date
            clusters = db.get_recent_event_clusters(days=days, limit=50)

            if not clusters:
                console.print(
                    f"[yellow]No event clusters found in the last {days} day(s).[/yellow]"
                )
                return

            # Filter by date range and min_articles (创新/创业 exempt from min_articles)
            cutoff_date = (datetime.now() - timedelta(days=days - 1)).date()
            filtered_clusters = [
                c
                for c in clusters
                if c.last_updated_at.date() >= cutoff_date
                and (c.article_count >= min_articles or c.category in ("创新", "创业"))
            ]

            if not filtered_clusters:
                date_range = "today" if days == 1 else f"the last {days} days"
                msg = f"No event clusters from {date_range} with >= {min_articles} articles."
                console.print(f"[yellow]{msg}[/yellow]")
                console.print(
                    f"[dim]Found {len(clusters)} clusters total, "
                    f"but none match the criteria.[/dim]"
                )
                return
        else:
            # Default: get clusters from today's email digest (delivered today)
            target_date = (datetime.now() - timedelta(days=days - 1)).date()
            clusters = db.get_delivered_event_clusters(target_date=target_date, limit=50)

            if not clusters:
                console.print(
                    "[yellow]No delivered event clusters found for today's digest.[/yellow]"
                )
                console.print(
                    "[dim]Run 'bb daily' first to send the digest, "
                    "or use --all-clusters to include all recent clusters.[/dim]"
                )
                return

            # Filter by min_articles (创新/创业 exempt)
            filtered_clusters = [
                c
                for c in clusters
                if c.article_count >= min_articles or c.category in ("创新", "创业")
            ]

            if not filtered_clusters:
                msg = f"No delivered clusters with >= {min_articles} articles."
                console.print(f"[yellow]{msg}[/yellow]")
                console.print(
                    f"[dim]Found {len(clusters)} delivered clusters, "
                    f"but none match the criteria.[/dim]"
                )
                return

        # Get representative importance for each cluster
        cluster_importance = {}
        for cluster in filtered_clusters:
            members = db.get_event_members(cluster.id)
            if members:
                max_importance = max(m.importance_score or 0 for m in members)
                cluster_importance[cluster.id] = max_importance
            else:
                cluster_importance[cluster.id] = 0

        # Sort: 创业/创新 first, then others by article_count and importance
        def sort_key(c):
            is_priority = c.category in ("创业", "创新")
            importance = cluster_importance.get(c.id, 0)
            return (0 if is_priority else 1, -c.article_count, -importance)

        filtered_clusters = sorted(filtered_clusters, key=sort_key)
        total_found = len(filtered_clusters)
        if limit and len(filtered_clusters) > limit:
            filtered_clusters = filtered_clusters[:limit]

        source = "today's digest" if not all_clusters else (
            "today" if days == 1 else f"the last {days} days"
        )
        if total_found > len(filtered_clusters):
            console.print(
                f"Found {total_found} event clusters from {source}, "
                f"using top [green]{len(filtered_clusters)}[/green] (--limit {limit})."
            )
        else:
            console.print(
                f"Found [green]{len(filtered_clusters)}[/green] "
                f"event clusters from {source}."
            )

        # Get members for each cluster
        members_dict = {}
        for cluster in filtered_clusters:
            members = db.get_event_members(cluster.id)
            if members:
                members_dict[cluster.id] = members

        # Display preview table
        table_title = (
            "Today's Digest Events" if not all_clusters else (
                "Today's Event Clusters" if days == 1
                else f"Event Clusters (Last {days} Days)"
            )
        )
        table = Table(title=table_title)
        table.add_column("#", justify="right", width=4)
        table.add_column("Articles", justify="center", width=8)
        table.add_column("Category", width=10)
        table.add_column("Event Title", width=50)

        for i, cluster in enumerate(filtered_clusters, 1):
            table.add_row(
                str(i),
                str(cluster.article_count),
                cluster.category or "未分类",
                cluster.event_title[:50] if cluster.event_title else "无标题",
            )

        console.print(table)

        if dry_run:
            console.print("\n[dim]Dry run mode - no video generated.[/dim]")

            # Generate bulletin content for preview
            console.print("\n[cyan]Generating bulletin content preview...[/cyan]")
            generator = BulletinGenerator()
            result = generator.generate_bulletin(filtered_clusters, members_dict)

            if result.success:
                console.print(
                    f"\n[green]Bulletin generated with {result.total_events} events.[/green]"
                )
                console.print("\n[bold]Headlines:[/bold]")
                for i, item in enumerate(result.items, 1):
                    console.print(f"  {i}. {item.headline}")
                    console.print(f"     [dim]{item.summary}[/dim]")

                console.print("\n[bold]TTS Script Preview:[/bold]")
                console.print(
                    f"[dim]{result.script[:500]}...[/dim]"
                    if len(result.script) > 500
                    else f"[dim]{result.script}[/dim]"
                )
            else:
                console.print(f"[red]Bulletin generation failed: {result.error}[/red]")
            return

        # Generate bulletin
        console.print("\n[cyan]Generating bulletin content...[/cyan]")
        bulletin_gen = BulletinGenerator()
        bulletin_result = bulletin_gen.generate_bulletin(filtered_clusters, members_dict)

        if not bulletin_result.success:
            console.print(f"[red]Bulletin generation failed: {bulletin_result.error}[/red]")
            return

        event_count = bulletin_result.total_events
        console.print(f"[green]✓[/green] Bulletin content generated with {event_count} events.")

        # Generate video
        console.print("\n[cyan]Generating video...[/cyan]")

        config = VideoConfig(
            output_dir=Path(output_dir),
            platform=platform,
            voice=voice,
        )

        video_gen = BulletinVideoGenerator(config, ai_bg=ai_bg)
        video_result = video_gen.generate_from_bulletin(bulletin_result)

        if video_result.success:
            console.print("\n[green]✓[/green] Video generated successfully!")
            console.print(f"  Path: {video_result.video_path}")
            console.print(f"  Duration: {video_result.duration:.1f}s")
            console.print(f"  Events: {video_result.event_count}")
        else:
            console.print(f"\n[red]✗[/red] Video generation failed: {video_result.error}")

    finally:
        # Always release GPU memory from image generator and TTS
        try:
            ig = getattr(getattr(video_gen, "template", None), "_image_generator", None)
            if ig is not None:
                ig.unload()
                console.print("[dim]Image generator GPU memory released.[/dim]")
        except NameError:
            pass
        try:
            from src.video.tts import QwenTTS

            tts = getattr(video_gen, "tts", None)
            if isinstance(tts, QwenTTS):
                tts.unload()
                console.print("[dim]Qwen TTS GPU memory released.[/dim]")
        except NameError:
            pass
        db.close()


@video.command("generate-event")
@click.argument("cluster_id", type=int)
@click.option("--output-dir", type=click.Path(), default="./videos", help="Output directory")
@click.option("--voice", default="yunxi", help="TTS voice")
@click.option("--auto-template", is_flag=True, help="Auto-select template based on content")
def generate_event(cluster_id, output_dir, voice, auto_template):
    """Generate video from an event cluster."""
    from src.video.generator import EventVideoGenerator, VideoConfig
    from src.video.templates import TemplateType

    db = get_db()

    try:
        # Get cluster and members
        cluster = db.get_event_cluster(cluster_id)
        if not cluster:
            console.print(f"[red]Event cluster {cluster_id} not found.[/red]")
            return

        members = db.get_event_members(cluster_id)
        if not members:
            console.print(f"[red]No members found for cluster {cluster_id}.[/red]")
            return

        console.print(f"Event: {cluster.event_title}")
        console.print(f"Members: {len(members)} articles")

        config = VideoConfig(
            output_dir=Path(output_dir),
            template_type=TemplateType.NEWS_BRIEF,
            auto_template=auto_template,
            voice=voice,
        )

        generator = EventVideoGenerator(config)
        result = generator.generate_from_event(cluster, members)

        if result.success:
            console.print(f"\n[green]✓[/green] Generated: {result.video_path}")
            console.print(f"  Duration: {result.duration:.1f}s")
        else:
            console.print(f"\n[red]✗[/red] Failed: {result.error}")

    finally:
        db.close()


@video.command("voices")
def list_voices():
    """List available TTS voices."""
    import asyncio

    from src.video.tts import CHINESE_VOICES, QWEN_SPEAKERS
    from src.video.tts import list_voices as async_list_voices

    console.print("[bold]Edge TTS voice shortcuts:[/bold]\n")
    for name, voice_id in CHINESE_VOICES.items():
        console.print(f"  {name:12} -> {voice_id}")

    console.print("\n[bold]Qwen3-TTS speakers:[/bold]\n")
    for shortcut, speaker_name in QWEN_SPEAKERS.items():
        console.print(f"  {shortcut:12} -> {speaker_name}")
    console.print("  [dim]Set TTS_PROVIDER=qwen to use Qwen3-TTS[/dim]")

    console.print("\n\n[bold]Fetching all Edge TTS voices...[/bold]")
    try:
        voices = asyncio.run(async_list_voices("zh"))
        console.print(f"\nFound {len(voices)} Chinese voices:\n")
        for v in voices[:20]:
            gender = v.get("Gender", "Unknown")
            locale = v.get("Locale", "Unknown")
            name = v.get("ShortName", "Unknown")
            console.print(f"  {name:30} ({gender}, {locale})")
    except Exception as e:
        console.print(f"[red]Error fetching voices: {e}[/red]")


@video.command("test-tts")
@click.argument("text")
@click.option("--voice", default="yunxi", help="Voice to use (edge) or speaker name (qwen)")
@click.option("--output", default=None, help="Output file (default: auto based on provider)")
@click.option(
    "--provider",
    type=click.Choice(["edge", "qwen", "auto"]),
    default="auto",
    help="TTS provider (auto reads TTS_PROVIDER env var)",
)
def test_tts(text, voice, output, provider):
    """Test TTS with a sample text."""
    import os

    from src.video.tts import QwenTTS, get_tts

    # Override TTS_PROVIDER if explicit --provider given
    if provider != "auto":
        os.environ["TTS_PROVIDER"] = provider
        import config.settings as _cs

        _cs._settings = None  # force re-read with new env var

    tts = get_tts(voice=voice)
    is_qwen = isinstance(tts, QwenTTS)
    provider_name = "qwen" if is_qwen else "edge"

    if output is None:
        output = f"./test_voice{'.wav' if is_qwen else '.mp3'}"

    console.print(f"Provider: {provider_name}")
    console.print(f"Voice: {voice}")
    console.print(f"Text: {text}")

    output_path = Path(output)

    try:
        result = tts.synthesize(text, output_path, voice if not is_qwen else None)
        console.print(f"\n[green]✓[/green] Audio saved to: {result}")
    except Exception as e:
        console.print(f"\n[red]✗[/red] Error: {e}")
    finally:
        if is_qwen:
            tts.unload()


@video.command("preview")
@click.argument("content_id", type=int)
@click.option(
    "--template",
    type=click.Choice(["news_brief", "key_points", "deep_analysis", "data_card"]),
    default="news_brief",
    help="Video template type (ignored if --auto-template)",
)
@click.option(
    "--auto-template",
    is_flag=True,
    help="Auto-select template based on content characteristics",
)
@click.option("--output", default="./preview.png", help="Output image file")
def preview(content_id, template, auto_template, output):
    """Preview a single slide for a content item.

    Use --auto-template to let the system choose the best template
    based on content characteristics (numeric data, key points, etc.)
    """
    from src.video.templates import SlideContent, TemplateType, get_template

    db = get_db()

    try:
        item = db.get_content_item(content_id)

        if not item:
            console.print(f"[red]Content item {content_id} not found.[/red]")
            return

        console.print(f"Title: {item.title}")

        # Select template
        if auto_template:
            from src.video.content_intelligence import extract_keywords, select_template

            template_type = select_template(item)
            console.print(f"[cyan]Auto-selected template: {template_type.value}[/cyan]")
            highlights = extract_keywords(item)
        else:
            template_map = {
                "news_brief": TemplateType.NEWS_BRIEF,
                "key_points": TemplateType.KEY_POINTS,
                "deep_analysis": TemplateType.DEEP_ANALYSIS,
                "data_card": TemplateType.DATA_CARD,
            }
            template_type = template_map.get(template, TemplateType.NEWS_BRIEF)
            highlights = []

        tpl = get_template(template_type)

        # Parse key points for bullet_points display
        bullet_points = []
        if item.key_points:
            try:
                kp_data = json.loads(item.key_points)
                bullet_points = [kp.get("value", "") for kp in kp_data if kp.get("value")]
            except json.JSONDecodeError:
                pass

        # Create slide
        slide = SlideContent(
            title=item.title or "Preview",
            body=item.summary or "",
            bullet_points=bullet_points[:4],
            category=item.category or "资讯",
            importance=item.importance_score or 5,
            highlights=highlights,
        )

        # Render
        img = tpl.render_slide(slide)
        img.save(output)

        console.print(f"\n[green]✓[/green] Preview saved to: {output}")
        if highlights:
            console.print(f"  Highlights: {', '.join(highlights[:3])}")

    finally:
        db.close()
