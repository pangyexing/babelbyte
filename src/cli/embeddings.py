"""Embeddings management CLI commands."""

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.cli.common import console, get_db


@click.group()
@click.pass_context
def embeddings(ctx):
    """Embedding management commands."""
    pass


@embeddings.command("compute")
@click.option("--limit", "-n", default=2000, help="Maximum items to process")
@click.option("--force", "-f", is_flag=True, help="Recompute existing embeddings")
@click.pass_context
def embeddings_compute(ctx, limit, force):
    """Compute embeddings for content items.

    Examples:
        bb embeddings compute --limit 100
        bb embeddings compute --force  # Recompute all
    """
    from config.settings import get_settings

    settings = get_settings()
    if not settings.embedding.enabled:
        console.print("[yellow]Embeddings are disabled. Set EMBEDDING_ENABLED=true to enable.[/yellow]")
        return

    try:
        from src.processors.embeddings import EmbeddingManager, embedding_to_bytes
    except ImportError as e:
        console.print(f"[red]Failed to import embeddings: {e}[/red]")
        console.print("[dim]Run: pip install sentence-transformers numpy[/dim]")
        return

    db = get_db()
    try:
        manager = EmbeddingManager.get_instance()

        if force:
            # Get all processed items
            items = db.get_undelivered_items(min_importance=1, limit=limit)
        else:
            # Get items without embeddings
            item_ids = db.get_content_ids_without_embeddings(limit=limit)
            items = [db.get_content_item(item_id) for item_id in item_ids]
            items = [item for item in items if item is not None]

        if not items:
            console.print("[dim]No items to process.[/dim]")
            return

        computed = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Computing embeddings for {len(items)} items...", total=len(items))

            for item in items:
                try:
                    text = f"{item.title} {item.summary or ''}"[:500]
                    embedding = manager.get_embedding(text)
                    embedding_bytes = embedding_to_bytes(embedding)

                    db.save_content_embedding(
                        content_id=item.id,
                        embedding=embedding_bytes,
                        model=settings.embedding.sentence_transformers_model
                        if settings.embedding.provider == "sentence-transformers"
                        else settings.embedding.openai_model,
                        dimension=manager.dimension,
                    )
                    computed += 1
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to compute embedding for item {item.id}: {e}[/yellow]")

                progress.advance(task)

        console.print(f"[green]Computed {computed} embeddings.[/green]")

    finally:
        db.close()


@embeddings.command("stats")
@click.pass_context
def embeddings_stats(ctx):
    """Show embedding statistics."""
    from config.settings import get_settings

    settings = get_settings()

    db = get_db()
    try:
        stats = db.get_embedding_stats()

        table = Table(title="Embedding Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Provider", settings.embedding.provider)
        table.add_row(
            "Model",
            settings.embedding.sentence_transformers_model
            if settings.embedding.provider == "sentence-transformers"
            else settings.embedding.openai_model,
        )
        table.add_row("Enabled", "Yes" if settings.embedding.enabled else "No")
        table.add_row("Content Embeddings", str(stats["content_embeddings"]))
        table.add_row("Cluster Centroids", str(stats["cluster_centroids"]))
        table.add_row("Processed Items", str(stats["processed_items"]))
        table.add_row("Coverage", f"{stats['coverage_percent']}%")
        table.add_row("Rule Weight", f"{settings.embedding.rule_weight:.0%}")
        table.add_row("Semantic Weight", f"{settings.embedding.semantic_weight:.0%}")

        console.print(table)

    finally:
        db.close()


@embeddings.command("rebuild-centroids")
@click.option("--limit", "-n", default=50, help="Maximum clusters to process")
@click.pass_context
def rebuild_centroids(ctx, limit):
    """Rebuild cluster centroids from member embeddings.

    Examples:
        bb embeddings rebuild-centroids
    """
    from config.settings import get_settings

    settings = get_settings()
    if not settings.embedding.enabled:
        console.print("[yellow]Embeddings are disabled.[/yellow]")
        return

    try:
        from src.processors.embeddings import (
            EmbeddingManager,
            bytes_to_embedding,
            compute_centroid,
            embedding_to_bytes,
        )
    except ImportError as e:
        console.print(f"[red]Failed to import embeddings: {e}[/red]")
        return

    db = get_db()
    try:
        manager = EmbeddingManager.get_instance()
        clusters = db.get_recent_event_clusters(days=30, limit=limit)

        if not clusters:
            console.print("[dim]No clusters found.[/dim]")
            return

        rebuilt = 0
        for cluster in clusters:
            members = db.get_event_members(cluster.id)
            if not members:
                continue

            # Collect member embeddings
            member_embeddings = []
            for member in members:
                emb_data = db.get_content_embedding(member.id)
                if emb_data:
                    emb_bytes, model, dim = emb_data
                    member_embeddings.append(bytes_to_embedding(emb_bytes, dim))

            if member_embeddings:
                try:
                    import numpy as np

                    centroid = compute_centroid(member_embeddings)
                    centroid_bytes = embedding_to_bytes(centroid)
                    db.save_cluster_centroid(cluster.id, centroid_bytes, len(member_embeddings))
                    rebuilt += 1
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to rebuild centroid for cluster {cluster.id}: {e}[/yellow]")

        console.print(f"[green]Rebuilt {rebuilt} cluster centroids.[/green]")

    finally:
        db.close()


def register_commands(cli):
    """Register embedding commands with the CLI."""
    cli.add_command(embeddings)
