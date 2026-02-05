"""CLI commands for BabelByte.

This module serves as the entry point for the CLI and registers
all command groups from their respective modules.
"""

import logging

import click

from src.cli import actions, db, embeddings, events, fetch, reports, search, subscribe, topics, validation, video


def setup_logging():
    """Setup logging configuration."""
    from config.settings import get_settings

    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.logging.level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.logging.path, encoding="utf-8"),
        ],
    )


@click.group()
@click.option("--mock", is_flag=True, help="Use mock data for testing")
@click.pass_context
def cli(ctx, mock):
    """BabelByte - AI Content Subscription System"""
    setup_logging()
    ctx.ensure_object(dict)
    ctx.obj["mock"] = mock


# Register all command groups
subscribe.register_commands(cli)
fetch.register_commands(cli)
db.register_commands(cli)
search.register_commands(cli)
events.register_commands(cli)
topics.register_commands(cli)
actions.register_commands(cli)
reports.register_commands(cli)
validation.register_commands(cli)
embeddings.register_commands(cli)

# Video generation commands
cli.add_command(video.video)


if __name__ == "__main__":
    cli()
