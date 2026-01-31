"""CLI commands for BabelByte.

This module serves as the entry point for the CLI and registers
all command groups from their respective modules.
"""

import click

from src.cli import actions, embeddings, events, fetch, reports, search, subscribe, topics, validation


@click.group()
@click.option("--mock", is_flag=True, help="Use mock data for testing")
@click.pass_context
def cli(ctx, mock):
    """BabelByte - AI Content Subscription System"""
    ctx.ensure_object(dict)
    ctx.obj["mock"] = mock


# Register all command groups
subscribe.register_commands(cli)
fetch.register_commands(cli)
search.register_commands(cli)
events.register_commands(cli)
topics.register_commands(cli)
actions.register_commands(cli)
reports.register_commands(cli)
validation.register_commands(cli)
embeddings.register_commands(cli)


if __name__ == "__main__":
    cli()
