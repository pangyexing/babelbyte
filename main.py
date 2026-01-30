#!/usr/bin/env python3
"""BabelByte - AI 内容订阅系统主入口"""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings


def setup_logging():
    """Setup logging configuration."""
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.logging.level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.logging.path, encoding="utf-8"),
        ],
    )


def main():
    """Main entry point."""
    setup_logging()

    # Import CLI after logging is configured
    from src.cli.commands import cli

    cli()


if __name__ == "__main__":
    main()
