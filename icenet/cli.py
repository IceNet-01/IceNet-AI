"""
Command-line interface for IceNet
"""

import argparse
import sys
import logging
from pathlib import Path

from icenet import __version__
from icenet.core.engine import IceNetEngine
from icenet.core.config import Config
from icenet.training.trainer import Trainer
from icenet.ui.app import run_ui
from icenet.utils.updater import check_for_updates, install_update
from icenet.chat import run_chat_loop
from icenet.data.local_loader import LocalFileLoader


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_interactive(args):
    """Launch interactive terminal UI"""
    logger.info("Launching IceNet interactive mode...")
    run_ui()


def cmd_train(args):
    """Train a model"""
    logger.info(f"Starting training with config: {args.config}")

    # Load config
    config = Config.from_yaml(args.config)

    # Override config with CLI args if provided
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.epochs:
        config.training.epochs = args.epochs

    # Create engine
    engine = IceNetEngine(config)
    logger.info(f"Engine initialized: {engine}")

    # Load data
    # TODO: Implement data loading based on config
    logger.warning("Data loading not yet implemented - placeholder only")

    # Create trainer
    # trainer = Trainer(engine, train_loader, val_loader)
    # trainer.train()

    logger.info("Training placeholder completed")


def cmd_eval(args):
    """Evaluate a model"""
    logger.info(f"Evaluating model: {args.checkpoint}")

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Create engine
    engine = IceNetEngine(config)

    # Load checkpoint
    engine.load_checkpoint(args.checkpoint)

    logger.info("Model loaded successfully")
    logger.info(f"Model summary: {engine.summary()}")

    # TODO: Implement evaluation


def cmd_generate(args):
    """Generate text with a trained model"""
    logger.info(f"Generating with model: {args.checkpoint}")

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Create engine
    engine = IceNetEngine(config)

    # Load checkpoint
    engine.load_checkpoint(args.checkpoint)
    engine.eval_mode()

    logger.info("Model loaded successfully")

    # TODO: Implement text generation
    logger.warning("Text generation not yet implemented")


def cmd_info(args):
    """Display system information"""
    from icenet.core.device import DeviceManager

    device_manager = DeviceManager()

    print("\n=== IceNet System Information ===\n")
    print(f"Version: {__version__}")
    print(f"\n{device_manager}")
    print(f"\nDevice Info:")

    info = device_manager.get_info_dict()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 40 + "\n")


def cmd_update(args):
    """Check for and install updates"""
    logger.info("Checking for updates...")

    has_update, latest_version = check_for_updates()

    if has_update:
        logger.info(f"Update available: {latest_version}")

        if args.auto or input("Install update? (y/n): ").lower() == "y":
            logger.info("Installing update...")
            success = install_update()

            if success:
                logger.info("Update installed successfully!")
                logger.info("Please restart IceNet to use the new version")
            else:
                logger.error("Update installation failed")
    else:
        logger.info("IceNet is up to date!")


def cmd_config(args):
    """Generate a sample configuration file"""
    config = Config()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config.to_yaml(output_path)

    logger.info(f"Sample config saved to: {output_path}")


def cmd_train_local(args):
    """Train on local files - Simple, no-config-needed training"""
    print("\n" + "=" * 60)
    print("IceNet Local File Training")
    print("=" * 60 + "\n")

    # Load files
    print(f"📂 Scanning files in: {args.path}")
    loader = LocalFileLoader(
        root_path=args.path,
        recursive=not args.no_recursive,
        max_file_size_mb=args.max_file_size,
    )

    # Get statistics
    files = loader.scan_files()
    stats = loader.get_statistics(files)

    print(f"\n📊 Dataset Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")
    print(f"\n  Files by type:")
    for ext, count in sorted(stats['by_extension'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {ext}: {count} files")

    # Confirm
    if not args.yes:
        response = input(f"\n⚡ Train on these {stats['total_files']} files? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return

    # Load data
    print("\n📖 Loading file contents...")
    chunks = loader.load_as_chunks(
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    print(f"✓ Created {len(chunks)} training chunks")

    # Save processed data
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file = output_dir / "training_data.txt"
    print(f"\n💾 Saving training data to: {data_file}")

    with open(data_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + "\n\n---\n\n")

    print(f"✓ Saved {len(chunks)} chunks")

    # Save metadata
    import json
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'source_path': str(args.path),
            'total_files': stats['total_files'],
            'total_chunks': len(chunks),
            'chunk_size': args.chunk_size,
            'statistics': stats,
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print("✅ READY TO CHAT! Your files are processed!")
    print(f"{'=' * 60}")
    print(f"\n📊 What I learned:")
    print(f"  • {stats['total_files']} files processed")
    print(f"  • {len(chunks)} chunks created")
    print(f"  • {stats['total_size_mb']:.2f} MB of data")
    print(f"\n💬 Start chatting NOW:")
    print(f"  icenet chat")
    print(f"\n   I'll answer questions by searching your files!")
    print(f"   Ask me anything about your documents.\n")
    print(f"💾 Data saved to: {output_dir}")
    print(f"   (Chat will automatically load from this location)")
    print()


def cmd_chat(args):
    """Start a chat session"""
    model_path = args.model if hasattr(args, 'model') and args.model else None
    # Use the same default path as train-local
    data_dir = str(Path("~/icenet/training").expanduser())
    run_chat_loop(model_path=model_path, data_dir=data_dir)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description=f"IceNet AI v{__version__} - Apple M4 Pro Optimized AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version", action="version", version=f"IceNet AI v{__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Interactive mode
    parser_interactive = subparsers.add_parser(
        "interactive", help="Launch interactive terminal UI", aliases=["ui", "i"]
    )
    parser_interactive.set_defaults(func=cmd_interactive)

    # Train command
    parser_train = subparsers.add_parser("train", help="Train a model")
    parser_train.add_argument("--config", "-c", required=True, help="Config file path")
    parser_train.add_argument("--batch-size", "-b", type=int, help="Batch size")
    parser_train.add_argument("--learning-rate", "-lr", type=float, help="Learning rate")
    parser_train.add_argument("--epochs", "-e", type=int, help="Number of epochs")
    parser_train.set_defaults(func=cmd_train)

    # Eval command
    parser_eval = subparsers.add_parser("eval", help="Evaluate a model")
    parser_eval.add_argument("--checkpoint", "-ckpt", required=True, help="Model checkpoint")
    parser_eval.add_argument("--config", "-c", help="Config file path")
    parser_eval.set_defaults(func=cmd_eval)

    # Generate command
    parser_generate = subparsers.add_parser("generate", help="Generate text")
    parser_generate.add_argument("--checkpoint", "-ckpt", required=True, help="Model checkpoint")
    parser_generate.add_argument("--config", "-c", help="Config file path")
    parser_generate.add_argument("--prompt", "-p", help="Input prompt")
    parser_generate.add_argument("--max-length", "-ml", type=int, default=100, help="Max length")
    parser_generate.set_defaults(func=cmd_generate)

    # Info command
    parser_info = subparsers.add_parser("info", help="Display system information")
    parser_info.set_defaults(func=cmd_info)

    # Update command
    parser_update = subparsers.add_parser("update", help="Check for updates")
    parser_update.add_argument("--auto", "-a", action="store_true", help="Auto-install updates")
    parser_update.set_defaults(func=cmd_update)

    # Config command
    parser_config = subparsers.add_parser("config", help="Generate sample config")
    parser_config.add_argument(
        "--output", "-o", default="config.yaml", help="Output path"
    )
    parser_config.set_defaults(func=cmd_config)

    # Train-local command (simple, no-config training)
    parser_train_local = subparsers.add_parser(
        "train-local",
        help="Train on your local files (no technical skills needed!)",
        aliases=["local", "learn"]
    )
    parser_train_local.add_argument(
        "path",
        help="Path to directory with your files (e.g., ~/Documents)"
    )
    parser_train_local.add_argument(
        "--output", "-o",
        default="~/icenet/training",
        help="Where to save training data (default: ~/icenet/training)"
    )
    parser_train_local.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of training chunks (default: 1000)"
    )
    parser_train_local.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Overlap between chunks (default: 100)"
    )
    parser_train_local.add_argument(
        "--max-file-size",
        type=float,
        default=10.0,
        help="Max file size in MB (default: 10)"
    )
    parser_train_local.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't scan subdirectories"
    )
    parser_train_local.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser_train_local.set_defaults(func=cmd_train_local)

    # Chat command
    parser_chat = subparsers.add_parser(
        "chat",
        help="Chat with IceNet AI",
        aliases=["talk", "ask"]
    )
    parser_chat.add_argument(
        "--model", "-m",
        help="Path to trained model (optional)"
    )
    parser_chat.set_defaults(func=cmd_chat)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help or launch interactive mode
    if not args.command:
        # Launch interactive mode by default
        cmd_interactive(args)
        return

    # Execute command
    if hasattr(args, "func"):
        try:
            args.func(args)
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
