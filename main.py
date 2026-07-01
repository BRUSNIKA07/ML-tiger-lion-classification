"""
Main training pipeline for tiger vs lion classification.
"""
import argparse
from pathlib import Path
from src.train import train_model

def main():
    parser = argparse.ArgumentParser(
        description="Train tiger vs lion classification model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to dataset directory (default: ./data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./artifacts/best_model.pt",
        help="Path to save model weights (default: ./artifacts/best_model.pt)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers for data loading (default: 2)",
    )
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Don't freeze backbone weights (full fine-tuning)",
    )

    args = parser.parse_args()

    # Validate data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{args.data_dir}' not found!")
        print("Please run: python src/splitDataset.py")
        return

    print("=" * 60)
    print("Tiger vs Lion Classification - Training Pipeline")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model output: {args.model_path}")
    print(f"Freeze backbone: {not args.no_freeze}")
    print("=" * 60)

    # Train model
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_path=args.model_path,
        num_workers=args.num_workers,
        freeze_backbone=not args.no_freeze,
    )

if __name__ == "__main__":
    main()
