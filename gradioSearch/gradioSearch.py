"""
Main module for gradioSearch CLI tool
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="CLI tool for searching FAISS vector databases with Gradio GUI"
    )
    
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the Langchain FAISS database"
    )
    
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Embedding model name for sentence-transformers"
    )
    
    parser.add_argument(
        "--metadata_keys",
        type=str,
        required=True,
        help="Comma-separated list of metadata columns to display"
    )
    
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of top results to retrieve (default: 50)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Validate db_path exists
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database path '{args.db_path}' does not exist")
        return 1
    
    # Parse metadata keys
    metadata_keys = [key.strip() for key in args.metadata_keys.split(',')]
    
    print(f"Database path: {args.db_path}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Metadata keys: {metadata_keys}")
    print(f"Top-k results: {args.topk}")
    
    # TODO: Initialize GUI and start application
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
