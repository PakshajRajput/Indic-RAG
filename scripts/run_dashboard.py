"""Launcher script for IndicRAG-QA Interactive Research Dashboard.

Usage:
    python scripts/run_dashboard.py [--host 127.0.0.1] [--port 8000] [--reload]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IndicRAG-QA Web Application & Research Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable live auto-reload for local development")
    args = parser.parse_args()

    print("=" * 65)
    print("INDICRAG-QA — RESEARCH OPERATIONS DASHBOARD")
    print(f"Server URL: http://{args.host}:{args.port}")
    print("Serving multi-agent graph, interactive workspace, and evaluation lab")
    print("=" * 65)

    uvicorn.run("indicrag.web.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
