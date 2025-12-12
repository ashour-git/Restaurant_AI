#!/usr/bin/env python3
"""
Smart Restaurant SaaS - Single Entry Point

Usage:
    python run.py              # Start backend on port 5000
    python run.py --port 8000  # Use custom port
    python run.py --prod       # Production mode (no reload)
"""

import argparse
import os
import sys
from pathlib import Path

# Add project paths
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "backend"))


def main():
    parser = argparse.ArgumentParser(description="Smart Restaurant SaaS")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--prod", action="store_true", help="Production mode")
    args = parser.parse_args()

    # Load environment from backend .env
    from dotenv import load_dotenv

    env_file = ROOT_DIR / "backend" / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    # Change to backend directory
    os.chdir(ROOT_DIR / "backend")

    # Print startup info
    print(
        f"""
╔══════════════════════════════════════════════════════════════════╗
║  🍽️  SMART RESTAURANT SAAS                                       ║
║  AI-Powered Restaurant Management Platform                       ║
╠══════════════════════════════════════════════════════════════════╣
║  📡 API:    http://{args.host}:{args.port}                                   ║
║  📚 Docs:   http://{args.host}:{args.port}/docs                              ║
║  🤖 AI:     Groq Llama 3.3 70B                                   ║
╚══════════════════════════════════════════════════════════════════╝
    """
    )

    # Start server
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.prod,
        log_level="info",
    )


if __name__ == "__main__":
    main()
