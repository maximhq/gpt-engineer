"""
CLI entry point for the GPT Engineer web application.

This module provides a command-line interface to run the web application.
"""
import argparse
import os
import sys

from pathlib import Path

from gpt_engineer.applications.web.app import app

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for the web application."""
    parser = argparse.ArgumentParser(
        description="GPT Engineer Web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m gpt_engineer.applications.web
  python -m gpt_engineer.applications.web --host 0.0.0.0 --port 8080
  python -m gpt_engineer.applications.web --debug
        """,
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to bind the server to (default: 5001)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload on code changes"
    )

    args = parser.parse_args()

    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "⚠️  Warning: OPENAI_API_KEY not set. Please set it in your environment or .env file."
        )
        print("   You can set it with: export OPENAI_API_KEY='your-api-key'")
        print()

    print("🚀 Starting GPT Engineer Web UI...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Debug: {args.debug}")
    print(f"   Auto-reload: {args.reload}")
    print()
    print(f"📱 Open your browser and navigate to: http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop the server")
    print()

    try:
        app.run(
            host=args.host, port=args.port, debug=args.debug, use_reloader=args.reload
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
