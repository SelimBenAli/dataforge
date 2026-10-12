"""Command-line entry point."""
import argparse
import socket
import sys


def validate_port(value: str) -> int:
    """Parse a TCP port and reject values outside the usable range."""
    try: port = int(value)
    except (TypeError, ValueError) as exc: raise argparse.ArgumentTypeError("Port must be an integer.") from exc
    if not 1 <= port <= 65535: raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")
    return port


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dataforge-sba", description="DataForge local data science toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    dashboard = sub.add_parser("dashboard", help="Launch the local dashboard")
    dashboard.add_argument("--port", type=validate_port, default=5000)
    dashboard.add_argument("--host", default="127.0.0.1", help="Bind address (defaults to this computer only)")
    args = parser.parse_args(argv)
    if args.command == "dashboard":
        try:
            with socket.socket() as sock: sock.bind((args.host, args.port))
        except OSError as exc:
            print(f"Cannot use {args.host}:{args.port}: {exc}. Choose another available port with --port.", file=sys.stderr)
            return 2
        from dataforge_sba.dashboard import create_app
        print(f"DataForge is available at http://{args.host}:{args.port} (press Ctrl+C to stop)")
        create_app().run(host=args.host, port=args.port, debug=False, use_reloader=False)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
