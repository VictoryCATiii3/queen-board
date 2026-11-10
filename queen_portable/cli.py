"""The `queen` command.

    queen analyze --fen "<FEN>"        # one position, prints the move and explanation
    queen serve                        # local HTTP server for chess apps (port 8765)
    queen fetch                        # download the model now instead of on first use
    queen info                         # what device/settings would be used

Common flags: --model {pawn-8,hce-4}  --device {auto,cuda,mps,cpu}  --quantize {int8,nf4}
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import add_config_args, config_from_args, pick_device


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="queen", description="Run Princeton's QUEEN chess model locally (unofficial).")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze one position")
    a.add_argument("--fen", required=True)
    a.add_argument("--history", nargs="*", default=[], help="prior FENs, oldest first")
    a.add_argument("--prompt", help="ask your own question instead of the standard analysis (experimental)")
    a.add_argument("--json", action="store_true", help="print the full JSON result")
    add_config_args(a)

    s = sub.add_parser("serve", help="run the local HTTP server")
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other machines on your network")
    s.add_argument("--port", type=int, default=8765)
    add_config_args(s)

    f = sub.add_parser("fetch", help="download the model now")
    add_config_args(f)

    i = sub.add_parser("info", help="show what would be used")
    add_config_args(i)

    args = p.parse_args(argv)
    cfg = config_from_args(args)

    if args.cmd == "info":
        print(json.dumps({"model": cfg.repo_id, "device": pick_device(cfg.device),
                          "quantize": cfg.quantize, "backend": cfg.backend,
                          "python": sys.version.split()[0]}, indent=2))
    elif args.cmd == "fetch":
        from .fetch import fetch
        print(fetch(cfg))
    elif args.cmd == "analyze":
        from .backends import load_backend
        from .types import AnalyzeRequest
        result = load_backend(cfg).analyze(AnalyzeRequest(fen=args.fen, history=args.history,
                                                            prompt=args.prompt))
        if args.json:
            print(json.dumps(result.raw, indent=2, ensure_ascii=False))
        else:
            print(result.text)
            print(f"\nBest move: {result.best_move_uci}  ({result.move_source})")
    elif args.cmd == "serve":
        from .server import serve
        serve(cfg, args.host, args.port)


if __name__ == "__main__":
    main()
