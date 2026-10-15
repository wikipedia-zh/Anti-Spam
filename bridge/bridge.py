#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from core import HEX_USERNAME, Ledger, POLICY, StopBridge, preview, read_source


def configuration(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"source_db", "state_dir", "destination_chat_id", "account_id", "hex_bot_id",
                "allowed_observers", "live", "receiver_scope_confirmed", "credentials_file", "session_file"}
    if set(config) != required:
        raise StopBridge("Unknown or missing configuration fields")
    if type(config["destination_chat_id"]) is not int or config["destination_chat_id"] >= 0:
        raise StopBridge("Set the destination's numeric group ID")
    for key in ("account_id", "hex_bot_id"):
        if type(config[key]) is not int or config[key] < 0:
            raise StopBridge("Invalid pinned Telegram identity")
    if any(type(config[k]) is not bool for k in ("live", "receiver_scope_confirmed")):
        raise StopBridge("Live switches must be booleans")
    if not isinstance(config["allowed_observers"], list) or len(config["allowed_observers"]) > 3 or any(type(i) is not int or i <= 0 for i in config["allowed_observers"]):
        raise StopBridge("Invalid observer allowlist")
    for key in ("source_db", "state_dir", "credentials_file", "session_file"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    for key in ("credentials_file", "session_file"):
        if config["source_db"] == config[key] or (Path(config[key]).exists() and Path(config["source_db"]).exists() and os.path.samefile(config["source_db"], config[key])):
            raise StopBridge("Credentials must not use the source database path")
    return config


def binding(config):
    if not config["account_id"] or not config["hex_bot_id"] or config["account_id"] == config["hex_bot_id"]:
        raise StopBridge("Pin the user account and hexDrill numeric IDs first")
    return {k: config[k] for k in ("source_db", "destination_chat_id", "account_id", "hex_bot_id", "allowed_observers")}


def snapshot(config):
    return read_source(config["source_db"], [config["account_id"], config["hex_bot_id"]])


async def telegram_command(config, command, ledger=None):
    from telegram_client import Telegram, cycle, make_client
    client = make_client(config)
    try:
        if command == "login":
            await client.start()
        else:
            await client.connect()
            if not await client.is_user_authorized():
                raise StopBridge("Log in interactively first; service mode never requests login codes")
        me = await client.get_me()
        bot = await client.get_entity(HEX_USERNAME)
        if me.bot or not bot.bot:
            raise StopBridge("Use a dedicated user account and the hexDrill bot")
        if command in ("login", "inspect"):
            groups = [d.id async for d in client.iter_dialogs() if d.id == config["destination_chat_id"]]
            print(json.dumps({"account_id": me.id, "hex_bot_id": bot.id, "destination_found": bool(groups)}))
            return
        transport = Telegram(client, config)
        last = None
        while True:
            result = await cycle(config, ledger, binding(config), transport, lambda: snapshot(config))
            if result != last:
                print(result, flush=True)
                last = result
            await asyncio.sleep(15)
    finally:
        await client.disconnect()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("preview")
    p.add_argument("--output", required=True)
    commands.add_parser("init")
    p = commands.add_parser("approve-history")
    p.add_argument("--file", required=True)
    p.add_argument("--sha256", required=True)
    commands.add_parser("status")
    commands.add_parser("login")
    commands.add_parser("inspect")
    commands.add_parser("run")
    args = parser.parse_args()
    config = configuration(args.config)
    if args.command == "preview":
        data = snapshot(config)
        fingerprint = preview(data, args.output)
        print(json.dumps({"policy": POLICY, "cases": len(data["cases"]), "users": len({v["user_id"] for v in data["cases"].values()}), "skipped": data["skipped"], "sha256": fingerprint}))
        return
    state = Path(config["state_dir"])
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix" and state.stat().st_mode & 0o077:
        raise StopBridge("State directory must have mode 700")
    # Login, inspection and delivery share the same session lock.
    if os.name != "posix":
        raise StopBridge("Run the service and ledger commands on Linux")
    import fcntl
    with (state / "bridge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command in ("login", "inspect"):
            asyncio.run(telegram_command(config, args.command))
            return
        bound = binding(config)
        ledger = Ledger(state / "state.db", config["source_db"])
        try:
            data = snapshot(config)
            if args.command == "init":
                ledger.initialize(data, bound)
                print("Initialized; historical cases await approval")
            elif args.command == "approve-history":
                print("Approved cases:", ledger.approve(args.file, args.sha256, data, bound))
            elif args.command == "status":
                ledger.check(data, bound)
                pending = ledger.pending()
                print(json.dumps({"pending": {k: pending[k] for k in ("id", "state", "message_id")} if pending else None}))
            else:
                if not config["live"] or not config["receiver_scope_confirmed"]:
                    raise StopBridge("Sending is disabled; complete preview review and receiver permissions first")
                ledger.check(data, bound)
                asyncio.run(telegram_command(config, "run", ledger))
        finally:
            ledger.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except StopBridge as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        # Exceptions from clients may contain phone numbers, request bodies or paths.
        print("Stopped:", type(error).__name__, file=sys.stderr)
        sys.exit(1)
