"""
One-command local demo.

    python scripts/demo.py

Runs preflight, starts the API, waits for it to report healthy, starts the
Streamlit UI, opens a browser. Ctrl+C stops both cleanly.

    --skip-preflight   start immediately, no checks
    --api-only         API without the UI
    --port / --ui-port override the defaults (8000 / 8501)
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import urllib.request
import urllib.error

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
)
if os.name == "nt" and not os.getenv("WT_SESSION"):
    GREEN = RED = YELLOW = DIM = BOLD = RESET = ""

processes: list[subprocess.Popen] = []


def say(msg, colour=""):
    print(f"{colour}{msg}{RESET}", flush=True)


def shutdown(*_):
    """
    Stop children on Ctrl+C.

    Terminate first and give each two seconds to exit — uvicorn needs that window
    to run its lifespan shutdown, and embedded Qdrant needs it to release the lock
    on its data directory. Killing immediately can leave that lock behind, and the
    next run then fails with a confusing 'already accessed by another instance'.
    """
    say("\nStopping...", DIM)
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 2
    for proc in processes:
        remaining = max(0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
    say("Stopped.", DIM)
    sys.exit(0)


def wait_for_health(url: str, timeout: int = 180) -> dict | None:
    """
    Poll /health until the API reports in.

    The timeout is generous on purpose. A first run loads the NeMo rails, and in
    local mode also downloads sentence-transformers weights (~420 MB) and the
    FlashRank ONNX model. That is minutes, not seconds, and only ever once.
    """
    import json

    deadline = time.time() + timeout
    spinner = "|/-\\"
    i = 0

    while time.time() < deadline:
        if any(p.poll() is not None for p in processes):
            say("\nThe API process exited. Its output is above.", RED)
            return None
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                print("\r" + " " * 70 + "\r", end="")
                return json.loads(response.read())
        except Exception:
            elapsed = int(timeout - (deadline - time.time()))
            print(f"\r  {spinner[i % 4]} waiting for the API... {elapsed}s", end="", flush=True)
            i += 1
            time.sleep(1)

    print()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    os.chdir(ROOT)

    # Check dependencies before anything else. uvicorn and streamlit are launched
    # as subprocesses, so a missing package surfaces as a child process dying with
    # a traceback in a window that may already have scrolled — much harder to read
    # than saying so up front.
    missing = []
    for module, package in (
        ("dotenv", "python-dotenv"),
        ("logfire", "logfire"),
        ("uvicorn", "uvicorn"),
        ("fastapi", "fastapi"),
        ("streamlit", "streamlit"),
        ("qdrant_client", "qdrant-client"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        activate = r".venv\Scripts\Activate.ps1" if os.name == "nt" else "source .venv/bin/activate"
        say(f"\nDependencies are not installed. Missing: {', '.join(missing)}", RED)
        if sys.prefix == sys.base_prefix:
            say("\n  You are not in a virtual environment:\n", DIM)
            say("    python -m venv .venv", DIM)
            say(f"    {activate}", DIM)
            say("    pip install -r requirements.txt\n", DIM)
        else:
            say("\n    pip install -r requirements.txt\n", DIM)
        say(f"  Interpreter: {sys.executable}\n", DIM)
        sys.exit(1)

    if not os.path.exists(".env"):
        say("\nNo .env file found.", RED)
        say("  cp .env.example .env      (Windows: copy .env.example .env)", DIM)
        say("  then fill in GROQ_API_KEY and set LOCAL_MODE=true\n", DIM)
        sys.exit(1)

    # ── Preflight ─────────────────────────────────────────────────────────────
    if not args.skip_preflight:
        say(f"\n{BOLD}Preflight{RESET}")
        result = subprocess.run([sys.executable, "scripts/preflight.py"])
        if result.returncode != 0:
            say("Fix the issues above, then run this again.\n", RED)
            sys.exit(1)

    # ── API ───────────────────────────────────────────────────────────────────
    say(f"\n{BOLD}Starting API{RESET} {DIM}on :{args.port}{RESET}")
    processes.append(
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", str(args.port)],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    )

    health = wait_for_health(f"http://127.0.0.1:{args.port}/health")
    if health is None:
        say("The API did not become healthy. See the traceback above.", RED)
        shutdown()

    # ── Report what came up ───────────────────────────────────────────────────
    say(f"\n{BOLD}Backends{RESET}")
    for layer, value in health.get("mode", {}).items():
        say(f"  {layer:12} {DIM}{value}{RESET}")

    index = health.get("index", {})
    if index.get("exists") and index.get("vectors"):
        say(f"  {'index':12} {DIM}{index['vectors']} vectors · {index['dimension']}-dim{RESET}")
    else:
        # Worth stopping for. An empty index produces no error — retrieval simply
        # returns nothing, and every answer becomes "the documentation does not
        # cover this", which looks like a broken demo rather than an empty one.
        say(f"\n  {RED}The vector index is empty.{RESET}")
        say(f"  {DIM}Every question will answer 'not covered by the documentation'.{RESET}")
        say(f"\n  Index the demo corpus first (6 files, about a minute):")
        say(f"    {BOLD}python -m app.ingestion.processor DATA/true_data true --wipe{RESET}\n")
        shutdown()

    if health.get("guardrails") != "ready":
        say(f"  {YELLOW}guardrails did not initialise — the gate is not protecting the endpoint{RESET}")

    # ── UI ────────────────────────────────────────────────────────────────────
    if args.api_only:
        say(f"\n{GREEN}API ready{RESET} → http://127.0.0.1:{args.port}/docs")
        say(f"{DIM}Ctrl+C to stop.{RESET}\n")
        processes[0].wait()
        return

    say(f"\n{BOLD}Starting UI{RESET} {DIM}on :{args.ui_port}{RESET}")
    processes.append(
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "ui/app.py",
             "--server.port", str(args.ui_port),
             "--server.headless", "true",
             "--browser.gatherUsageStats", "false"],
            env={
                **os.environ,
                "BACKEND_URL": f"http://127.0.0.1:{args.port}",
                "PYTHONUNBUFFERED": "1",
            },
        )
    )

    time.sleep(3)
    ui_url = f"http://localhost:{args.ui_port}"

    say(f"\n{GREEN}{BOLD}Running.{RESET}")
    say(f"  Chat UI   {ui_url}")
    say(f"  API docs  http://127.0.0.1:{args.port}/docs")
    say(f"  Health    http://127.0.0.1:{args.port}/health")
    say(f"\n{DIM}Ctrl+C to stop both.{RESET}\n")

    if not args.no_browser:
        try:
            webbrowser.open(ui_url)
        except Exception:
            pass

    while all(p.poll() is None for p in processes):
        time.sleep(1)

    say("\nA process exited unexpectedly.", RED)
    shutdown()


if __name__ == "__main__":
    main()
