"""
First-time setup for the local demo.

    python scripts/setup.py

Creates a virtual environment, installs requirements, copies .env.example to
.env if no .env exists, and runs preflight. Handles both Unix and Windows.

Flags:
    --no-venv       skip venv creation (use the current interpreter)
    --no-install    skip pip install (dependencies already present)
    --no-preflight  skip the preflight check at the end
    --venv-dir DIR  venv directory name (default: .venv)
"""

import argparse
import os
import subprocess
import sys
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── ANSI colours (disabled on old Windows consoles) ──────────────────────────
GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
)
if os.name == "nt" and not os.getenv("WT_SESSION"):
    GREEN = RED = YELLOW = DIM = BOLD = RESET = ""

OK   = f"{GREEN}✓{RESET}"
FAIL = f"{RED}✗{RESET}"
INFO = f"{DIM}·{RESET}"


def say(msg: str, colour: str = ""):
    print(f"{colour}{msg}{RESET}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, streaming output live."""
    return subprocess.run(cmd, **kwargs)


def venv_python(venv_dir: str) -> str:
    """Return the path to the Python binary inside the venv."""
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def venv_activate_hint(venv_dir: str) -> str:
    """Human-readable activation command for the detected shell."""
    if os.name == "nt":
        return f"{venv_dir}\\Scripts\\activate"
    return f"source {venv_dir}/bin/activate"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-venv",       action="store_true", help="skip venv creation")
    parser.add_argument("--no-install",    action="store_true", help="skip pip install")
    parser.add_argument("--no-preflight",  action="store_true", help="skip preflight check")
    parser.add_argument("--venv-dir",      default=".venv",     help="venv directory (default: .venv)")
    args = parser.parse_args()

    os.chdir(ROOT)
    say(f"\n{BOLD}Enterprise RAG Pipeline — local demo setup{RESET}\n")

    # ── Step 1: Virtual environment ───────────────────────────────────────────
    python_exe = sys.executable   # default: current interpreter

    if not args.no_venv:
        say(f"{BOLD}1. Virtual environment{RESET}")
        venv_path = os.path.join(ROOT, args.venv_dir)

        if os.path.isdir(venv_path):
            say(f"  {OK} {args.venv_dir}/ already exists — reusing it")
        else:
            say(f"  {INFO} creating {args.venv_dir}/ with {sys.executable} …")
            result = run([sys.executable, "-m", "venv", venv_path])
            if result.returncode != 0:
                say(f"  {FAIL} venv creation failed", RED)
                sys.exit(1)
            say(f"  {OK} created {args.venv_dir}/")

        python_exe = venv_python(venv_path)
        if not os.path.isfile(python_exe):
            say(f"  {FAIL} expected Python at {python_exe} but it is missing", RED)
            sys.exit(1)
    else:
        say(f"{BOLD}1. Virtual environment{RESET}")
        say(f"  {INFO} --no-venv: using {python_exe}")

    # ── Step 2: Install dependencies ──────────────────────────────────────────
    say(f"\n{BOLD}2. Dependencies{RESET}")
    req = os.path.join(ROOT, "requirements.txt")

    if not os.path.isfile(req):
        say(f"  {FAIL} requirements.txt not found at {req}", RED)
        sys.exit(1)

    if args.no_install:
        say(f"  {INFO} --no-install: skipping pip")
    else:
        say(f"  {INFO} pip install -r requirements.txt  (this may take several minutes on a fresh venv)")
        say(f"  {DIM}  sentence-transformers pulls PyTorch (~400 MB) on first install{RESET}")
        result = run([python_exe, "-m", "pip", "install", "-r", req])
        if result.returncode != 0:
            say(f"\n  {FAIL} pip install failed — see output above", RED)
            sys.exit(1)
        say(f"  {OK} all packages installed")

    # ── Step 3: .env file ─────────────────────────────────────────────────────
    say(f"\n{BOLD}3. Environment file{RESET}")
    env_path     = os.path.join(ROOT, ".env")
    example_path = os.path.join(ROOT, ".env.example")

    if os.path.isfile(env_path):
        say(f"  {OK} .env already exists — leaving it untouched")
    elif os.path.isfile(example_path):
        shutil.copy(example_path, env_path)
        say(f"  {OK} copied .env.example → .env")
        say(f"\n  {YELLOW}Open .env and set GROQ_API_KEY before continuing.{RESET}")
        say(f"  {DIM}  Get a free key at https://console.groq.com/keys{RESET}")
        say(f"  {DIM}  LOCAL_MODE is already set to true.{RESET}")
        say(f"\n  Once the key is set, re-run this script or go straight to ingestion:")
        say(f"  {BOLD}  python -m app.ingestion.processor DATA/true_data true --wipe{RESET}")
        say(f"  {BOLD}  python scripts/demo.py{RESET}\n")
        sys.exit(0)   # exit cleanly — user must fill in the key first
    else:
        say(f"  {FAIL} neither .env nor .env.example found", RED)
        sys.exit(1)

    # ── Step 4: Corpus quick-check ────────────────────────────────────────────
    say(f"\n{BOLD}4. Corpus{RESET}")
    true_data = os.path.join(ROOT, "DATA", "true_data")
    if os.path.isdir(true_data):
        files = [f for f in os.listdir(true_data) if os.path.isfile(os.path.join(true_data, f))]
        say(f"  {OK} DATA/true_data/ — {len(files)} file(s)")
    else:
        say(f"  {FAIL} DATA/true_data/ not found — the demo corpus is missing", RED)
        say(f"  {DIM}  The folder should be tracked in git; try: git checkout HEAD -- DATA/true_data/{RESET}")

    # ── Step 5: Preflight ─────────────────────────────────────────────────────
    say(f"\n{BOLD}5. Preflight check{RESET}")
    if args.no_preflight:
        say(f"  {INFO} --no-preflight: skipped")
    else:
        result = run([python_exe, "scripts/preflight.py"])
        if result.returncode != 0:
            say(f"\n  {FAIL} Preflight found issues — see above", RED)
            say(f"  {DIM}  Fix them, then run:  python scripts/demo.py{RESET}\n")
            sys.exit(1)

    # ── Done ──────────────────────────────────────────────────────────────────
    say(f"\n{GREEN}{BOLD}Setup complete.{RESET}")
    say(f"\nNext steps:")

    qdrant_local = os.path.join(ROOT, ".qdrant_local")
    if not os.path.isdir(qdrant_local):
        say(f"  {BOLD}1.{RESET} Index the corpus  {DIM}(once — about 1 minute, downloads ~420 MB on first run){RESET}")
        say(f"     python -m app.ingestion.processor DATA/true_data true --wipe")
        say(f"  {BOLD}2.{RESET} Start the demo")
        say(f"     python scripts/demo.py")
    else:
        say(f"  {BOLD}python scripts/demo.py{RESET}  — API on :8000, chat UI on :8501, browser opens")

    if not args.no_venv:
        say(f"\n{DIM}Activate the venv first if you open a new terminal:{RESET}")
        say(f"  {venv_activate_hint(args.venv_dir)}\n")
    else:
        print()


if __name__ == "__main__":
    main()
