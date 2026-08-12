"""
Preflight check — run this before anything else.

Answers, in one pass: which keys are set, which services actually respond, and
whether the vector collection exists with the right dimensions. Every check is
independent and none of them raise, so one failure does not hide the rest.

    python scripts/preflight.py

Exit code 0 means the demo will start. Non-zero lists what is blocking it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _check_interpreter():
    """
    Verify the Python running this script is one the project supports.

    Two failure modes this catches, both of which otherwise surface as a wall of
    compiler errors rather than anything resembling the real problem:

      Too new — on a just-released Python, most packages have no prebuilt wheels,
      so pip falls back to building from source and needs CMake and a C++
      toolchain. pyarrow (via streamlit) is usually the first to fail.

      Wrong interpreter — if a virtualenv was created without pip, `pip install`
      silently resolves to the system pip and installs somewhere else entirely.
      The venv stays empty and nothing says so.
    """
    major, minor = sys.version_info[:2]

    if (major, minor) < (3, 10):
        print(f"\n  Python {major}.{minor} is too old — this project needs 3.10 to 3.13.\n")
        sys.exit(1)

    if (major, minor) >= (3, 14):
        print(f"\n  Python {major}.{minor} is newer than this project's dependencies support.\n")
        print("  Packages like pyarrow have no prebuilt wheels for it yet, so pip tries")
        print("  to compile them from source and fails without CMake and Visual Studio.\n")
        print("  Use Python 3.12 or 3.13:\n")
        print(r"    py -3.12 -m venv .venv")
        print(r"    .venv\Scripts\Activate.ps1" if os.name == "nt" else "    source .venv/bin/activate")
        print("    python -m pip install -r requirements-demo.txt\n")
        print(f"  Current interpreter: {sys.executable}\n")
        sys.exit(1)


def _require_dependencies():
    """
    Check the imports this script needs before using any of them.

    Without this the first missing package raises a bare ModuleNotFoundError,
    which says nothing about the actual problem — that dependencies were never
    installed, or that the shell is not in the virtualenv where they were. The
    whole point of a preflight script is to explain what is wrong.
    """
    required = {
        "dotenv": "python-dotenv",
        "logfire": "logfire",
        "qdrant_client": "qdrant-client",
        "openai": "openai",
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if not missing:
        return

    in_venv = sys.prefix != sys.base_prefix
    activate = (
        r".venv\Scripts\Activate.ps1" if os.name == "nt" else "source .venv/bin/activate"
    )

    # A venv without its own pip is the trap worth naming explicitly: `pip install`
    # then resolves to the system pip, installs into a different interpreter, and
    # reports success while this venv stays empty.
    pip_dir = os.path.join(sys.prefix, "Scripts" if os.name == "nt" else "bin")
    has_pip = any(f.startswith("pip") for f in os.listdir(pip_dir)) if os.path.isdir(pip_dir) else False

    print("\n  Dependencies are not installed.\n")
    print(f"  Missing: {', '.join(missing)}\n")

    if not in_venv:
        print("  You are not in a virtual environment. Create and activate one:\n")
        print(f"    py -3.12 -m venv .venv" if os.name == "nt" else "    python3 -m venv .venv")
        print(f"    {activate}")
        print(f"    python -m pip install -r requirements-demo.txt\n")
    elif not has_pip:
        print(f"  This virtualenv has no pip of its own ({sys.prefix}).")
        print("  `pip install` would silently install into your system Python instead.")
        print("  Recreate it:\n")
        print(r"    Remove-Item -Recurse -Force .venv" if os.name == "nt" else "    rm -rf .venv")
        print(f"    py -3.12 -m venv .venv" if os.name == "nt" else "    python3 -m venv .venv")
        print(f"    {activate}")
        print(f"    python -m pip install -r requirements-demo.txt\n")
    else:
        print(f"  Virtualenv active ({sys.prefix}), but packages are missing:\n")
        print("    python -m pip install -r requirements-demo.txt\n")

    print(f"  Interpreter: {sys.executable}")
    print("  Tip: always use `python -m pip`, never bare `pip` — it guarantees")
    print("  the package lands in the interpreter you think it does.\n")
    sys.exit(1)


_check_interpreter()
_require_dependencies()

from dotenv import load_dotenv

load_dotenv()

import logfire

logfire.configure(send_to_logfire=False, console=False)

from app.config import settings


GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
)
if os.name == "nt" and not os.getenv("WT_SESSION"):
    # Old Windows consoles render ANSI codes as literal text.
    GREEN = RED = YELLOW = DIM = BOLD = RESET = ""

OK, FAIL, WARN = f"{GREEN}✓{RESET}", f"{RED}✗{RESET}", f"{YELLOW}!{RESET}"

blockers: list[str] = []
warnings: list[str] = []


def header(text):
    print(f"\n{BOLD}{text}{RESET}")


def line(status, label, detail=""):
    print(f"  {status} {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


# ── Mode ──────────────────────────────────────────────────────────────────────

header("Run mode")
if settings.LOCAL_MODE:
    print(f"  {DIM}LOCAL_MODE=true — embedded Qdrant, local embeddings, direct Groq{RESET}")
else:
    print(f"  {DIM}Cloud mode — Qdrant Cloud, Gemini embeddings, Portkey gateway{RESET}")
for layer, value in settings.describe_mode().items():
    line(" ", f"{layer:12}", value)


# ── Groq: the one key that is always required ─────────────────────────────────

header("Groq  (guardrails + generation)")
if not settings.GROQ_API_KEY:
    line(FAIL, "GROQ_API_KEY", "not set")
    blockers.append("GROQ_API_KEY is required in every mode. Get one free at console.groq.com/keys")
else:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        models = {m.id for m in client.models.list().data}
        line(OK, "GROQ_API_KEY", f"valid · {len(models)} models available")

        for label, name in (("generation", settings.GROQ_MODEL), ("guardrail", settings.GROQ_GUARD_MODEL)):
            if name in models:
                line(OK, f"  {label} model", name)
            else:
                line(FAIL, f"  {label} model", f"{name} not available on this account")
                blockers.append(
                    f"Model '{name}' is not available to your Groq account. "
                    f"Set {'GROQ_MODEL' if label == 'generation' else 'GROQ_GUARD_MODEL'} "
                    "to one that is — run this script again to see the list."
                )
    except Exception as exc:
        line(FAIL, "GROQ_API_KEY", f"rejected — {str(exc)[:90]}")
        blockers.append("Groq rejected the API key. Check it at console.groq.com/keys")


# ── LLM routing ───────────────────────────────────────────────────────────────

header("LLM routing")
if settings.USE_GATEWAY:
    if not settings.PORTKEY_API_KEY:
        line(FAIL, "PORTKEY_API_KEY", "not set")
        blockers.append(
            "PORTKEY_API_KEY is required in gateway mode. "
            "Either set it, or run the demo with LOCAL_MODE=true to call Groq directly."
        )
    else:
        line(OK, "PORTKEY_API_KEY", "set")
        line(WARN, "virtual keys", f"@{settings.GROQ_SLUG} and @{settings.GROQ_SLUG_2} must exist in your Portkey dashboard")
        warnings.append(
            f"Portkey virtual keys '{settings.GROQ_SLUG}' and '{settings.GROQ_SLUG_2}' cannot be "
            "verified from here. If generation fails with a routing error, that is why."
        )
else:
    line(OK, "direct Groq", "gateway bypassed — no Portkey account needed")
    warnings.append(
        "Direct mode: the 70B→8B fallback, response cache and retry policy are gateway "
        "features and are inactive. Everything else behaves identically."
    )


# ── Embeddings ────────────────────────────────────────────────────────────────

header("Embeddings")
if settings.USE_LOCAL_EMBEDDINGS:
    try:
        import sentence_transformers  # noqa: F401

        line(OK, "sentence-transformers", f"installed · {settings.LOCAL_EMBED_MODEL} · 768-dim")
        line(WARN, "first run", "downloads ~420 MB of model weights, once")
    except ImportError:
        line(FAIL, "sentence-transformers", "not installed")
        blockers.append("pip install sentence-transformers")
else:
    if not settings.GEMINI_API_KEY:
        line(FAIL, "GEMINI_API_KEY", "not set")
        blockers.append(
            "GEMINI_API_KEY is required in cloud mode. Get one free at aistudio.google.com/apikey, "
            "or run with LOCAL_MODE=true to embed locally instead."
        )
    else:
        try:
            from app.services.retrieval.embedding import embed_query

            vector = embed_query("preflight probe")
            line(OK, "Gemini", f"{settings.GEMINI_EMBED_MODEL} · {len(vector)}-dim")
            if len(vector) != 3072:
                line(WARN, "dimension", f"expected 3072, got {len(vector)}")
                warnings.append(
                    f"Gemini returned {len(vector)}-dim vectors, not the documented 3072. "
                    "The README and DOCS claim 3072 — one of them is wrong."
                )
        except Exception as exc:
            line(FAIL, "Gemini", str(exc).split("\n")[0][:90])
            blockers.append(
                "Gemini embeddings failed. Check GEMINI_API_KEY, or set LOCAL_MODE=true "
                "to embed locally (requires re-ingesting with --wipe: 3072-dim → 768-dim)."
            )


# ── Vector store ──────────────────────────────────────────────────────────────

header("Vector store")
if settings.USE_LOCAL_QDRANT:
    line(OK, "embedded Qdrant", settings.QDRANT_LOCAL_PATH)
elif not settings.QDRANT_URL:
    line(FAIL, "QDRANT_CLUSTER_ENDPOINT", "not set")
    blockers.append(
        "QDRANT_CLUSTER_ENDPOINT is required in cloud mode. Create a free cluster at "
        "cloud.qdrant.io, or run with LOCAL_MODE=true for an embedded store."
    )
else:
    line(" ", "endpoint", settings.QDRANT_URL)

if not (settings.USE_LOCAL_QDRANT or settings.QDRANT_URL):
    line(" ", "collection", "skipped — no endpoint")
else:
    try:
        from app.services.retrieval.qdrant_service import collection_stats

        stats = collection_stats()

        if stats.get("error"):
            line(FAIL, "connection", str(stats["error"])[:90])
            blockers.append(f"Cannot reach the vector store: {str(stats['error'])[:120]}")
        elif not stats["exists"]:
            line(FAIL, "collection", f"'{settings.QDRANT_COLLECTION}' does not exist")
            blockers.append(
                "The collection has not been created. Index the corpus:\n"
                "      python -m app.ingestion.processor DATA/true_data true --wipe"
            )
        elif not stats["vectors"]:
            line(FAIL, "collection", f"'{stats['collection']}' exists but is empty")
            blockers.append(
                "The collection exists with 0 vectors — retrieval will return nothing.\n"
                "      python -m app.ingestion.processor DATA/true_data true --wipe"
            )
        else:
            line(OK, "collection", f"{stats['collection']} · {stats['vectors']} vectors · {stats['dimension']}-dim")

            # The failure that produces the most confusing symptoms: the index was
            # built with one embedding model and is being queried with another.
            # Every search fails, and the error message says nothing about why.
            try:
                from app.services.retrieval.embedding import get_embedding_dim

                expected = get_embedding_dim()
                if expected != stats["dimension"]:
                    line(FAIL, "dimension", f"index is {stats['dimension']}-dim, active model is {expected}-dim")
                    blockers.append(
                        f"Dimension mismatch: the collection was indexed at {stats['dimension']} dims "
                        f"but the active embedding model produces {expected}. Every search will fail.\n"
                        "      Re-index:  python -m app.ingestion.processor DATA/true_data true --wipe"
                    )
                else:
                    line(OK, "dimension", f"matches the active embedding model ({expected})")
            except Exception:
                pass  # already reported under Embeddings
    except Exception as exc:
        line(FAIL, "vector store", str(exc).split("\n")[0][:90])
        blockers.append(f"Vector store check failed: {str(exc)[:120]}")


# ── Optional services ─────────────────────────────────────────────────────────

header("Optional")
line(OK if settings.LOGFIRE_TOKEN else WARN, "LOGFIRE_TOKEN",
     "tracing enabled" if settings.LOGFIRE_TOKEN else "not set — traces stay local, app runs fine")
line(OK if settings.LANGSMITH_API_KEY else WARN, "LANGSMITH_API_KEY",
     "step tracing enabled" if settings.LANGSMITH_API_KEY else "not set — optional")
line(OK if os.getenv("JUDGE_GROQ") else WARN, "JUDGE_GROQ",
     "separate eval judge key" if os.getenv("JUDGE_GROQ") else "not set — evals will use GROQ_API_KEY and compete for its rate limit")
line(OK if settings.API_KEY else WARN, "API_KEY",
     "endpoint auth enabled" if settings.API_KEY else "not set — fine locally, required before deploying")


# ── Verdict ───────────────────────────────────────────────────────────────────

# Release the embedded Qdrant lock before exiting. Left to __del__ it runs during
# interpreter shutdown and prints an ImportError traceback that looks alarming and
# means nothing.
try:
    from app.services.retrieval import qdrant_service

    if qdrant_service._client is not None:
        qdrant_service._client.close()
        qdrant_service._client = None
except Exception:
    pass

print()
if warnings:
    header("Warnings")
    for w in warnings:
        print(f"  {YELLOW}!{RESET} {w}")

if blockers:
    header(f"{RED}Blocked — {len(blockers)} issue(s){RESET}")
    for i, b in enumerate(blockers, 1):
        print(f"  {RED}{i}.{RESET} {b}")
    print()
    sys.exit(1)

print(f"\n{GREEN}{BOLD}Ready.{RESET}  Start the demo with:  {BOLD}python scripts/demo.py{RESET}\n")
