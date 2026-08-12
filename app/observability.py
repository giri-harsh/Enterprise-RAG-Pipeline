"""
Single place where Logfire gets configured.

Tracing is optional in this project — DEMO.md says LOGFIRE_TOKEN can stay blank —
but that was only true by accident. Each entry point called logfire.configure()
in its own way, and three of the four got it wrong:

    app/main.py            logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
    app/ingestion/...      logfire.configure(service_name=...)
    evals/app.py           logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

With no token set, `token=` is None and there is no send_to_logfire=False, so
Logfire looks for credentials it cannot find and raises — before a single
document is indexed or a single request served. Only ui/app.py survived, and
only because it happened to wrap the call in try/except.

The fix is one function used everywhere, which decides the mode from whether a
token actually exists rather than from what each caller remembered to pass.
"""

import os

import logfire


def configure_logfire(service_name: str, console: bool = True) -> bool:
    """
    Set up tracing for one process. Returns True if spans are being exported.

    With a token: spans go to Logfire.
    Without one: Logfire runs in local-only mode. Instrumentation still works and
    spans are still created, they just are not shipped anywhere — which is what
    makes the token genuinely optional rather than nominally optional.

    `console=False` silences the local span printout; the default leaves Logfire's
    own behaviour alone. Note that logfire.configure's own `console` parameter
    accepts ConsoleOptions or False, never True — passing True raises
    `'bool' object has no attribute 'span_style'`, so the boolean is translated
    here rather than forwarded.

    Never raises. Observability failing is not a reason for the application to
    fail, and a misconfigured token should not be the thing that stops a demo.
    """
    token = (os.getenv("LOGFIRE_TOKEN") or "").strip()
    console_option = None if console else False

    try:
        if token:
            logfire.configure(token=token, service_name=service_name, console=console_option)
            return True

        logfire.configure(send_to_logfire=False, service_name=service_name, console=console_option)
        return False

    except Exception as exc:  # pragma: no cover - defensive
        # Last resort: try the most minimal configuration possible so that the
        # logfire.span() and logfire.info() calls scattered through the codebase
        # remain valid no-ops rather than AttributeErrors at call time.
        try:
            logfire.configure(send_to_logfire=False, console=False)
        except Exception:
            pass
        print(f"[observability] Logfire setup failed, continuing without tracing: {exc}")
        return False
