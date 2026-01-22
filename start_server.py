"""Start the FastAPI server."""
import sys
import io
import os
from pathlib import Path
import uvicorn

# Fix Unicode encoding for Windows to prevent stdout flush errors
if sys.platform == 'win32':
    # In some Windows shells (and especially when started in the background),
    # sys.stdout.flush() can raise OSError: [Errno 22] Invalid argument.
    #
    # Most reliable fix: redirect stdout/stderr to files on Windows by default.
    # Opt out by setting KEEP_CONSOLE_LOGS=true.
    # Only keep console logs if explicitly requested *and* stdout is actually usable.
    # In Cursor/background contexts on Windows, stdout may report isatty=True but still
    # raise OSError: [Errno 22] on flush during startup/shutdown.
    keep_console_requested = os.getenv("KEEP_CONSOLE_LOGS", "").strip().lower() in ("1", "true", "yes")

    def _stream_usable(stream) -> bool:
        try:
            # Write nothing, then flush: this is enough to detect the Errno 22 case.
            stream.write("")
            stream.flush()
            return True
        except Exception:
            return False

    try:
        is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    except Exception:
        is_tty = False

    stdout_ok = _stream_usable(sys.stdout)
    stderr_ok = _stream_usable(sys.stderr)
    keep_console = keep_console_requested and is_tty and stdout_ok and stderr_ok
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not keep_console:
        # Redirect to log files to avoid Windows stdout flush errors
        stdout_file = open(log_dir / "server_stdout.log", "a", encoding="utf-8", errors="replace", buffering=1)
        stderr_file = open(log_dir / "server_stderr.log", "a", encoding="utf-8", errors="replace", buffering=1)
        sys.stdout = stdout_file
        sys.stderr = stderr_file
    else:
        # Interactive console: keep stdout/stderr, just ensure UTF-8 encoding.
        # Wrap in try-except to handle any flush errors gracefully
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            try:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                # If all else fails, redirect to log file
                log_dir.mkdir(parents=True, exist_ok=True)
                sys.stdout = open(log_dir / "server_stdout.log", "a", encoding="utf-8", errors="replace", buffering=1)
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            try:
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                # If all else fails, redirect to log file
                log_dir.mkdir(parents=True, exist_ok=True)
                sys.stderr = open(log_dir / "server_stderr.log", "a", encoding="utf-8", errors="replace", buffering=1)

if __name__ == "__main__":
    # Suppress stdout flush errors on Windows during shutdown
    import atexit
    def _suppress_flush_error():
        """Suppress OSError on stdout/stderr flush during shutdown."""
        try:
            if hasattr(sys.stdout, 'flush'):
                sys.stdout.flush()
        except (OSError, AttributeError):
            pass
        try:
            if hasattr(sys.stderr, 'flush'):
                sys.stderr.flush()
        except (OSError, AttributeError):
            pass
    atexit.register(_suppress_flush_error)
    
    from src.utils.config import settings
    
    # Get configuration from settings
    host = os.getenv("API_HOST", settings.api_host)
    port = int(os.getenv("API_PORT", settings.api_port))
    reload = os.getenv("ENVIRONMENT", settings.environment).lower() != "production"
    # Uvicorn reload on Windows can spawn multiple processes and lead to confusing behavior
    # (including stale code paths and port binding oddities). Default to no-reload on Windows
    # unless explicitly forced.
    if sys.platform == "win32":
        force_reload = os.getenv("FORCE_RELOAD", "").strip().lower() in ("1", "true", "yes")
        if not force_reload:
            reload = False
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print("=" * 60)
    print("Starting AI Shopping Assistant Server")
    print("=" * 60)
    print(f"Server will be available at: http://{host}:{port}")
    print(f"Environment: {settings.environment}")
    print(f"Reload enabled: {reload}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )

