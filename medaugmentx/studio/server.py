"""Local HTTP server for the policy studio (standard library only).

The server binds to the loopback interface by default and requires a random
per-session token on every API call. Host and Origin headers are checked so
other websites open in the same browser cannot drive the API, and no page asset
is fetched from the network, so image data never leaves the machine.
"""

from __future__ import annotations

import hmac
import inspect
import ipaddress
import json
import math
import mimetypes
import os
import secrets
import socket
import socketserver
import threading
import webbrowser
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np

from medaugmentx._version import __version__
from medaugmentx.core.base import Transform
from medaugmentx.core.volume import MedVolume
from medaugmentx.studio.catalog import CATEGORY_ORDER, catalog
from medaugmentx.studio.session import (
    MAX_GALLERY,
    MAX_STRESS_DRAWS,
    PHANTOMS,
    PRESETS,
    PolicyError,
    StudioSession,
)
from medaugmentx.validation import VolumeValidator

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
# Unread bodies of rejected requests are discarded up to this size so the
# client receives the error; larger ones close the connection instead.
MAX_DRAIN_BYTES = 256 * 1024 * 1024
WILDCARD_HOSTS = ("0.0.0.0", "::", "")
TOKEN_HEADER = "X-Studio-Token"
_STATIC = {
    "/": "index.html",
    "/index.html": "index.html",
    "/static/studio.css": "studio.css",
    "/static/studio.js": "studio.js",
    "/static/favicon.svg": "favicon.svg",
}
_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
    "connect-src 'self'; font-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
_CLIENT_ERRORS = (ValueError, TypeError, KeyError, FileNotFoundError, ImportError, IndexError)


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _clean(value: Any) -> Any:
    """JSON-safe copy: numpy scalars/arrays converted, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _presets() -> list[dict[str, Any]]:
    entries = []
    for name, factory in PRESETS.items():
        options = [
            key
            for key, parameter in inspect.signature(factory).parameters.items()
            if parameter.kind == parameter.KEYWORD_ONLY
        ]
        summary = " ".join((inspect.getdoc(factory) or "").split("\n\n", 1)[0].split())
        entries.append({"name": name, "options": options, "summary": summary})
    return entries


class _Handler(BaseHTTPRequestHandler):
    server: _StudioHTTPServer
    protocol_version = "HTTP/1.1"
    server_version = f"MedAugmentXStudio/{__version__}"

    # -- plumbing --------------------------------------------------------
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.server.verbose:
            super().log_message(format, *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        if status >= 400:
            self._discard_body()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        # Assets are tiny and local; never cache so upgrades take effect at once.
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_clean(payload), allow_nan=False, separators=(",", ":")).encode()
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str, **extra: Any) -> None:
        self._json({"error": message, **extra}, status)

    def _discard_body(self) -> None:
        """Consume an unread request body so an error response reaches the client."""
        if getattr(self, "_body_consumed", True):
            return
        self._body_consumed = True
        try:
            remaining = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            remaining = -1
        if remaining < 0 or remaining > MAX_DRAIN_BYTES:
            self.close_connection = True
            return
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1 << 20))
            if not chunk:
                self.close_connection = True
                return
            remaining -= len(chunk)

    def _name_allowed(self, name: str | None) -> bool:
        public = self.server.public_host
        # A wildcard bind serves any interface; the token is the protection.
        return _is_loopback(name or "") or public == "*" or (public is not None and name == public)

    def _host_allowed(self) -> bool:
        try:
            name = urlsplit(f"//{self.headers.get('Host', '')}").hostname
        except ValueError:
            return False
        return self._name_allowed(name)

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        try:
            parts = urlsplit(origin)
            name = parts.hostname
        except ValueError:
            return False
        return parts.scheme == "http" and self._name_allowed(name)

    def _authorised(self) -> bool:
        supplied = self.headers.get(TOKEN_HEADER, "")
        return hmac.compare_digest(supplied.encode(), self.server.token.encode())

    def _read_body(self, limit: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0 or length > limit:
            raise ValueError(f"Request body exceeds the {limit // (1024 * 1024)} MiB limit")
        self._body_consumed = True
        return self.rfile.read(length)

    def _stream_to(self, target: str, limit: int) -> None:
        try:
            remaining = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if remaining <= 0:
            raise ValueError("Upload is empty")
        if remaining > limit:
            raise ValueError(f"Upload exceeds the {limit // (1024**3)} GiB limit")
        self._body_consumed = True
        with open(target, "wb") as handle:
            while remaining:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    self.close_connection = True
                    raise ValueError("Upload ended early")
                handle.write(chunk)
                remaining -= len(chunk)

    # -- verbs -----------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Host not allowed")
            return
        path = urlsplit(self.path).path
        if path in _STATIC:
            asset = files("medaugmentx.studio").joinpath("static").joinpath(_STATIC[path])
            kind = mimetypes.guess_type(_STATIC[path])[0] or "application/octet-stream"
            if kind.startswith("text/") or kind.endswith(("javascript", "svg+xml")):
                kind += "; charset=utf-8"
            self._send(HTTPStatus.OK, asset.read_bytes(), kind)
            return
        if not path.startswith("/api/"):
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if not self._authorised():
            self._error(HTTPStatus.UNAUTHORIZED, "Missing or invalid studio token")
            return
        session = self.server.session
        try:
            if path == "/api/catalog":
                self._json(
                    {
                        "version": __version__,
                        "categories": list(CATEGORY_ORDER),
                        "transforms": catalog(),
                        "phantoms": sorted(PHANTOMS),
                        "presets": _presets(),
                        "validator": VolumeValidator().to_dict(),
                        "limits": {"gallery": MAX_GALLERY, "stress": MAX_STRESS_DRAWS},
                    }
                )
            elif path == "/api/state":
                self._json(session.state())
            else:
                self._error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except _CLIENT_ERRORS as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        self._body_consumed = False
        if not self._host_allowed() or not self._origin_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Host or origin not allowed")
            return
        parts = urlsplit(self.path)
        if not parts.path.startswith("/api/"):
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if not self._authorised():
            self._error(HTTPStatus.UNAUTHORIZED, "Missing or invalid studio token")
            return
        try:
            if parts.path == "/api/upload":
                self._json(self._upload(parse_qs(parts.query)))
                return
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                raise ValueError("Requests must send application/json")
            raw = self._read_body(MAX_JSON_BYTES)
            payload = json.loads(raw or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            self._json(self._dispatch(parts.path, payload))
        except PolicyError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc), path=exc.path)
        except json.JSONDecodeError:
            self._error(HTTPStatus.BAD_REQUEST, "Request body is not valid JSON")
        except _CLIENT_ERRORS as exc:
            message = str(exc).strip("'\"") or type(exc).__name__
            self._error(HTTPStatus.BAD_REQUEST, message)
        except Exception as exc:  # noqa: BLE001 - surface unexpected failures to the UI
            self.log_error("Unhandled studio error: %r", exc)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    # -- routes ----------------------------------------------------------
    def _dispatch(self, path: str, payload: Mapping[str, Any]) -> Any:
        session = self.server.session
        if path == "/api/load/phantom":
            return session.load_phantom(
                str(payload.get("name", "")), demo_targets=bool(payload.get("demo_targets", True))
            )
        if path == "/api/load/path":
            return session.load_path(
                str(payload.get("image", "")),
                payload.get("mask") or None,
                resample_mask=bool(payload.get("resample_mask", False)),
                strict=bool(payload.get("strict", True)),
            )
        if path == "/api/channel/add":
            return session.add_channel(str(payload.get("name", "")), str(payload.get("path", "")))
        if path == "/api/channel/remove":
            return session.remove_channel(str(payload.get("name", "")))
        if path == "/api/policy":
            return session.set_policy(payload.get("policy"))
        if path == "/api/step":
            return session.new_step(str(payload.get("name", "")))
        if path == "/api/preset":
            return session.preset(str(payload.get("name", "")), payload.get("options") or {})
        if path == "/api/preview":
            return session.preview(payload)
        if path == "/api/gallery":
            return session.gallery(payload)
        if path == "/api/stress":
            return session.stress(payload)
        if path == "/api/record":
            return session.record(payload)
        if path == "/api/export":
            return session.export(payload.get("policy", session.policy), str(payload.get("format")))
        if path == "/api/import":
            return session.import_text(payload.get("text", ""))
        raise KeyError("Unknown endpoint")

    def _upload(self, query: Mapping[str, list[str]]) -> Any:
        session = self.server.session
        role = (query.get("role") or [""])[0]
        name = (query.get("name") or [""])[0]
        channel = (query.get("channel") or [""])[0]
        if role == "mask" and session.volume is None:
            raise ValueError("Upload or load an image before its mask")
        stored = session.upload_target(role, name)
        try:
            self._stream_to(stored, MAX_UPLOAD_BYTES)
            if role == "image":
                return session.load_path(stored)
            if role == "mask":
                resample = (query.get("resample_mask") or ["0"])[0] == "1"
                return session.attach_mask(stored, resample_mask=resample)
            return session.add_channel(channel, stored)
        except BaseException:
            # A rejected or partial upload must not linger on disk.
            if os.path.exists(stored) and stored not in session._paths.values():
                os.remove(stored)
            raise


class _StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self) -> None:
        # HTTPServer.server_bind calls socket.getfqdn(), a reverse DNS lookup
        # that can stall for a long time on machines with slow or blocked DNS.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)

    def __init__(
        self,
        address: tuple[str, int],
        session: StudioSession,
        token: str,
        public_host: str | None,
        verbose: bool,
    ) -> None:
        self.session = session
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, _Handler)
        self.token = token
        self.public_host = public_host
        self.verbose = verbose


class StudioServer:
    """A running policy studio. Use :func:`launch` for the common case.

    Args:
        session: State to serve; a fresh :class:`StudioSession` by default.
        host: Interface to bind. Non-loopback hosts expose your images to the
            network and are only accepted with ``allow_remote=True``.
        port: TCP port, or 0 to choose a free one.
        token: API token; a random one is generated by default.
        allow_remote: Permit binding a non-loopback interface.
        verbose: Log every request to stderr.
    """

    def __init__(
        self,
        session: StudioSession | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        token: str | None = None,
        allow_remote: bool = False,
        verbose: bool = False,
    ) -> None:
        if not _is_loopback(host) and not allow_remote:
            raise ValueError(
                "Refusing to serve images on a non-loopback interface; pass allow_remote=True"
            )
        if token is not None and len(token) < 8:
            raise ValueError("token must be at least 8 characters")
        self.session = session or StudioSession()
        self.token = token or secrets.token_urlsafe(24)
        self._httpd = _StudioHTTPServer(
            (host, int(port)),
            self.session,
            self.token,
            None if _is_loopback(host) else ("*" if host in WILDCARD_HOSTS else host),
            verbose,
        )
        self.host = host
        self.port = int(self._httpd.server_address[1])
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Address to open, including the session token."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/?token={self.token}"

    def start(self) -> StudioServer:
        """Serve in a background daemon thread and return immediately."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._httpd.serve_forever, name="medaugmentx-studio", daemon=True
            )
            self._thread.start()
        return self

    def serve_forever(self) -> None:
        """Serve in the calling thread until interrupted or :meth:`stop`."""
        try:
            self._httpd.serve_forever()
        finally:
            self._httpd.server_close()

    def stop(self) -> None:
        """Stop serving and delete uploaded files."""
        if self._thread is not None:
            # shutdown() waits for serve_forever, so only call it while serving.
            self._httpd.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._httpd.server_close()
        self.session.close()

    def current_policy(self) -> Transform:
        """The policy currently in the editor, as a ready-to-use transform."""
        return self.session.current_policy()

    def __enter__(self) -> StudioServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def launch(
    policy: Transform | Mapping[str, Any] | None = None,
    volume: MedVolume | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    block: bool = True,
    token: str | None = None,
    allow_remote: bool = False,
    verbose: bool = False,
    source: str | None = None,
) -> StudioServer:
    """Open the policy studio, optionally starting from a policy and volume.

    With ``block=False`` (handy in notebooks) the server runs in the background;
    call ``server.current_policy()`` to retrieve the edited policy and
    ``server.stop()`` when done. With ``block=True`` it serves until Ctrl+C.

    Args:
        policy: Transform or ``to_dict()`` mapping to start editing.
        volume: In-memory volume to preview; load data from the UI otherwise.
        host: Interface to bind (loopback by default).
        port: TCP port, or 0 for any free port.
        open_browser: Open the studio in the default web browser.
        block: Serve in the foreground until interrupted.
        token: Fixed API token instead of a random one.
        allow_remote: Permit a non-loopback ``host``.
        verbose: Log requests to stderr.
        source: Label shown for ``volume`` (for example its filename).
    """
    session = StudioSession(volume, policy, source=source)
    server = StudioServer(
        session, host=host, port=port, token=token, allow_remote=allow_remote, verbose=verbose
    )
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(server.url,)).start()
    if not block:
        return server.start()
    print(f"MedAugmentX Policy Studio running at {server.url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
    return server


__all__ = ["MAX_JSON_BYTES", "MAX_UPLOAD_BYTES", "StudioServer", "TOKEN_HEADER", "launch"]
