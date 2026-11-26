"""Studio HTTP server, security checks, uploads and the command-line entry point."""

import http.client
import json

import numpy as np
import pytest

from medaugmentx import Compose, MedVolume
from medaugmentx.studio import StudioServer, StudioSession, launch
from medaugmentx.transforms import GaussianNoise

TOKEN = "test-token-123"


@pytest.fixture
def server():
    studio = StudioServer(token=TOKEN).start()
    yield studio
    studio.stop()


def request(server, method, path, body=None, *, token=TOKEN, headers=None, raw=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    sent = {"X-Studio-Token": token, **(headers or {})}
    payload = raw
    if body is not None:
        payload = json.dumps(body).encode()
        sent.setdefault("Content-Type", "application/json")
    connection.request(method, path, body=payload, headers=sent)
    response = connection.getresponse()
    data = response.read()
    connection.close()
    kind = response.getheader("Content-Type", "")
    return response.status, (json.loads(data) if "json" in kind else data), response


def test_static_assets_and_security_headers(server):
    for path in ("/", "/static/studio.js", "/static/studio.css", "/static/favicon.svg"):
        status, body, response = request(server, "GET", path, token="")
        assert status == 200 and body
        assert "script-src 'self'" in response.getheader("Content-Security-Policy")
        assert response.getheader("Cache-Control") == "no-store"
        assert response.getheader("X-Content-Type-Options") == "nosniff"
    assert b"Policy Studio" in request(server, "GET", "/")[1]
    assert request(server, "GET", "/nope")[0] == 404
    assert request(server, "HEAD", "/")[0] == 200
    assert "token=" + TOKEN in server.url


def test_api_requires_token_local_host_and_origin(server):
    assert request(server, "GET", "/api/state", token="wrong")[0] == 401
    assert request(server, "POST", "/api/policy", {}, token="")[0] == 401
    evil_host = {"Host": "attacker.example"}
    assert request(server, "GET", "/api/state", headers=evil_host)[0] == 403
    evil_origin = {"Origin": "http://attacker.example"}
    assert (
        request(server, "POST", "/api/load/phantom", {"name": "ct"}, headers=evil_origin)[0] == 403
    )
    local_origin = {"Origin": f"http://localhost:{server.port}"}
    assert (
        request(server, "POST", "/api/load/phantom", {"name": "ct"}, headers=local_origin)[0] == 200
    )
    status, body, _ = request(
        server, "POST", "/api/policy", raw=b"{}", headers={"Content-Type": "text/plain"}
    )
    assert status == 400 and "application/json" in body["error"]
    status, body, _ = request(
        server, "POST", "/api/policy", raw=b"{nope", headers={"Content-Type": "application/json"}
    )
    assert status == 400 and "not valid JSON" in body["error"]
    assert request(server, "POST", "/api/unknown", {})[0] == 400
    assert request(server, "GET", "/api/unknown")[0] == 404
    status, body, _ = request(server, "POST", "/api/policy", ["list"])
    assert status == 400 and "JSON object" in body["error"]


def test_editing_and_preview_workflow(server):
    status, catalog, _ = request(server, "GET", "/api/catalog")
    assert status == 200 and "PerChannel" in {t["name"] for t in catalog["transforms"]}
    assert {p["name"] for p in catalog["presets"]} == {"mri", "ct", "dxr", "dbt"}
    assert request(server, "GET", "/api/state")[1]["loaded"] is False
    assert request(server, "POST", "/api/preview", {})[0] == 400

    state = request(server, "POST", "/api/load/phantom", {"name": "xray"})[1]
    assert state["summary"]["shape"] == [256, 256]
    step = request(server, "POST", "/api/step", {"name": "GaussianNoise"})[1]
    policy = {"name": "Compose", "params": {"transforms": [step]}}
    assert request(server, "POST", "/api/policy", {"policy": policy})[1]["ok"]
    preview = request(server, "POST", "/api/preview", {"seed": 4})[1]
    assert preview["output"]["png"].startswith("data:image/png")
    bad = {"name": "Compose", "params": {"transforms": [{"name": "Nope", "params": {}}]}}
    status, body, _ = request(server, "POST", "/api/preview", {"policy": bad})
    assert status == 400 and body["path"] == [0]
    assert len(request(server, "POST", "/api/gallery", {"count": 2})[1]["items"]) == 2
    assert request(server, "POST", "/api/stress", {"draws": 3})[1]["completed"] == 3
    assert request(server, "POST", "/api/record", {})[1]["record_version"] == 1
    preset = request(server, "POST", "/api/preset", {"name": "dxr"})[1]
    exported = request(server, "POST", "/api/export", {"policy": preset, "format": "json"})[1]
    imported = request(server, "POST", "/api/import", {"text": exported["text"]})[1]
    assert imported["policy"] == preset
    assert server.current_policy().to_dict()["name"] == "Compose"


def test_streamed_uploads_load_images_masks_and_channels(server, tmp_path):
    image = np.random.default_rng(1).random((5, 6)).astype(np.float32)
    for name, array in (("img", image), ("seg", (image > 0.5).astype(np.uint8)), ("t2", image)):
        np.save(tmp_path / f"{name}.npy", array)

    def upload(role, filename, **extra):
        query = "&".join(f"{k}={v}" for k, v in {"role": role, "name": filename, **extra}.items())
        data = (tmp_path / filename).read_bytes()
        return request(
            server,
            "POST",
            f"/api/upload?{query}",
            raw=data,
            headers={"Content-Type": "application/octet-stream"},
        )

    status, body, _ = upload("mask", "seg.npy")
    assert status == 400 and "before its mask" in body["error"]
    assert upload("image", "img.npy")[1]["summary"]["shape"] == [5, 6]
    assert upload("mask", "seg.npy")[1]["has_mask"]
    assert upload("channel", "t2.npy", channel="t2")[1]["channels"] == ["t2"]
    status, body, _ = request(
        server, "POST", "/api/upload?role=image&name=x.npy", raw=b"", headers={}
    )
    assert status == 400 and "empty" in body["error"]
    status, body, _ = request(
        server, "POST", "/api/upload?role=image&name=x.exe", raw=b"abc", headers={}
    )
    assert status == 400 and "Uploads must" in body["error"]
    assert request(server, "POST", "/api/channel/remove", {"name": "t2"})[1]["channels"] == []


def test_server_configuration_guards():
    with pytest.raises(ValueError, match="non-loopback"):
        StudioServer(host="0.0.0.0")
    with pytest.raises(ValueError, match="at least 8"):
        StudioServer(token="short")
    studio = launch(
        Compose([GaussianNoise()]),
        StudioSession().volume,
        open_browser=False,
        block=False,
        token=TOKEN,
    )
    try:
        assert request(studio, "GET", "/api/state")[1]["policy"]["name"] == "Compose"
        assert isinstance(studio.current_policy(), Compose)
    finally:
        studio.stop()
    with StudioServer() as context:
        assert context.url.startswith("http://127.0.0.1:")


def test_cli_parses_and_launches(monkeypatch, tmp_path, capsys):
    from medaugmentx import cli

    captured = {}

    def fake_launch(policy, volume, **kwargs):
        captured.update(policy=policy, volume=volume, **kwargs)

    monkeypatch.setattr("medaugmentx.studio.server.launch", fake_launch)
    policy_path = tmp_path / "policy.json"
    from medaugmentx.serialization import to_json

    policy_path.write_text(to_json(Compose([GaussianNoise()])))
    assert (
        cli.main(["studio", "--phantom", "ct", "--policy", str(policy_path), "--no-browser"]) == 0
    )
    assert captured["volume"].shape == (256, 256)
    assert captured["policy"]["name"] == "Compose" and captured["open_browser"] is False
    assert "CT phantom" in captured["source"]

    np.save(tmp_path / "scan.npy", np.zeros((4, 4), np.float32))
    assert cli.main(["studio", "--image", str(tmp_path / "scan.npy"), "--port", "9"]) == 0
    assert captured["source"] == "scan.npy" and captured["port"] == 9

    assert cli.main(["studio", "--image", str(tmp_path / "missing.npy")]) == 1
    assert "does not exist" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["studio", "--mask", "x.npy"])
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "medaugmentx" in capsys.readouterr().out


def test_rejected_large_bodies_still_receive_the_error(server):
    big = b"x" * (5 * 1024 * 1024)
    status, body, _ = request(
        server, "POST", "/api/upload?role=image&name=x.txt", raw=big, headers={}
    )
    assert status == 400 and "Uploads must" in body["error"]
    status, body, _ = request(
        server, "POST", "/api/policy", raw=big, headers={"Content-Type": "text/plain"}
    )
    assert status == 400 and "application/json" in body["error"]
    status, body, _ = request(server, "POST", "/api/policy", raw=big, token="wrong", headers={})
    assert status == 401
    assert request(server, "GET", "/api/state", headers={"Host": "[::1"})[0] == 403
    assert request(server, "GET", "/api/state")[0] == 200


def test_mask_upload_onto_an_in_memory_volume(tmp_path):
    image = np.random.default_rng(3).random((5, 6)).astype(np.float32)
    np.save(tmp_path / "seg.npy", (image > 0.5).astype(np.uint8))
    studio = launch(volume=MedVolume(image), open_browser=False, block=False, token=TOKEN)
    try:
        status, body, _ = request(
            studio,
            "POST",
            "/api/upload?role=mask&name=seg.npy",
            raw=(tmp_path / "seg.npy").read_bytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert status == 200 and body["has_mask"]
    finally:
        studio.stop()


def test_bind_variants_and_stop_without_start():
    import time

    idle = StudioServer()
    started = time.perf_counter()
    idle.stop()
    assert time.perf_counter() - started < 2

    # Exercise the wildcard Host rule on loopback; binding 0.0.0.0 in tests can
    # trigger desktop firewall prompts. StudioServer maps 0.0.0.0/:: to "*".
    wildcard = StudioServer(token=TOKEN).start()
    try:
        lan = {"Host": f"192.168.1.5:{wildcard.port}"}
        assert request(wildcard, "GET", "/api/state", headers=lan)[0] == 403
        wildcard._httpd.public_host = "*"
        assert request(wildcard, "GET", "/api/state", headers=lan)[0] == 200
    finally:
        wildcard.stop()

    import socket

    if not socket.has_ipv6:
        return
    try:
        ipv6 = StudioServer(host="::1", token=TOKEN).start()
    except OSError:
        return  # IPv6 loopback unavailable on this runner
    try:
        assert ipv6.url.startswith("http://[::1]:")
        connection = http.client.HTTPConnection("::1", ipv6.port, timeout=10)
        connection.request("GET", "/api/state", headers={"X-Studio-Token": TOKEN})
        assert connection.getresponse().status == 200
        connection.close()
    finally:
        ipv6.stop()
