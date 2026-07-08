"""Smoke test for the Elephante AI server (merged Groq/HF/Llama version).

Runs against the real app if torch is installed; otherwise stubs out the
torch/torchvision layer so the API surface (routes, /analyze dual contract,
CORS, graceful fallbacks) can still be verified.

Usage: python3 smoke_test.py
"""
import base64
import io
import sys
import types


def _install_torch_stubs():
    class _Fake(types.ModuleType):
        def __getattr__(self, name):
            return _fake_callable

    def _fake_callable(*args, **kwargs):
        return _FakeObj()

    class _FakeObj:
        def __call__(self, *a, **k):
            return self

        def __getattr__(self, name):
            return _fake_callable

        def __bool__(self):
            return False

    torch = _Fake("torch")
    torch.device = lambda *a, **k: "cpu"
    torch.load = lambda *a, **k: {}
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _Module:
        def __init__(self, *a, **k):
            self.fc = _FakeObj()

        def to(self, *a):
            return self

        def eval(self):
            return self

        def load_state_dict(self, *a, **k):
            pass

    nn = _Fake("torch.nn")
    nn.Module = _Module
    functional = _Fake("torch.nn.functional")
    torch.nn = nn

    torchvision = _Fake("torchvision")
    tv_transforms = _Fake("torchvision.transforms")
    tv_transforms.Compose = lambda x: x
    tv_transforms.Resize = _fake_callable
    tv_transforms.ToTensor = _fake_callable
    tv_transforms.Normalize = _fake_callable
    tv_models = _Fake("torchvision.models")
    tv_models.resnet50 = lambda *a, **k: _Module()
    tv_models.ResNet50_Weights = types.SimpleNamespace(IMAGENET1K_V2=None)
    torchvision.transforms = tv_transforms
    torchvision.models = tv_models

    for name, mod in {
        "torch": torch,
        "torch.nn": nn,
        "torch.nn.functional": functional,
        "torchvision": torchvision,
        "torchvision.transforms": tv_transforms,
        "torchvision.models": tv_models,
    }.items():
        sys.modules[name] = mod


try:
    import torch  # noqa: F401
    STUBBED = False
except ImportError:
    _install_torch_stubs()
    STUBBED = True

from PIL import Image  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

client = TestClient(main.app)
failures = []

CLIENT_FIELDS = (
    "perfect_visual_prompt",
    "cultural_tags",
    "cultural_context",
    "style_origin",
    "occasion_arabic",
    "formality_level",
    "status",
)


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        failures.append(name)


def tiny_png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 30, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


check("app boots", True, f"(torch stubbed={STUBBED})")

r = client.get("/")
check("GET / returns 200", r.status_code == 200, str(r.json()))
r = client.get("/health")
check("GET /health returns 200", r.status_code == 200)
for key in ("encoder", "match_head", "groq_configured", "sam"):
    check(f"health reports {key}", key in r.json())

r = client.options(
    "/analyze",
    headers={
        "Origin": "https://elephante.vercel.app",
        "Access-Control-Request-Method": "POST",
    },
)
check(
    "CORS preflight allowed",
    r.headers.get("access-control-allow-origin") in ("*", "https://elephante.vercel.app"),
    r.headers.get("access-control-allow-origin", "<none>"),
)

# Web-app contract: image_url + metadata (unreachable URL -> heuristic fallback).
web_payload = {
    "image_url": "https://example.invalid/garment.jpg",
    "item_name": "Linen thobe",
    "item_type": "thobe",
    "color": "white",
    "occasion": "eid",
}
r = client.post("/analyze", json=web_payload)
check("POST /analyze (image_url) returns 200", r.status_code == 200, str(r.status_code))
body = r.json()
for field in CLIENT_FIELDS:
    check(f"/analyze has {field}", field in body, repr(body.get(field))[:70])
check("arabic occasion mapped", body.get("occasion_arabic") == "العيد")
check("formality mapped", body.get("formality_level") == "festive")
check(
    "prompt includes item + color",
    "white" in body.get("perfect_visual_prompt", "") and "thobe" in body.get("perfect_visual_prompt", ""),
)

# Base64 contract (no GROQ_API_KEY in test env -> heuristic fallback, still 200).
r = client.post("/analyze", json={"image_base64": tiny_png_b64(), "item_type": "shirt"})
check(
    "POST /analyze (image_base64) returns 200",
    r.status_code == 200,
    f"{r.status_code} source={r.json().get('analysis_source') if r.status_code == 200 else ''}",
)

# SSRF guard degrades gracefully for URL contract.
r = client.post("/analyze", json={**web_payload, "image_url": "http://127.0.0.1/x.png"})
check(
    "/analyze private host -> heuristic 200",
    r.status_code == 200 and r.json().get("analysis_source") == "heuristic",
)

# Validation.
r = client.post("/analyze", json={"item_name": "x"})
check("/analyze with no image -> 400", r.status_code == 400)
r = client.post("/analyze", json={"image_base64": "!!!not-base64!!!"})
check("/analyze bad base64 -> 400", r.status_code == 400)

r = client.get("/sam-status")
check("GET /sam-status returns 200", r.status_code == 200)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("All smoke tests passed.")
