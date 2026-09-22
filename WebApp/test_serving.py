"""Tests for the serving layer: API keys, the engine's input contract, and
the HTTP surface.

The key tests need nothing but the code. The engine and API tests need the
trained bundles in models/ (or MCH_MODEL_REPO set) and are skipped cleanly
when they are absent, so this still runs on a clean checkout.

    cd WebApp && pytest test_serving.py -q
"""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from core.apikeys import (InvalidKey, issue_key, mask,      # noqa: E402
                          verify_key)
from core.engine import band_for                             # noqa: E402
from core.model_store import is_complete, models_dir         # noqa: E402

SECRET = "unit-test-secret"
HAVE_BUNDLES = is_complete(models_dir())
needs_bundles = pytest.mark.skipif(
    not HAVE_BUNDLES, reason="trained bundles not present in models/")


# ===========================================================================
# 1. API keys are stateless and self-verifying.
# ===========================================================================
class TestApiKeys:

    def test_roundtrip(self):
        key, info = issue_key(SECRET, "ada@example.org", scope="research")
        back = verify_key(SECRET, key, revoked=set())
        assert back.owner == "ada@example.org"
        assert back.scope == "research"
        assert back.key_id == info.key_id
        assert back.expires_at is None

    def test_wrong_secret_is_rejected(self):
        key, _ = issue_key(SECRET, "ada")
        with pytest.raises(InvalidKey) as exc:
            verify_key("another-secret", key, revoked=set())
        assert exc.value.reason == "bad signature"

    def test_tampered_payload_is_rejected(self):
        key, _ = issue_key(SECRET, "ada")
        body, sig = key[len("mch_"):].split(".")
        forged = "mch_" + body[:-2] + "AA" + "." + sig
        with pytest.raises(InvalidKey):
            verify_key(SECRET, forged, revoked=set())

    @pytest.mark.parametrize("bad", ["", "nope", "mch_", "mch_abc", "sk_live_x.y"])
    def test_malformed(self, bad):
        with pytest.raises(InvalidKey):
            verify_key(SECRET, bad, revoked=set())

    def test_expiry(self, monkeypatch):
        key, info = issue_key(SECRET, "ada", ttl_days=1)
        assert info.expires_at is not None
        verify_key(SECRET, key, revoked=set())
        monkeypatch.setattr(time, "time", lambda: info.expires_at + 1)
        with pytest.raises(InvalidKey) as exc:
            verify_key(SECRET, key, revoked=set())
        assert exc.value.reason == "expired"

    def test_revocation_by_key_id(self):
        key, info = issue_key(SECRET, "ada")
        with pytest.raises(InvalidKey) as exc:
            verify_key(SECRET, key, revoked={info.key_id})
        assert exc.value.reason == "revoked"

    def test_keys_are_unique_per_issue(self):
        a, _ = issue_key(SECRET, "ada")
        b, _ = issue_key(SECRET, "ada")
        assert a != b

    def test_separator_in_owner_cannot_break_parsing(self):
        key, _ = issue_key(SECRET, "a|b|c")
        assert verify_key(SECRET, key, revoked=set()).owner == "a/b/c"

    def test_mask_hides_the_middle(self):
        key, _ = issue_key(SECRET, "ada")
        m = mask(key)
        assert m.startswith("mch_") and m.endswith(key[-6:]) and len(m) < len(key)


# ===========================================================================
# 2. The triage band.
# ===========================================================================
class TestBand:

    @pytest.mark.parametrize("n,band", [(0, "ROUTINE"), (1, "MONITOR"),
                                        (2, "PRIORITY"), (4, "PRIORITY")])
    def test_band_for(self, n, band):
        assert band_for(n) == band


# ===========================================================================
# 3. The engine's input contract, on the real bundles.
# ===========================================================================
@pytest.fixture(scope="module")
def engine():
    if not HAVE_BUNDLES:
        pytest.skip("bundles not present")
    from core.engine import CascadeEngine
    return CascadeEngine()


@needs_bundles
class TestEngineContract:

    def test_every_target_loaded(self, engine):
        assert engine.targets == ["facility_delivery", "neonatal_death", "stunted",
                                  "severe_stunting", "underweight_child"]

    def test_single_and_batch_agree(self, engine):
        from core.presets import PRESETS
        records = [p["record"] for p in PRESETS.values()]
        singles = [engine.assess(r) for r in records]
        batch, _ = engine.assess_many(records)
        for s, b in zip(singles, batch):
            for o in s["outcomes"]:
                assert abs(o["probability"] - b["outcomes"][o["target"]]["probability"]) < 1e-4
            assert s["band"] == b["band"]

    def test_omitted_is_defaulted_but_null_is_missing(self, engine):
        r = engine.assess({"v012": 30, "v024": None}, include_record=True)
        assert "v024" in r["inputs"]["provided"]
        assert "v024" not in r["inputs"]["defaulted"]
        assert r["record_used"]["v024"] is None
        assert "v190" in r["inputs"]["defaulted"]

    def test_unknown_category_becomes_missing_with_warning(self, engine):
        r = engine.assess({"v024": "Mars"}, include_record=True)
        assert r["record_used"]["v024"] is None
        assert any(w.startswith("v024") for w in r["inputs"]["warnings"])

    def test_unknown_field_is_ignored_not_fatal(self, engine):
        r = engine.assess({"favourite_colour": "teal"})
        assert r["inputs"]["ignored"] == ["favourite_colour"]

    def test_bmi_is_derived_when_absent(self, engine):
        r = engine.assess({"height_cm": 150, "weight_kg": 45}, include_record=True)
        assert r["record_used"]["bmi"] == 20.0
        assert r["record_used"]["v013"] == "20-24"      # from the template age

    def test_case_and_whitespace_are_normalised(self, engine):
        r = engine.assess({"v190": " RICHEST "}, include_record=True)
        assert r["record_used"]["v190"] == "richest"
        assert not r["inputs"]["warnings"]

    def test_facility_delivery_never_drives_the_band(self, engine):
        r = engine.assess({"v190": "richest", "v025": "urban", "m14": 8})
        fd = next(o for o in r["outcomes"] if o["target"] == "facility_delivery")
        adverse_flags = sum(o["flagged"] for o in r["outcomes"] if o["kind"] == "adverse")
        assert fd["kind"] == "pathway"
        assert r["n_flagged"] == adverse_flags

    def test_response_is_json_serialisable(self, engine):
        import json
        json.dumps(engine.assess({"v012": 22}, include_record=True))

    def test_schema_exposes_vocabulary_from_the_encoders(self, engine):
        s = engine.schema_payload()
        v024 = next(f for f in s["fields"] if f["name"] == "v024")
        assert set(v024["allowed"]) >= {"dhaka", "sylhet", "khulna"}
        assert "__missing__" not in v024["allowed"]


# ===========================================================================
# 4. The HTTP surface.
# ===========================================================================
@pytest.fixture(scope="module")
def client():
    if not HAVE_BUNDLES:
        pytest.skip("bundles not present")
    pytest.importorskip("fastapi")
    os.environ["MCH_API_SECRET"] = SECRET
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


@needs_bundles
class TestApi:

    def test_health_is_open(self, client):
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"

    def test_predict_requires_a_key(self, client):
        r = client.post("/v1/predict", json={"record": {"v012": 20}})
        assert r.status_code == 401
        assert "error" in r.json()

    def test_predict_with_a_dashboard_issued_key(self, client):
        key, info = issue_key(SECRET, "tester")
        r = client.post("/v1/predict", json={"record": {"v012": 20}},
                        headers={"X-API-Key": key})
        assert r.status_code == 200
        body = r.json()
        assert body["band"] in {"ROUTINE", "MONITOR", "PRIORITY"}
        assert len(body["outcomes"]) == 5
        assert body["meta"]["key_id"] == info.key_id
        assert r.headers["X-RateLimit-Limit"]

    def test_bearer_header_also_works(self, client):
        key, _ = issue_key(SECRET, "tester")
        r = client.get("/v1/keys/verify", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200 and r.json()["valid"] is True

    def test_batch_limit(self, client):
        from api.main import MAX_BATCH
        key, _ = issue_key(SECRET, "tester")
        r = client.post("/v1/predict/batch",
                        json={"records": [{"v012": 20}] * (MAX_BATCH + 1)},
                        headers={"X-API-Key": key})
        assert r.status_code == 413

    def test_validation_error_is_422(self, client):
        key, _ = issue_key(SECRET, "tester")
        r = client.post("/v1/predict", json={"record": "not-an-object"},
                        headers={"X-API-Key": key})
        assert r.status_code == 422


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
