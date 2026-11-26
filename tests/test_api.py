"""Integration tests for FastAPI REST endpoints and web UI."""

from fastapi.testclient import TestClient

from openk.api.app import create_app

client = TestClient(create_app())


def test_home_and_health():
    r = client.get("/")
    assert r.status_code == 200
    assert "OpenK Data Studio" in r.text

    h = client.get("/api/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"


def test_upload_cluster_merge_undo_export_flow():
    csv_bytes = b"city,score\nNew York,10\nnew york,20\nYork, New,30\nBoston,40\n"
    # Note: quote "York, New" properly
    csv_bytes = b'city,score\n"New York",10\n"new york",20\n"York, New",30\n"Boston",40\n'
    up = client.post(
        "/api/sessions/upload",
        files={"file": ("cities.csv", csv_bytes, "text/csv")},
    )
    assert up.status_code == 200
    data = up.json()
    sid = data["session_id"]
    assert data["total_rows"] == 4

    rec = client.get(f"/api/sessions/{sid}/recommend?column=city")
    assert rec.status_code == 200

    cl = client.post(
        f"/api/sessions/{sid}/cluster",
        json={"column": "city", "method": "fingerprint"},
    )
    assert cl.status_code == 200
    clusters = cl.json()["clusters"]
    assert len(clusters) == 1

    merge = client.post(
        f"/api/sessions/{sid}/merge",
        json={
            "column": "city",
            "merges": [
                {
                    "from_values": [m["value"] for m in clusters[0]["members"]],
                    "to_value": "New York",
                }
            ],
        },
    )
    assert merge.status_code == 200
    assert merge.json()["operation"]["rows_affected"] >= 2

    undo = client.post(f"/api/sessions/{sid}/undo")
    assert undo.status_code == 200
    assert undo.json()["undone"] is True

    exp = client.get(f"/api/sessions/{sid}/export?fmt=csv")
    assert exp.status_code == 200
    assert "Boston" in exp.text


def test_samples_and_benchmarks_endpoints():
    samples = client.get("/api/samples")
    assert samples.status_code == 200
    assert "samples" in samples.json()

    bench = client.get("/api/benchmarks")
    assert bench.status_code == 200
    assert "runs" in bench.json()
