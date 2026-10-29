"""Consistency checks for the Prometheus/Grafana configuration and compose.yaml.

The alert rules themselves are unit-tested with promtool
(monitoring/prometheus/tests/deepvision_rules_test.yml, run in CI).
"""

import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from conftest import image_bytes, make_checkpoint
from fastapi.testclient import TestClient

from deepvision.api.app import create_app
from deepvision.api.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
MONITORING = ROOT / "monitoring"
DASHBOARDS = sorted((MONITORING / "grafana" / "dashboards").glob("*.json"))
RULES = MONITORING / "prometheus" / "rules" / "deepvision.yml"
# Metric names in PromQL: deepvision_*, process_*, recording rules (deepvision:*), and up.
METRIC = re.compile(r"\b(deepvision_[a-z0-9_]+|process_[a-z0-9_]+|deepvision:[a-z0-9_:]+|up)\b")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def compose() -> dict:
    return load_yaml(ROOT / "compose.yaml")


def dashboard_exprs() -> list[tuple[str, str]]:
    exprs = []
    for path in DASHBOARDS:
        for panel in json.loads(path.read_text(encoding="utf-8"))["panels"]:
            exprs += [(panel["title"], t["expr"]) for t in panel.get("targets", [])]
    return exprs


def rule_exprs() -> list[tuple[str, str]]:
    rules = [rule for group in load_yaml(RULES)["groups"] for rule in group["rules"]]
    return [(rule.get("alert") or rule["record"], rule["expr"]) for rule in rules]


@pytest.fixture(scope="module")
def exposition(tmp_path_factory) -> str:
    """/metrics output after one successful and one failed prediction."""
    tmp = tmp_path_factory.mktemp("monitoring")
    settings = Settings(checkpoint=make_checkpoint(tmp / "best.pt"), evaluation_metrics=tmp / "m")
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        client.post("/predict", files={"file": ("a.png", image_bytes(), "image/png")})
        client.post("/predict", files={"file": ("a.png", b"bad", "image/png")})
        return client.get("/metrics").text


# --- Prometheus -------------------------------------------------------------------------


def test_prometheus_scrapes_the_api_service():
    config = load_yaml(MONITORING / "prometheus" / "prometheus.yml")
    [job] = [j for j in config["scrape_configs"] if j["job_name"] == "deepvision-api"]
    assert job["metrics_path"] == "/metrics"
    assert job["static_configs"][0]["targets"] == ["api:8000"]
    assert "api" in compose()["services"]
    assert config["rule_files"] == ["/etc/prometheus/rules/*.yml"]


def test_alerts_cover_availability_errors_and_latency():
    alerts = {name: expr for name, expr in rule_exprs() if not name.startswith("deepvision:")}
    assert set(alerts) == {
        "DeepVisionApiDown", "DeepVisionModelNotReady", "DeepVisionHighErrorRate",
        "DeepVisionHighLatency",
    }  # fmt: skip
    assert 'up{job="deepvision-api"}' in alerts["DeepVisionApiDown"]
    assert "deepvision_model_ready" in alerts["DeepVisionModelNotReady"]


def test_error_ratio_ignores_operational_endpoints():
    [ratio] = [e for name, e in rule_exprs() if name == "deepvision:request_errors:ratio_rate5m"]
    assert ratio.count('endpoint!~"/metrics|/health|/ready"') == 2


# --- Grafana ----------------------------------------------------------------------------


def test_grafana_datasource_points_at_prometheus():
    config = load_yaml(MONITORING / "grafana" / "provisioning" / "datasources" / "prometheus.yml")
    [source] = config["datasources"]
    assert source["type"] == "prometheus" and source["uid"] == "prometheus"
    assert source["url"] == "http://prometheus:9090"


def test_dashboard_provider_path_matches_the_compose_mount():
    provider = load_yaml(MONITORING / "grafana" / "provisioning" / "dashboards" / "deepvision.yml")
    path = provider["providers"][0]["options"]["path"]
    targets = [v["target"] for v in compose()["services"]["grafana"]["volumes"] if "target" in v]
    assert path in targets


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_dashboard_is_well_formed(path):
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    panels = dashboard["panels"]
    ids = [p["id"] for p in panels]
    assert dashboard["uid"] and len(ids) == len(set(ids))
    for panel in panels:
        if panel["type"] == "row":
            continue
        assert panel["title"] and panel["targets"], panel
        assert panel["datasource"]["uid"] == "prometheus"
        assert all(t["datasource"]["uid"] == "prometheus" and t["expr"] for t in panel["targets"])
        # Panels must not overlap on the 24-column grid.
        assert panel["gridPos"]["x"] + panel["gridPos"]["w"] <= 24


def test_dashboard_covers_performance_errors_latency_and_availability():
    titles = " | ".join(title for title, _ in dashboard_exprs()).lower()
    for topic in ("requests per second", "error", "latency", "model availability", "memory"):
        assert topic in titles, topic


# --- Metric names: dashboards and rules must use metrics the API really exports --------------


def test_dashboards_and_rules_only_use_exported_metrics(exposition):
    recorded = {name for name, _ in rule_exprs() if name.startswith("deepvision:")}
    used = {
        (where, name)
        for where, expr in dashboard_exprs() + rule_exprs()
        for name in METRIC.findall(expr)
    }
    assert used
    missing = []
    for where, name in sorted(used):
        if name == "up" or name in recorded:
            continue  # produced by Prometheus itself
        if name.startswith("process_") and sys.platform != "linux":
            continue  # process metrics are only collected on Linux (checked in CI)
        if not re.search(rf"^{re.escape(name)}[{{ ]", exposition, re.MULTILINE):
            missing.append(f"{where}: {name}")
    assert missing == []


def test_rule_expressions_reference_recording_rules_that_exist():
    recorded = {name for name, _ in rule_exprs() if name.startswith("deepvision:")}
    used = {n for _, e in dashboard_exprs() + rule_exprs() for n in METRIC.findall(e) if ":" in n}
    assert used <= recorded


# --- compose.yaml hardening ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["api", "prometheus", "grafana"])
def test_services_are_hardened_and_localhost_only(name):
    service = compose()["services"][name]
    assert service["ports"] and all(p.startswith("127.0.0.1:") for p in service["ports"])
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["mem_limit"] and service["pids_limit"]
    for volume in service["volumes"]:
        if isinstance(volume, dict) and volume["type"] == "bind":
            assert volume["read_only"] is True, volume
            assert (ROOT / volume["source"]).exists() or "checkpoints" in volume["source"]


@pytest.mark.parametrize("name", ["prometheus", "grafana"])
def test_monitoring_images_are_pinned(name):
    image = compose()["services"][name]["image"]
    assert re.fullmatch(r"[a-z/]+:v?\d+\.\d+\.\d+@sha256:[0-9a-f]{64}", image), image


def test_grafana_is_isolated_from_the_api():
    services = compose()["services"]
    assert set(services["grafana"]["networks"]).isdisjoint(services["api"]["networks"])
    assert set(services["prometheus"]["networks"]) >= set(services["api"]["networks"])


def test_grafana_is_offline_and_read_only_for_anonymous_users():
    env = compose()["services"]["grafana"]["environment"]
    assert env["GF_AUTH_ANONYMOUS_ORG_ROLE"] == "Viewer"
    assert env["GF_USERS_ALLOW_SIGN_UP"] == "false"
    for key in ("GF_ANALYTICS_REPORTING_ENABLED", "GF_ANALYTICS_CHECK_FOR_UPDATES"):
        assert env[key] == "false"
    assert env["GF_PLUGINS_PREINSTALL_DISABLED"] == "true"
