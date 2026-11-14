import pytest
from fastapi.testclient import TestClient

from recommender.fetch import FetchError, _check_host, html_to_text
from recommender.main import create_app


@pytest.fixture(scope="module")
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_recommend_shape(client):
    body = client.post("/api/recommend", json={"query": "Hiring a data scientist with Python and statistics"}).json()
    assert 1 <= len(body["results"]) <= 10
    first = body["results"][0]
    assert {"name", "url", "minutes", "test_type_names", "reason", "rank"} <= first.keys()
    assert body["pipeline"]["reranked_by"] == "balanced"


def test_overrides_apply(client):
    body = client.post("/api/recommend", json={"query": "Java developer", "overrides": {"max_minutes": 10}}).json()
    assert body["requirements"]["max_minutes"] == 10
    assert all(r["minutes"] is None or r["minutes"] <= 10 for r in body["results"])


def test_assignment_compatible_endpoint(client):
    body = client.post("/recommend", json={"query": "QA tester with Selenium"}).json()
    item = body["recommended_assessments"][0]
    assert set(item) == {"url", "name", "adaptive_support", "description", "duration", "remote_support", "test_type"}
    assert item["remote_support"] in ("Yes", "No")


def test_private_urls_are_refused(client):
    for url in ("http://127.0.0.1:7860/api/health", "http://localhost/", "http://169.254.169.254/latest/meta-data/"):
        response = client.post("/api/recommend", json={"query": url})
        assert response.status_code == 400, url


def test_ssrf_guard_rejects_bad_schemes():
    with pytest.raises(FetchError):
        _check_host("file:///etc/passwd")
    with pytest.raises(FetchError):
        _check_host("http://10.0.0.5/")


def test_html_to_text_drops_scripts_and_navigation():
    html = "<html><nav>Menu</nav><script>evil()</script><h1>Data Analyst</h1><p>SQL and Excel required.</p></html>"
    text = html_to_text(html)
    assert "Data Analyst" in text and "SQL and Excel" in text
    assert "evil" not in text and "Menu" not in text


def test_catalog_stats(client):
    stats = client.get("/api/catalog/stats").json()
    assert stats["assessments"] + stats["reports_excluded"] == 377


def test_validation(client):
    assert client.post("/api/recommend", json={"query": "hi"}).status_code == 422
