from main import create_app
from src.query_expansion.models import (
    GroundedExpansionCandidate,
    GroundingStatus,
    QueryExpansionResponse,
)


class FakeQueryExpansionService:
    last_request = None

    def __init__(self, generator=None):
        self.generator = generator

    def expand(self, request):
        self.__class__.last_request = request
        return QueryExpansionResponse(
            term=request.term,
            language=request.language,
            expansions={
                "synonym": [
                    GroundedExpansionCandidate(
                        term="heart attack",
                        category="synonym",
                        status=GroundingStatus.LLM_ONLY,
                        confidence=0.0,
                    )
                ]
            },
        )


def test_query_expansion_profiles_route_lists_profiles(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().get("/query-expansion/profiles")

    assert response.status_code == 200
    names = {profile["name"] for profile in response.json["profiles"]}
    assert {"en", "de"}.issubset(names)
    english = next(
        profile for profile in response.json["profiles"] if profile["name"] == "en"
    )
    assert "synonym" in {category["id"] for category in english["categories"]}
    assert english["default_categories"]


def test_query_expansion_profile_route_returns_profile_metadata(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().get("/query-expansion/profiles/de")

    assert response.status_code == 200
    assert response.json["name"] == "de"
    assert response.json["language_name"] == "Deutsch"
    assert "symptom" in {category["id"] for category in response.json["categories"]}


def test_query_expansion_profile_route_404s_for_unknown_profile(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().get("/query-expansion/profiles/unknown")

    assert response.status_code == 404
    assert "Unknown query-expansion domain profile" in response.json["error"]


def test_query_expansion_route_returns_service_response(monkeypatch, tmp_path):
    import src.api.routes.query_expansion as query_expansion_routes

    FakeQueryExpansionService.last_request = None
    monkeypatch.setattr(
        query_expansion_routes, "QueryExpansionService", FakeQueryExpansionService
    )

    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().post(
        "/query-expansion",
        json={
            "term": "myocardial infarction",
            "language": "en",
            "categories": ["synonym"],
            "llm": {"model": "test-model"},
        },
    )

    assert response.status_code == 200
    assert response.json["term"] == "myocardial infarction"
    assert response.json["expansions"]["synonym"][0]["term"] == "heart attack"
    assert FakeQueryExpansionService.last_request.term == "myocardial infarction"


def test_query_expansion_route_uses_bearer_token_as_llm_api_key(monkeypatch, tmp_path):
    import src.api.routes.query_expansion as query_expansion_routes

    FakeQueryExpansionService.last_request = None
    monkeypatch.setattr(
        query_expansion_routes, "QueryExpansionService", FakeQueryExpansionService
    )

    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().post(
        "/query-expansion",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "term": "myocardial infarction",
            "categories": ["synonym"],
            "llm": {
                "model": "alias-fast",
                "options": {"provider": "blablador"},
            },
        },
    )

    assert response.status_code == 200
    assert (
        FakeQueryExpansionService.last_request.llm.options["api_key"] == "secret-token"
    )


def test_query_expansion_route_accepts_proxy_friendly_api_key_header(
    monkeypatch, tmp_path
):
    import src.api.routes.query_expansion as query_expansion_routes

    FakeQueryExpansionService.last_request = None
    monkeypatch.setattr(
        query_expansion_routes, "QueryExpansionService", FakeQueryExpansionService
    )

    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().post(
        "/query-expansion",
        headers={"X-LLM-API-Key": "secret-token"},
        json={
            "term": "myocardial infarction",
            "categories": ["synonym"],
            "llm": {
                "model": "alias-fast",
                "options": {"provider": "blablador"},
            },
        },
    )

    assert response.status_code == 200
    assert (
        FakeQueryExpansionService.last_request.llm.options["api_key"] == "secret-token"
    )


def test_query_expansion_route_validates_request_body(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().post("/query-expansion", json={"language": "en"})

    assert response.status_code == 400
    assert response.json["error"] == "Invalid query-expansion request."
