from main import create_app
from src.query_expansion.models import (
    ExpansionConcept,
    ExpansionSemanticRelation,
    GroundedExpansionCandidate,
    GroundingStatus,
    QueryExpansionResponse,
)
from src.query_expansion.relations import MEDICAL_RELATION_DEFINITIONS


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
    assert {"medical_en", "medical_de"}.issubset(names)
    english = next(
        profile for profile in response.json["profiles"] if profile["name"] == "medical_en"
    )
    assert "synonym" in {category["id"] for category in english["categories"]}
    assert "equivalent_to" in {relation["id"] for relation in english["relations"]}
    assert english["default_categories"]


def test_query_expansion_profile_route_returns_profile_metadata(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().get("/query-expansion/profiles/medical_de")

    assert response.status_code == 200
    assert response.json["name"] == "medical_de"
    assert response.json["language_name"] == "Deutsch"
    assert "symptom" in {category["id"] for category in response.json["categories"]}
    relation_ids = {relation["id"] for relation in response.json["relations"]}
    assert {relation.id for relation in MEDICAL_RELATION_DEFINITIONS}.issubset(
        relation_ids
    )
    may_indicate = next(
        relation
        for relation in response.json["relations"]
        if relation["id"] == "may_indicate"
    )
    assert may_indicate["label"] == "May Indicate"
    assert may_indicate["source_categories"] == ["symptom"]
    assert may_indicate["target_categories"] == ["diagnosis"]


def test_query_expansion_profile_relation_ids_cover_response_relation_ids(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().get("/query-expansion/profiles/medical_en")

    profile_relation_ids = {relation["id"] for relation in response.json["relations"]}
    expansion = QueryExpansionResponse(
        term="abdominal pain",
        language="en",
        expansions={},
        concepts=[
            ExpansionConcept(
                id="abdominal_pain",
                label="abdominal pain",
                category="symptom",
                terms=["abdominal pain"],
            ),
            ExpansionConcept(
                id="appendicitis",
                label="appendicitis",
                category="diagnosis",
                terms=["appendicitis"],
            ),
        ],
        relations=[
            ExpansionSemanticRelation(
                source_concept_id="abdominal_pain",
                relation="may_indicate",
                target_concept_id="appendicitis",
                confidence=0.8,
            )
        ],
    )

    assert {relation.relation for relation in expansion.relations}.issubset(
        profile_relation_ids
    )


def test_query_expansion_profile_route_normalizes_separators(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().get("/query-expansion/profiles/medical-de")

    assert response.status_code == 200
    assert response.json["name"] == "medical_de"


def test_query_expansion_profile_route_404s_for_unknown_profile(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().get("/query-expansion/profiles/unknown")

    assert response.status_code == 404
    assert "Unknown query-expansion domain profile" in response.json["error"]


def test_query_expansion_profile_route_does_not_resolve_bare_language(tmp_path):
    app = create_app(file_storage_dir=str(tmp_path), logging_setup_tuples=[])
    response = app.test_client().get("/query-expansion/profiles/de")

    assert response.status_code == 404


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
