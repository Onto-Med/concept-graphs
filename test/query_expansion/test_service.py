from src.query_expansion.generator import LangChainExpansionGenerator
from src.query_expansion.models import (
    ExpansionConcept,
    ExpansionGeneration,
    ExpansionSemanticRelation,
    GeneratedExpansionCandidate,
    GroundingOptions,
    GroundingStatus,
    LLMConfig,
    QueryExpansionRequest,
)
from src.query_expansion.service import QueryExpansionService
from src.query_expansion.sources.local import LocalTerminologySource


class FakeStructuredLLM:
    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        return {
            "candidates": [
                {
                    "term": "heart attack",
                    "category": "synonym",
                    "rationale": "common lay term",
                }
            ]
        }


class FakeJsonLLM:
    def invoke(self, prompt):
        return '{"candidates": [{"term": "aspirin", "category": "medication"}]}'


class FakeGenerator:
    def generate(self, request):
        return ExpansionGeneration(
            candidates=[
                GeneratedExpansionCandidate(
                    term="heart attack", category="synonym", rationale="lay term"
                ),
                GeneratedExpansionCandidate(
                    term="aspirin", category="medication", rationale="common therapy"
                ),
                GeneratedExpansionCandidate(
                    term="unverified", category="symptom", rationale="test"
                ),
            ]
        )


class FakeSemanticGenerator:
    def generate(self, request):
        return ExpansionGeneration(
            candidates=[
                GeneratedExpansionCandidate(term="abdominal pain", category="symptom"),
                GeneratedExpansionCandidate(term="appendicitis", category="diagnosis"),
                GeneratedExpansionCandidate(term="aspirin", category="medication"),
            ],
            concepts=[
                ExpansionConcept(
                    id="c1",
                    label="abdominal pain",
                    category="symptom",
                    terms=["Bauchschmerzen", "abdominal pain"],
                ),
                ExpansionConcept(
                    id="c2",
                    label="appendicitis",
                    category="diagnosis",
                    terms=["appendicitis"],
                ),
                ExpansionConcept(
                    id="c3",
                    label="aspirin",
                    category="medication",
                    terms=["aspirin"],
                ),
            ],
            relations=[
                ExpansionSemanticRelation(
                    source_concept_id="c1",
                    relation="may_indicate",
                    target_concept_id="c2",
                    confidence=0.8,
                ),
                ExpansionSemanticRelation(
                    source_concept_id="c1",
                    relation="treated_by",
                    target_concept_id="c3",
                    confidence=0.8,
                ),
            ],
        )


def test_langchain_generator_validates_structured_output_with_pydantic():
    request = QueryExpansionRequest(
        term="myocardial infarction",
        categories=["synonym"],
        llm=LLMConfig(model="test-model"),
    )

    generation = LangChainExpansionGenerator(llm=FakeStructuredLLM()).generate(request)

    assert generation.candidates[0].term == "heart attack"
    assert generation.candidates[0].category == "synonym"


def test_langchain_generator_can_parse_json_fallback_output():
    request = QueryExpansionRequest(
        term="myocardial infarction",
        categories=["medication"],
        llm=LLMConfig(model="test-model"),
    )

    generation = LangChainExpansionGenerator(llm=FakeJsonLLM()).generate(request)

    assert generation.candidates[0].term == "aspirin"
    assert generation.candidates[0].category == "medication"


def test_query_expansion_service_generates_and_grounds_candidates(tmp_path):
    source_file = tmp_path / "terms.yaml"
    source_file.write_text(
        """
terms:
  - id: C001
    term: myocardial infarction
    synonyms: [heart attack, MI]
  - id: C002
    term: aspirin
"""
    )
    source = LocalTerminologySource("local", source_file)
    request = QueryExpansionRequest(
        term="myocardial infarction",
        categories=["synonym", "medication", "symptom"],
        llm=LLMConfig(model="test-model"),
    )

    response = QueryExpansionService(generator=FakeGenerator()).expand(
        request, sources=[source]
    )

    assert response.term == "myocardial infarction"
    assert response.expansions["synonym"][0].term == "heart attack"
    assert response.expansions["synonym"][0].status == GroundingStatus.GROUNDED
    assert response.expansions["medication"][0].term == "aspirin"
    assert response.expansions["medication"][0].evidence[0].source_id == "C002"
    assert response.expansions["symptom"][0].status == GroundingStatus.LLM_ONLY


def test_local_grounding_respects_optional_category_metadata(tmp_path):
    source_file = tmp_path / "terms.yaml"
    source_file.write_text(
        """
terms:
  - id: C001
    term: aspirin
    category: medication
  - id: C002
    term: fatigue
    categories: [symptom, related_term]
"""
    )
    source = LocalTerminologySource("local", source_file)

    assert source.ground(
        GeneratedExpansionCandidate(term="aspirin", category="medication")
    )
    assert not source.ground(
        GeneratedExpansionCandidate(term="aspirin", category="symptom")
    )
    assert source.ground(
        GeneratedExpansionCandidate(term="fatigue", category="symptom")
    )


def test_query_expansion_service_validates_generated_concepts_and_relations():
    request = QueryExpansionRequest(
        term="Bauchschmerzen",
        categories=["symptom", "diagnosis", "medication"],
        relations=["may_indicate"],
        llm=LLMConfig(model="test-model"),
    )

    response = QueryExpansionService(generator=FakeSemanticGenerator()).expand(
        request, sources=[]
    )

    assert [concept.id for concept in response.concepts] == ["c1", "c2", "c3"]
    assert len(response.relations) == 1
    assert response.relations[0].relation == "may_indicate"
    assert response.relations[0].source_concept_id == "c1"
    assert response.relations[0].target_concept_id == "c2"


def test_query_expansion_service_can_exclude_llm_only_candidates(tmp_path):
    source_file = tmp_path / "terms.yaml"
    source_file.write_text("terms: []")
    request = QueryExpansionRequest(
        term="x",
        categories=["symptom"],
        llm=LLMConfig(model="test-model"),
        grounding=GroundingOptions(include_llm_only=False),
    )

    response = QueryExpansionService(generator=FakeGenerator()).expand(
        request, sources=[LocalTerminologySource("local", source_file)]
    )

    assert response.expansions == {"symptom": []}
