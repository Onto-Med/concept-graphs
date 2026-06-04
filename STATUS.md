# Project Cleanup Status

## Current status

The project has been refactored from a script-heavy/prototype layout into a package-oriented Flask API with an app factory, Blueprints, grouped runtime context, focused service modules, OpenAPI documentation, RAG/query-expansion support, and a substantially broader test suite.

Current validation:

```text
ruff: all checks passed
pytest: 66 passed
```

## Where it came from

The project originally had most runtime behavior concentrated in root-level modules and broad utility files, including:

```text
main.py
main_methods.py
main_utils.py
src/common/util_functions.py
src/*_functions.py
src/marqo_external_utils.py
src/tests/
src/run/
```

Problems addressed:

- root-level and top-level `src/` modules mixed unrelated responsibilities
- Flask app/routes/runtime state were tightly coupled
- app/context was created at import time
- routes depended on broad shared objects
- tests and experiments lived inside production package code
- Docker entrypoints assumed the older structure
- large domain modules were hard to navigate
- broad exception handling and print/debug output were scattered
- OpenAPI docs lagged behind implemented endpoints

## Current high-level layout

```text
main.py                         # Flask app factory entrypoint
api/                            # Swagger UI assets + OpenAPI spec
conf/                           # default pipeline and query-expansion configs
src/
  api/                          # routes, request parsing, services, pipeline route support
  common/                       # shared small helpers
  core/                         # data, embedding, clustering, graph, integration domain code
  nlp/negation/                 # project-owned negation code
  pipeline/                     # pipeline utilities, status, document add/delete workflows
  pruning/                      # graph pruning algorithms
  rag/                          # RAG orchestration, chatters, chunk stores
  query_expansion/              # LLM query expansion, grounding, prompt profiles
  storage/                      # storage interfaces and Marqo implementations
test/                           # grouped test suite + fixtures/scripts
experiments/                    # former evaluation/run scripts
```

Only one Python file remains in the repository root:

```text
main.py
```

## Runtime architecture

`main.py` exposes:

```python
create_app()
create_app_context()
configure_logging()
```

The app no longer creates global runtime state at import time.

Shared state is attached to Flask via:

```python
app.extensions["concept_graphs_context"]
```

Grouped context lives in `src/api/context.py`:

```text
ProcessContext        # running processes and threads
PipelineContext       # active pipeline step objects
StorageContext        # file storage directory
RagContext            # per-process active RAG components
AppContext            # composed application context
```

RAG runtime state is now per process/corpus:

```python
app_context.rag.active_by_process[process]
```

Question answering is stateless with respect to retrieved documents; request-local documents are no longer stored on the shared `RAG` object.

## API structure

Routes are organized as Flask Blueprint factories under:

```text
src/api/routes/
```

Main route groups:

```text
artifacts.py           # preprocessing/embedding/clustering/graph inspection
graph_documents.py     # document add/status/delete
pipeline.py            # pipeline start/configuration
processes.py           # process list/status/stop/delete
rag.py                 # RAG init/question
query_expansion.py     # LLM query expansion
status.py              # document-server and RAG status
static.py              # Swagger UI/static docs
```

The `/pipeline` orchestration was split into:

```text
src/api/pipeline.py
src/api/pipeline_support/models.py
src/api/pipeline_support/request_data.py
src/api/pipeline_support/vectorstore.py
src/api/pipeline_support/document_server.py
src/api/pipeline_support/steps.py
src/api/pipeline_support/execution.py
```

## Domain/package cleanup

Former broad modules were split or moved into focused packages.

Core domain split:

```text
src/core/data_functions.py       -> compatibility exports for src/core/data/
src/core/cluster_functions.py    -> compatibility exports for src/core/clustering/
src/core/graph_functions.py      -> compatibility exports for src/core/graph/
```

Current focused packages include:

```text
src/core/data/
src/core/clustering/
src/core/graph/
src/storage/marqo/
src/nlp/negation/
src/pipeline/steps/
```

Compatibility exports/redirects remain where useful for old imports and pickle paths.

## Storage, document addition, and deletion

Marqo storage was split from one large module into:

```text
src/storage/marqo/documents.py
src/storage/marqo/embedding_store.py
src/storage/marqo/document_store.py
```

`src/storage/marqo_external_utils.py` remains as a compatibility export module.

Document addition now stores provenance in Marqo/vector-store entries:

```json
{
  "documents": [{"id": "document-id", "offsets": [[0, 10]]}],
  "source": "document_addition"
}
```

Implemented:

```text
DELETE /graph/document/{document_id}
```

It removes document provenance from serialized graph nodes and, when vector-store configuration is available, from Marqo entries. It can optionally remove unreferenced graph nodes and unreferenced vector-store entries.

Document addition currently updates graph/vector-store state only:

| Target | Added by document addition? |
|---|---:|
| Concept graph pickle | yes |
| Vector store / Marqo phrase index | yes |
| Existing processed data pickle | no |
| External document index server | no |

## Query expansion

Implemented initial LLM-first query expansion:

```text
POST /query-expansion
```

Package layout:

```text
src/query_expansion/
  categories.py
  models.py
  prompts.py
  generator.py
  grounding.py
  service.py
  sources/
```

Current behavior:

- default generator uses LangChain
- Pydantic validates structured LLM output
- PydanticAI generator remains available for future/custom use
- fixed stable category IDs are used for API compatibility
- optional local YAML/JSON grounding source is implemented
- HTTP source adapter is still placeholder-level
- provider API keys can be supplied via request headers such as `Authorization: Bearer ...` or `X-LLM-API-Key`

Prompt profiles live in:

```text
conf/query-expansion/localization/en.yml
conf/query-expansion/localization/de.yml
conf/query-expansion/grounding/medical_terms.example.yml
```

Prompt templates and category descriptions can be overridden per request while keeping category IDs stable.

## OpenAPI / Swagger

Swagger UI assets live in:

```text
api/
```

OpenAPI spec:

```text
api/concept-graphs-api.yml
```

Recent updates:

- version updated to `1.1.0`
- endpoints reviewed against Flask route registrations
- tags and operation IDs added
- schema `required` usage cleaned up
- RAG question descriptions added
- `DELETE /graph/document/{document_id}` documented
- `POST /query-expansion` documented with request/response schemas and examples
- Swagger UI options improved

Static/documentation routes (`/`, `/openapi`, static files) are intentionally excluded from business API parity.

## Docker/runtime cleanup

Docker entrypoints now use the app factory:

```text
waitress-serve --call --port=9007 main:create_app
```

Startup uses:

```text
uv run --no-sync
```

Production compose no longer bind-mounts the project over `/rest_api`, avoiding accidental hiding of the image-built `.venv`.

`.dockerignore` excludes tests, experiments, notebooks, and development/cache artifacts.

## Testing

Tests were moved out of production package code and grouped under `test/`.

Current structure:

```text
test/
  api/
    routes/
    pipeline_support/
  core/
  pipeline/
  pruning/
  rag/
  query_expansion/
  storage/marqo/
  data/
```

Current coverage includes:

- app factory/context smoke tests
- OpenAPI/Flask route parity
- API route tests for RAG and graph-document add/delete
- configuration/request parsing
- pipeline step scheduling/skip behavior
- Marqo provenance add/remove behavior with fake clients
- document deletion workflow
- core data/graph/clustering behavior
- NetworkX pruning behavior
- query-expansion service, prompt, and API-route behavior

Current result:

```text
66 passed
```

## Tooling and style

Ruff replaced Black and is used for formatting and linting.

Enabled Ruff groups:

```text
E    pycodestyle errors
F    pyflakes
I    import sorting
UP   pyupgrade modernization
```

Standard validation commands:

```bash
uv run --group test ruff format .
uv run --group test ruff check .
uv run --no-sync python -m compileall -q main.py src test
uv run --no-sync pytest -q
```

## Other completed cleanup

- removed `sys.path.insert(...)` usage from app code
- removed wildcard imports from app code
- replaced remaining `namedtuple` usage with dataclasses
- replaced most runtime/library/test prints with logging
- reorganized custom negation code from `src/negspacy/` to `src/nlp/negation/`
- renamed RAG modules to lowercase Pythonic module names
- replaced deprecated `scipy.stats.binom_test` with `binomtest`
- pruning support is now explicitly NetworkX-only
- RAG chunk metadata now includes retrieved snippets and document-character offsets when available
- broad exception handling narrowed; remaining broad catches are intentional route/workflow/process safety nets

## Remaining recommended work

1. Continue optional Ruff expansion:
   - `B` bugbear-style checks
   - `G` logging format checks
   - `SIM` simplification checks
2. Optional deeper module decomposition:
   - `src/core/clustering/word_embedding.py`
   - `src/core/data/factory.py`
   - `src/storage/marqo/embedding_store.py`
   - `src/pipeline/document_addition.py`
3. Optional document-server consistency update:
   - add full document insertion to the external document server during `/graph/document/add`
4. Optional cleanup:
   - remove generated cache directories such as `src/negspacy/__pycache__/`
   - remove obsolete local artifacts such as root-level `graph_dump.pickle` if still present
   - review/remove old commented-out historical code in `src/pruning/unimodal.py`

## Known warnings

The test suite passes. Remaining warnings are from third-party dependencies, including Click/spaCy/Pydantic/Transformers/SciPy-related deprecations.
