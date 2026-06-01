# Project Cleanup Status

## Summary

The project has been substantially reorganized from a script-heavy/prototype layout into a package-oriented Flask application.

Major completed work includes:

- top-level module cleanup
- domain package organization
- Flask Blueprint route structure
- narrower route dependencies
- Flask app factory pattern
- grouped app runtime context
- split/removal of `main_methods.py`
- split/removal of `main_utils.py`
- split/removal of `src/common/util_functions.py`
- relocation of tests out of `src/`
- relocation of experiment/evaluation scripts out of `src/`
- Docker entrypoint and compose cleanup
- fixture-generation helper for pipeline test data
- test suite stabilization
- Ruff adoption in place of Black
- RAG module rename to lowercase Pythonic modules
- `namedtuple` replacement with dataclasses
- print-to-logging cleanup
- negation/negspacy package reorganization
- broad exception cleanup phases 1–6
- split of large core modules into `src/core/data`, `src/core/clustering`, and `src/core/graph`
- split of Marqo storage helpers into `src/storage/marqo`
- OpenAPI/Swagger UI review and cleanup
- Marqo document provenance for vector-store document additions
- `DELETE /graph/document/{document_id}` endpoint implementation
- NetworkX-only pruning support
- `binom_test` replacement with `binomtest`

Current validation status:

```text
ruff: all checks passed
pytest: 12 passed
```

## Current Top-Level Layout

Only one Python file remains in the repository root:

```text
main.py
```

`main.py` exposes the Flask app factory API:

```python
create_app()
create_app_context()
configure_logging()
```

The application no longer creates a global app/context at import time.

## Completed Improvements

### Package and import cleanup

Former top-level implementation modules were moved into focused packages:

```text
src/core/
src/common/
src/storage/
src/pipeline/
src/api/
src/nlp/
```

Examples:

```text
src/data_functions.py            -> src/core/data_functions.py
src/embedding_functions.py       -> src/core/embedding_functions.py
src/cluster_functions.py         -> src/core/cluster_functions.py
src/graph_functions.py           -> src/core/graph_functions.py
src/integration_functions.py     -> src/core/integration_functions.py
src/util_functions.py            -> split across focused common/core/storage modules
src/marqo_external_utils.py      -> src/storage/marqo_external_utils.py
```

Root utility files were also moved:

```text
load_utils.py              -> src/pipeline/load_utils.py
preprocessing_util.py      -> src/pipeline/steps/preprocessing_util.py
embedding_util.py          -> src/pipeline/steps/embedding_util.py
clustering_util.py         -> src/pipeline/steps/clustering_util.py
graph_creation_util.py     -> src/pipeline/steps/graph_creation_util.py
integration_util.py        -> src/pipeline/steps/integration_util.py
download_models.py         -> src/scripts/download_models.py
```

Other import cleanup:

- removed `sys.path.insert(...)` usage from app code
- removed wildcard imports from app code
- updated package build config to include nested `src/**/*.py`
- added compatibility import redirects in `src/__init__.py` for old pickle/module paths where useful

### Flask app factory

Completed.

`main.py` now uses the Flask app factory pattern:

```python
def create_app(...) -> flask.Flask:
    ...
```

The app context is attached to:

```python
app.extensions["concept_graphs_context"]
```

Docker/Waitress starts the app with factory-call mode:

```text
waitress-serve --call main:create_app
```

### App context refactor

Completed.

Runtime state is grouped in:

```text
src/api/context.py
```

Current context classes:

```python
ActiveRAG
ProcessContext
PipelineContext
StorageContext
RagContext
AppContext
```

Runtime state is now grouped by responsibility:

```python
app_context.processes.running
app_context.processes.threads
app_context.pipeline.active_objects
app_context.storage.file_storage_dir
app_context.rag.active
```

### Flask Blueprints

Completed.

Route modules under `src/api/routes/` use Blueprint factories:

```text
src/api/routes/artifacts.py
src/api/routes/graph_documents.py
src/api/routes/pipeline.py
src/api/routes/processes.py
src/api/routes/rag.py
src/api/routes/status.py
src/api/routes/static.py
```

Current Blueprint factory signatures use narrowed dependencies:

```python
create_static_blueprint(app)
create_status_blueprint(app, rag)
create_artifact_blueprint(app, storage, pipeline)
create_graph_document_blueprint(app, processes, pipeline, storage)
create_pipeline_blueprint(app, processes, pipeline, storage)
create_process_blueprint(app, processes, pipeline, storage)
create_rag_blueprint(rag, processes, storage, pipeline)
```

### Pipeline route refactor

Completed.

The `/pipeline` route orchestration was moved into a compact public module:

```text
src/api/pipeline.py
```

Focused helpers now live under:

```text
src/api/pipeline_support/
```

Modules:

```text
src/api/pipeline_support/models.py
src/api/pipeline_support/request_data.py
src/api/pipeline_support/vectorstore.py
src/api/pipeline_support/document_server.py
src/api/pipeline_support/steps.py
src/api/pipeline_support/execution.py
```

`src/api/pipeline.py` remains the stable import location for:

```python
run_complete_pipeline(...)
```

### Service and utility splits

Completed.

`main_methods.py` was split into focused service modules and removed:

```text
src/api/services/pipeline_params.py
src/api/services/document_server.py
src/api/services/configuration.py
src/api/services/artifact_responses.py
src/api/services/process_management.py
src/api/services/rag_vectorstore.py
```

`main_utils.py` was split into focused modules and removed:

```text
src/api/context.py
src/api/responses.py
src/common/parsing.py
src/common/spacy_utils.py
src/common/threads.py
src/pipeline/base.py
src/pipeline/document_results.py
src/pipeline/status.py
```

`src/common/util_functions.py` was split and removed:

```text
src/common/config_loading.py
src/common/io.py
src/common/iterables.py
src/common/meta.py
src/common/spacy_extensions.py
src/common/colors.py
src/core/clustering_config.py
src/core/documents.py
src/core/metrics.py
src/core/reduction.py
src/storage/interfaces.py
```

### Test and experiment relocation/stabilization

Completed.

Tests were moved out of production package code:

```text
src/tests/test_data_functions.py      -> test/test_data_functions.py
src/tests/test_graph_functions.py     -> test/test_graph_functions.py
src/pruning/test_unimodal.py          -> test/test_pruning_unimodal.py
```

Experiment/evaluation scripts were moved out of `src/`:

```text
src/run/run.py    -> experiments/run.py
src/run/test.py   -> experiments/evaluation.py
src/run/cross.py  -> experiments/cross.py
```

The old `src/tests/` and `src/run/` packages were removed.

The fixture-dependent tests now use fixtures under:

```text
test/data/results/grascco
```

Current test status:

```text
12 passed
```

### Pipeline fixture helper

Completed.

Added:

```text
test/data/scripts/run_pipeline_on_folder.py
```

The helper can:

- zip a document folder in memory
- call `main.create_app(...)`
- POST to `/pipeline`
- consume complete JSON config files such as `conf/pipeline-config_de.json`
- skip already-present pipeline artifacts by default
- force recomputation with `--no-skip-present`
- skip selected steps such as integration

### Docker cleanup

Completed.

Improvements:

- `.dockerignore` excludes tests, notebooks, virtual environments, caches, experiments, and dev artifacts
- Docker entrypoints updated for the app factory
- `uv run --no-sync` added to avoid dependency sync at container startup
- `docker-compose-network.yml` no longer bind-mounts the project over `/rest_api`
- `docker-compose.yml` image tag updated to `0.9.6`
- added `docker-compose-es.yml` for Elasticsearch

### Ruff adoption

Completed.

Black was replaced by Ruff in the test/dev dependency group.

Configured in:

```text
pyproject.toml
```

Current standard commands:

```bash
uv run --group test ruff format .
uv run --group test ruff check .
```

Enabled initial rule families:

```text
E  pycodestyle errors
F  pyflakes
I  import sorting
```

### Pythonic cleanup

Completed.

- Replaced remaining `namedtuple` usage with dataclasses.
- Replaced runtime/library/test/experiment `print(...)` calls with logging, except intentional CLI output in `test/data/scripts/run_pipeline_on_folder.py`.
- Renamed RAG modules to lowercase Pythonic names:

```text
src/rag/TextSplitters.py                         -> src/rag/text_splitters.py
src/rag/chatters/AbstractChatter.py             -> src/rag/chatters/base.py
src/rag/chatters/BlabladorChatter.py            -> src/rag/chatters/blablador.py
src/rag/chatters/OllamaChatter.py               -> src/rag/chatters/ollama.py
src/rag/embedding_stores/AbstractEmbeddingStore.py      -> src/rag/embedding_stores/base.py
src/rag/embedding_stores/MarqoChunkEmbeddingStore.py    -> src/rag/embedding_stores/marqo.py
```

Legacy redirects were added for old RAG import paths.

### Negation package cleanup

Completed.

Moved custom/project-owned negation code from the misleading top-level package:

```text
src/negspacy/
```

to:

```text
src/nlp/negation/
```

Moved files:

```text
src/negspacy/context.py   -> src/nlp/negation/context.py
src/negspacy/negation.py  -> src/nlp/negation/negation.py
src/negspacy/termsets.py  -> src/nlp/negation/termsets.py
src/negspacy/utils.py     -> src/nlp/negation/utils.py
```

Kept a lightweight compatibility package at `src/negspacy/__init__.py` and legacy redirects in `src/__init__.py`.

### Pruning cleanup

Completed.

- `src/pruning/unimodal.py` now explicitly supports NetworkX simple graphs only.
- Multigraphs, self-loops, missing weights, and non-numeric weights are rejected clearly.
- `scipy.stats.binom_test` was replaced by `scipy.stats.binomtest`.
- Broad numerical fallback was narrowed to specific numerical/data exceptions.

### Broad exception cleanup

Completed through phases 1–6.

Most broad `except Exception` blocks were replaced with specific exception handling or consolidated at appropriate boundaries.

Remaining broad catches are intentional safety nets:

```text
src/api/pipeline.py                  # route safety net with traceback logging
src/pipeline/document_addition.py    # workflow safety net with traceback logging
src/pipeline/base.py                 # process execution boundary with traceback logging
```

## Current Validation Status

Recent successful checks:

```bash
uv run --group test ruff format .
uv run --group test ruff check .
uv run --no-sync python -m compileall -q main.py src test
uv run --no-sync pytest -q
```

Result:

```text
All checks passed!
12 passed
```

Known warnings remain from dependencies and third-party libraries, including Click/spaCy/Pydantic/Transformers/SciPy-related deprecations.

### Core module split

Completed first structural split.

Compatibility modules now re-export from focused packages:

```text
src/core/data_functions.py       -> src/core/data/
src/core/cluster_functions.py    -> src/core/clustering/
src/core/graph_functions.py      -> src/core/graph/
```

New package layout:

```text
src/core/data/factory.py
src/core/data/text.py
src/core/clustering/word_embedding.py
src/core/clustering/phrase.py
src/core/graph/creation.py
src/core/graph/incorporation.py
src/core/graph/algorithms.py
```

Some new modules are still sizeable and can be decomposed further in later domain-focused passes.

### Marqo storage split

Completed first structural split.

Compatibility module now re-exports from the focused package:

```text
src/storage/marqo_external_utils.py -> src/storage/marqo/
```

New package layout:

```text
src/storage/marqo/documents.py
src/storage/marqo/embedding_store.py
src/storage/marqo/document_store.py
```

Project imports and dynamic class paths now prefer `src.storage.marqo`.

### OpenAPI / Swagger UI cleanup

Completed.

Reviewed `api/concept-graphs-api.yml` against Flask route registrations and updated the Swagger UI/spec metadata.

Changes include:

- OpenAPI version updated to `0.9.6`.
- Tags added for clearer Swagger UI grouping.
- `operationId` values added.
- `GET /status/document-server` placeholder endpoint documented.
- Invalid schema-level `required: true/false` fields fixed in request schemas.
- Swagger UI options improved in `api/index.html`.

Functional business endpoints are represented. Documentation/static routes (`/`, `/openapi`, static files) are intentionally omitted from the business API spec.

Not promoted in Swagger UI because they appear historical or non-functional:

```text
POST /graph/<path_arg>              # behaves like retrieval
DELETE /graph/document/<path_arg>   # returns 501 / not implemented
```

A later cleanup can remove or fully implement/document these methods.

### Marqo document provenance

Completed.

Document additions now store provenance metadata on Marqo vector-store entries:

```json
{
  "documents": [{"id": "document-id", "offsets": [[0, 10]]}],
  "source": "document_addition"
}
```

Retained/deduplicated phrase entries are also updated with the new document provenance. Added vector-store-side tests in `test/test_marqo_provenance.py`.

### Graph document deletion

Completed.

Implemented:

```text
DELETE /graph/document/{document_id}
```

The endpoint removes document references from serialized graph nodes and removes Marqo provenance when vector-store configuration is available. It can optionally remove unreferenced graph nodes and unreferenced vector-store entries. Added OpenAPI documentation and tests in `test/test_document_deletion.py`.

## Remaining Issues / Recommended Next Steps

### 1. Expand Ruff gradually
Priority: medium.

Current Ruff config includes:

- pycodestyle errors (`E`)
- pyflakes (`F`)
- import sorting (`I`)
- modernization / pyupgrade (`UP`)

Future additions could include selected rules for:

- bugbear-style checks (`B`)
- logging format issues (`G`)
- simplifications (`SIM`)

### 2. Improve docs around runtime/config behavior
Priority: medium.

Recommended docs:

- app factory / deployment entrypoint
- pipeline config schema expectations
- fixture-generation workflow
- vector-store vs pickle storage behavior

### 3. Document addition/document-server consistency
Priority: low to medium.

Document addition currently updates the graph/vector-store side only. It does not insert the full document into the external document index server.

| Target | Added by document addition? |
|---|---:|
| Concept graph pickle | yes |
| Vector store / Marqo phrase index | yes |
| Existing processed data pickle | no |
| External document index server | no |

Potential future update: add optional document-server insertion during document addition, respecting the configured document-server schema/index mapping.

### 4. Optional cleanup
Priority: low.

- Remove generated cache directories such as `src/negspacy/__pycache__/`.
- Remove any obsolete local generated artifacts such as root-level `graph_dump.pickle` if still present and unneeded.
- Review old commented-out historical code in `src/pruning/unimodal.py`.

## Overall Status

The major application-structure refactors are complete. The project now has a much cleaner Flask/package architecture:

- app factory
- Blueprints
- grouped app context
- service modules
- only `main.py` at root
- domain packages under `src/`
- tests outside production package code
- Docker entrypoints aligned with app factory
- Ruff as project formatter/linter
- passing test suite

The next best investments are expanding Ruff gradually, improving runtime/config documentation, and optionally continuing deeper decomposition of still-sizeable domain modules such as `src/core/clustering/word_embedding.py`, `src/core/data/factory.py`, `src/storage/marqo/embedding_store.py`, and `src/pipeline/document_addition.py`.
