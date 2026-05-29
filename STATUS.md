# Project Cleanup Status

## Summary

The project has been substantially reorganized from a script-heavy/prototype layout into a more package-oriented Flask application.

Major completed work includes:

- top-level module cleanup
- domain package organization
- Flask Blueprint route structure
- narrower route dependencies
- Flask app factory pattern
- split of `main_methods.py`
- split of `main_utils.py`
- split/removal of `src/common/util_functions.py`
- relocation of tests out of `src/`
- relocation of experiment/evaluation scripts out of `src/`
- Docker entrypoint and compose cleanup

At this point, the remaining cleanup work is mostly around test fixtures, large core modules, linting, and smaller Pythonic improvements.

## Current Top-Level Layout

Only one Python file remains in the repository root:

```text
main.py
```

`main.py` now exposes a Flask app factory:

```python
create_app()
create_app_context()
configure_logging()
```

The application no longer creates a global app/context at import time.

## Completed Improvements

### Package and import cleanup

Completed.

Former top-level implementation modules were moved into focused packages:

```text
src/core/
src/common/
src/storage/
src/pipeline/
src/api/
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

- removed `sys.path.insert(...)` usage
- removed wildcard imports from app code
- updated package build config to include nested `src/**/*.py`
- added a compatibility import redirector in `src/__init__.py` for old pickle/module paths where target modules still exist

### Flask app factory

Completed.

`main.py` now uses the standard Flask app factory pattern:

```python
def create_app(...) -> flask.Flask:
    ...
```

The app context is attached to:

```python
app.extensions["concept_graphs_context"]
```

Docker/Waitress now starts the app with factory-call mode:

```text
waitress-serve --call main:create_app
```

### App context refactor

Completed.

The former broad `main_objects`/`PersistentObjects` style was replaced with grouped runtime context objects:

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

All route modules under `src/api/routes/` now use Blueprint factories.

Current route modules:

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

The `/pipeline` route orchestration was moved into:

```text
src/api/pipeline.py
```

The module now separates concerns such as:

- request parsing
- temporary upload handling
- vector-store config normalization
- document-server loading
- process preparation
- skipped step handling
- background thread startup
- response creation

### `main_methods.py` split

Completed.

`main_methods.py` was split into focused service modules and removed.

New modules:

```text
src/api/services/pipeline_params.py
src/api/services/document_server.py
src/api/services/configuration.py
src/api/services/artifact_responses.py
src/api/services/process_management.py
src/api/services/rag_vectorstore.py
```

### `main_utils.py` split

Completed.

`main_utils.py` was split into focused modules and removed.

New modules:

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

### `src/common/util_functions.py` split

Completed.

The former mixed utility module was split and removed. New focused modules include:

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

### Test and experiment relocation

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

### Docker cleanup

Completed.

Improvements:

- `.dockerignore` excludes tests, notebooks, virtual environments, caches, and dev artifacts
- Docker entrypoint updated for the app factory
- `uv run --no-sync` added to avoid dependency sync at container startup
- `docker-compose-network.yml` no longer bind-mounts the project over `/rest_api`
- `docker-compose.yml` image tag updated to `0.9.6`
- added `docker-compose-es.yml` for Elasticsearch

## Current Validation Status

Recent successful checks:

```bash
uv run python -m compileall -q main.py src test experiments
```

Flask smoke checks have passed for representative endpoints, including:

```text
GET /                                                 -> 200
GET /openapi                                          -> 200
GET /status/rag                                       -> 404 when RAG is not initialized
GET /processes                                        -> 404 when no processes exist
GET /pipeline/configuration?default=true&language=en  -> 200
GET /rag/question?q=test                              -> 404 when RAG is not initialized
```

## Known Test Status

The full test suite now collects successfully. Current status is:

```text
1 passed, 1 skipped, 7 failed
```

Remaining failures are pre-existing fixture/data issues:

```text
test/test_graph_functions.py
  incomplete TestGraphCreator fixture setup

test/test_main_utils.py
test/test_document_clustering_on_corpus.py
  missing pickle fixtures under tmp/
```

The pruning tests are skipped cleanly when optional `igraph` is unavailable.

Fixing or quarantining these tests is now one of the highest-value next steps.

## Remaining Issues / Recommended Next Steps

### 1. Fix or quarantine broken tests

Priority: high.

Suggested actions:

- skip fixture-dependent tests when pickle fixtures are missing
- repair incomplete graph test setup
- add small smoke/unit tests for app factory and route registration

### 2. Add Ruff

Priority: high.

Ruff would help detect:

- unused imports
- broad exceptions
- undefined names
- import ordering issues
- simple modernization opportunities

### 3. Replace namedtuples/config containers with dataclasses

Some request/query/config objects still use namedtuple-style structures.

Suggested target:

```python
@dataclass(frozen=True)
class PipelineQueryParams:
    ...
```

### 4. Replace remaining `print(...)` calls with logging

Several library/runtime modules still contain `print(...)` calls. These should use module loggers instead.

### 5. Replace remaining broad `except Exception` blocks where practical

Broad exception handling still exists in multiple places. Some may be acceptable at route boundaries, but internal logic should become more specific where possible.

### 6. Continue splitting large domain modules

The biggest remaining modules are still in the domain layer:

```text
src/core/cluster_functions.py
src/core/data_functions.py
src/core/graph_functions.py
src/storage/marqo_external_utils.py
```

Possible future structure:

```text
src/core/data/
  factory.py
  processing.py
  cleaning.py

src/core/clustering/
  phrase_cluster.py
  word_embedding.py
  metrics.py

src/core/graph/
  creation.py
  incorporation.py
  visualization.py

src/storage/marqo/
  client.py
  documents.py
  embeddings.py
```

## Overall Status

The major application-structure refactors are complete. The project now has a much cleaner Flask/package architecture:

- app factory
- Blueprints
- grouped app context
- service modules
- only `main.py` at root
- domain packages under `src/`

The next best investments are test stabilization, linting, and gradual decomposition of large core/domain modules.
