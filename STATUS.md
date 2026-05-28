# Project Cleanup Status

## Summary

The project is now significantly cleaner and more Pythonic than at the start of the refactoring. The main improvements are structural: imports are explicit, route handling has been split out of `main.py`, and formerly top-level modules have been moved into clearer packages.

The remaining work is mostly about deeper internal decomposition, test reliability, and smaller Pythonic cleanups such as safer error handling, logging, and configuration models.

## Completed Improvements

### Package and import cleanup

- Removed `sys.path.insert(...)` usage.
- Replaced wildcard import from `main_methods` with explicit imports.
- Moved former top-level `src/*.py` modules into domain packages:

```text
src/core/
src/common/
src/storage/
src/pipeline/
src/api/
```

- Added a compatibility import redirector in `src/__init__.py` for older pickle/module paths.

### Root directory cleanup

Only these Python files now remain in the project root:

```text
main.py
main_methods.py
main_utils.py
```

Moved root utility files into package locations, for example:

```text
load_utils.py              -> src/pipeline/load_utils.py
preprocessing_util.py      -> src/pipeline/steps/preprocessing_util.py
embedding_util.py          -> src/pipeline/steps/embedding_util.py
clustering_util.py         -> src/pipeline/steps/clustering_util.py
graph_creation_util.py     -> src/pipeline/steps/graph_creation_util.py
integration_util.py        -> src/pipeline/steps/integration_util.py
download_models.py         -> src/scripts/download_models.py
```

### API structure

`main.py` is now mostly responsible for:

- creating the Flask app
- setting up logging
- initializing shared runtime state
- registering routes
- starting the server

Routes were split into modules under:

```text
src/api/routes/
```

Current route modules include:

```text
artifacts.py
graph_documents.py
pipeline.py
processes.py
rag.py
status.py
static.py
```

### Pipeline route refactor

The large `/pipeline` handler was moved into `src/api/pipeline.py` and decomposed into smaller helper functions.

`run_complete_pipeline()` is now short and readable, while helpers handle:

- request parsing
- temporary uploads
- vector-store config normalization
- document-server loading
- process preparation
- skipped step handling
- background thread startup
- response creation

### Docker cleanup

`.dockerignore` was expanded so Docker builds exclude tests, notebooks, virtual environments, caches, and local development artifacts.

Excluded examples:

```text
test/
src/tests/
jupyter-notebooks/
*.ipynb
.venv/
.pytest_cache/
__pycache__/
.idea/
```

## Current Assessment

The project structure is now much more maintainable. The main top-level organization problems have been addressed.

The project now resembles an application package rather than a research/prototype script collection. The remaining issues are mostly inside large domain modules and tests.

## Remaining Issues

### `main_methods.py` is still too broad

`main_methods.py` is smaller than before, but still mixes several responsibilities:

- config loading
- graph response helpers
- data-server helpers
- RAG/vectorstore helpers
- deletion helpers

Suggested future split:

```text
src/api/services/configuration.py
src/api/services/data_server.py
src/api/services/graph_responses.py
src/api/services/rag_vectorstore.py
src/pipeline/deletion.py
```

### Route modules could use Flask Blueprints

The route modules currently use registration functions with nested route handlers. This works, but a more idiomatic Flask structure would use `Blueprint`s.

Example target:

```python
bp = Blueprint("rag", __name__)

@bp.route("/rag/init", methods=["POST"])
def init_rag():
    ...
```

Then register blueprints in `main.py`.

### Core modules remain large

Large files still include:

```text
src/core/cluster_functions.py
src/core/data_functions.py
src/core/graph_functions.py
src/storage/marqo_external_utils.py
src/common/util_functions.py
```

These are now in better locations, but could later be split internally.

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

### Utility module is still mixed

`src/common/util_functions.py` contains several unrelated concerns:

- pickle I/O
- color handling
- store abstractions
- clustering helpers
- spaCy extension setup

Suggested split:

```text
src/common/io.py
src/common/colors.py
src/common/spacy_extensions.py
src/common/stores.py
src/common/iterables.py
```

### Pythonic cleanup still needed

Remaining patterns to improve:

- broad `except Exception` blocks
- `print(...)` calls in reusable/library code
- `yaml.load` usage
- namedtuple config objects
- large methods
- some Java-style class names

### Tests need attention

The test suite still fails for pre-existing reasons unrelated to the recent refactors:

- missing `ig`/igraph usage in pruning tests
- tests depending on absent pickle fixtures under `tmp/`
- incomplete `TestGraphCreator` fixture setup

Fixing or quarantining these tests should be a priority before deeper refactoring.

## Recommended Next Steps

1. Fix or quarantine broken tests.
2. Convert route modules to Flask Blueprints.
3. Split `main_methods.py` into focused service modules.
4. Replace request/config namedtuples with dataclasses.
5. Replace `yaml.load` with `yaml.safe_load` where possible.
6. Replace library `print(...)` calls with logging.
7. Add Ruff for linting and import cleanup.
8. Gradually split large core modules into smaller domain modules.

## Overall Status

The project is in a much better structural state. The biggest remaining risk is the unreliable test suite, which makes future refactoring harder to verify. Once tests are stabilized, the project will be well positioned for deeper cleanup of the core domain modules.
