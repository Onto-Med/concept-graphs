---
type: Module
title: src.nlp.negation package
description: Project-owned NegEx-style negation detection integrated into preprocessing.
tags: [module, nlp, negation, spacy]
status: stable
generated: { by: pi-agent/gpt-5.5, at: 2026-09-01T00:00:00Z }
sources:
  - id: preprocessing
    resource: /src/pipeline/steps/preprocessing_util.py
    title: PreprocessingUtil negation configuration
  - id: negation
    resource: /src/nlp/negation/negation.py
    title: Negex implementation
  - id: termsets
    resource: /src/nlp/negation/termsets.py
    title: Negation termsets
  - id: utils
    resource: /src/nlp/negation/utils.py
    title: Negation options/enums
---

# Responsibility

`src.nlp.negation` contains the project-owned negation component used during preprocessing to flag or omit negated named entities and noun chunks.

# Pipeline integration

`PreprocessingUtil.read_config()` accepts legacy/config-driven `negspacy` options, converts feature-of-interest strings such as `nc`, `ne`, and `both`, and stores a `NegspacyConfig` plus `omit_negated_chunks` flag in preprocessing config.[^preprocessing]

`DataProcessingFactory` then adds the negation pipe to the spaCy pipeline when configured and excludes negated chunks from chunk counts/document matrices when `omit_negated_chunks` is enabled.

# Module contents

* `negation.py`: NegEx-like spaCy component.
* `termsets.py`: language/domain termsets.
* `context.py`: context-building helpers.
* `utils.py`: feature-of-interest constants.
* `README.md`: local notes for this package.

# Compatibility note

The repository still contains old `src/negspacy/__pycache__/` files, but project status indicates custom negation code was reorganized to `src/nlp/negation/`.

[^preprocessing]: PreprocessingUtil negation configuration
