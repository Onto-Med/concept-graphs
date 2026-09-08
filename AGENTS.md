# Agent Notes for Concept Graphs

When asked to describe, summarize or otherwise document,
or before architecture, module, pipeline, API, storage, RAG, or testing work, consult the local OKF bundle:

1. Start at `okf/concept-graphs-architecture/index.md`.
2. Follow only the concepts relevant to the task.
3. Prefer the OKF bundle's local architecture notes over guesses; then verify details in the linked source files.
4. When changing behavior, update affected OKF concept files and `okf/concept-graphs-architecture/log.md` if the architecture/working model changes.

## OKF Update Checklist

When changing architecture, API shape, request/response models, prompt variables, workflow behavior, storage/artifact behavior, GUI behavior, or testing/validation expectations:

1. Update the directly affected OKF module file under `okf/concept-graphs-architecture/modules/`.
2. Update related OKF operation/interface/workflow files when the change affects:
   - API request/response shape or endpoint behavior,
   - prompt profile variables, prompt selection, or localization behavior,
   - runtime/app context behavior,
   - pipeline/workflow behavior,
   - storage, artifacts, source adapters, or external integrations,
   - GUI behavior,
   - testing or validation commands/expectations.
3. Update `okf/concept-graphs-architecture/log.md` with the architecture/working-model change.
4. Before the final response, search the OKF bundle for stale wording related to the changed concept, renamed fields, removed behavior, or superseded design ideas.
   Example: `rg -n "old-field|old-term|query_mode|future work" okf/concept-graphs-architecture`.
5. If a design note is superseded, mark it clearly as superseded or historical context instead of leaving it as active implementation guidance.
6. In the final response, list which OKF files were updated or explicitly state that no OKF update was needed.

Key entry points:

- System overview: `okf/concept-graphs-architecture/overview.md`
- Runtime/app context: `okf/concept-graphs-architecture/runtime.md`
- Pipeline workflow: `okf/concept-graphs-architecture/workflows/concept-graph-pipeline.md`
- API surface: `okf/concept-graphs-architecture/interfaces/api-surface.md`
- Module docs: `okf/concept-graphs-architecture/modules/`
- Artifact/storage notes: `okf/concept-graphs-architecture/operations/artifacts-and-storage.md`
- Prompt profiles: `okf/concept-graphs-architecture/operations/prompt-profiles.md`
- Version management: `okf/concept-graphs-architecture/operations/version-management.md`
- Testing/validation: `okf/concept-graphs-architecture/operations/testing-and-validation.md`
