"""Streamlit GUI for Concept Graphs.

Run with:
    streamlit run src/gui/app.py
"""

from __future__ import annotations

import html
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components

from src.gui.api_client import APIError, ConceptGraphsClient
from src.query_expansion.relations import MEDICAL_RELATION_DEFINITIONS

PIPELINE_STEPS = ["data", "embedding", "clustering", "graph", "integration"]
RELATIONS = [relation.id for relation in MEDICAL_RELATION_DEFINITIONS]
DEFAULT_RELATION_DEFINITIONS_BY_ID = {
    relation.id: relation.model_dump(mode="json")
    for relation in MEDICAL_RELATION_DEFINITIONS
}


def relation_definition_controls(
    selected_relations: list[str], categories: list[str]
) -> list[dict[str, Any]]:
    """Render user-friendly mini-ontology controls for selected relations."""
    relation_definitions = []
    if not selected_relations:
        st.info("Select at least one relation to configure relation definitions.")
        return relation_definitions

    for relation_id in selected_relations:
        default = DEFAULT_RELATION_DEFINITIONS_BY_ID.get(relation_id, {})
        with st.expander(f"Relation: {relation_id}", expanded=False):
            source_categories = st.multiselect(
                "Source categories",
                categories,
                default=[
                    category
                    for category in default.get("source_categories", [])
                    if category in categories
                ],
                key=f"qe_relation_{relation_id}_source_categories",
            )
            target_categories = st.multiselect(
                "Target categories",
                categories,
                default=[
                    category
                    for category in default.get("target_categories", [])
                    if category in categories
                ],
                key=f"qe_relation_{relation_id}_target_categories",
            )
            description = st.text_area(
                "Definition",
                value=default.get("description", ""),
                key=f"qe_relation_{relation_id}_description",
                height=80,
            )
            relation_definitions.append(
                {
                    "id": relation_id,
                    "source_categories": source_categories,
                    "target_categories": target_categories,
                    "description": description,
                }
            )
    return relation_definitions


def get_client() -> ConceptGraphsClient:
    return ConceptGraphsClient(st.session_state.api_base_url)


def show_error(exc: Exception) -> None:
    if isinstance(exc, APIError):
        st.error(str(exc))
        if exc.payload:
            st.caption("Response payload")
            st.json(exc.payload)
    else:
        st.error(str(exc))


def json_editor(label: str, value: Any, *, height: int = 360, key: str) -> Any | None:
    text = st.text_area(label, json.dumps(value, indent=2, ensure_ascii=False), height=height, key=key)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        return None


def process_progress(status_payload: dict[str, Any]) -> tuple[float, str]:
    statuses = status_payload.get("status", []) if isinstance(status_payload, dict) else []
    if not statuses:
        return 0.0, "No step status available yet."
    by_name = {entry.get("name"): entry.get("status") for entry in statuses}
    finished = sum(1 for step in PIPELINE_STEPS if by_name.get(step) == "finished")
    running = [step for step in PIPELINE_STEPS if by_name.get(step) in {"started", "running"}]
    failed = [step for step in PIPELINE_STEPS if by_name.get(step) in {"aborted", "stopped"}]
    progress = finished / len(PIPELINE_STEPS)
    if running:
        progress = max(progress, (PIPELINE_STEPS.index(running[0]) + 0.35) / len(PIPELINE_STEPS))
    if failed:
        return progress, f"Stopped/aborted at: {', '.join(failed)}"
    if finished == len(PIPELINE_STEPS):
        return 1.0, "Pipeline finished."
    if running:
        return progress, f"Currently running: {', '.join(running)}"
    return progress, f"Finished {finished}/{len(PIPELINE_STEPS)} steps."


def status_panel(process: str) -> None:
    try:
        payload = get_client().status(process)
    except Exception as exc:
        show_error(exc)
        return
    progress, text = process_progress(payload)
    st.progress(progress, text=text)
    st.json(payload)


def uploaded_file_tuple(uploaded_file: Any) -> tuple[str, io.BytesIO, str]:
    data = io.BytesIO(uploaded_file.getvalue())
    return (uploaded_file.name, data, uploaded_file.type or "application/octet-stream")


def pipeline_tab(process: str, language: str) -> None:
    st.subheader("Start complete pipeline")
    client = get_client()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Load default config"):
            try:
                loaded = client.pipeline_configuration(process, language, default=True)
                st.session_state.pipeline_config = loaded
                st.session_state.pipeline_config_text = json.dumps(loaded, indent=2, ensure_ascii=False)
            except Exception as exc:
                show_error(exc)
    with col_b:
        if st.button("Load process config"):
            try:
                loaded = client.pipeline_configuration(process, language, default=False)
                st.session_state.pipeline_config = loaded
                st.session_state.pipeline_config_text = json.dumps(loaded, indent=2, ensure_ascii=False)
            except Exception as exc:
                show_error(exc)
    with col_c:
        if st.button("Refresh status"):
            st.session_state.watch_pipeline = True

    st.markdown("#### Corpus input")
    input_mode = st.radio(
        "How should the corpus be provided?",
        ["Reference document server", "Upload ZIP corpus"],
        horizontal=True,
        help=(
            "The API currently supports either a ZIP upload or fetching documents "
            "from a configured document/index server."
        ),
    )

    default_config = st.session_state.get(
        "pipeline_config", {"name": process, "language": language, "config": {}}
    )
    default_doc_server = default_config.get("document_server", {})

    uploaded_zip = None
    config_uploads: dict[str, Any] = {}
    document_server_override: dict[str, Any] | None = None
    if input_mode == "Upload ZIP corpus":
        uploaded_zip = st.file_uploader(
            "Upload corpus ZIP",
            type=["zip"],
            help="ZIP file containing the text documents to process.",
        )
        with st.expander("Optional config file uploads"):
            st.caption(
                "These files map to the multipart `/pipeline` fields. Usually you can "
                "leave them empty for a first run."
            )
            for field in [
                "document_server_config",
                "vectorstore_server_config",
                "data_config",
                "embedding_config",
                "clustering_config",
                "graph_config",
            ]:
                config_uploads[field] = st.file_uploader(
                    field, type=["yml", "yaml", "json"], key=field
                )
    else:
        st.caption("Reference an existing document/index server as the pipeline corpus source.")
        server_cols = st.columns([2, 1, 2, 1])
        with server_cols[0]:
            ds_url = st.text_input("Document server URL", value=str(default_doc_server.get("url", "http://localhost")))
        with server_cols[1]:
            ds_port = st.number_input("Port", min_value=1, max_value=65535, value=int(default_doc_server.get("port", 9008)))
        with server_cols[2]:
            ds_index = st.text_input("Index / collection", value=str(default_doc_server.get("index", "documents")))
        with server_cols[3]:
            ds_size = st.number_input("Fetch size", min_value=1, value=int(default_doc_server.get("size", 30)))
        meta_cols = st.columns(3)
        with meta_cols[0]:
            ds_label = st.text_input("Label key", value=str(default_doc_server.get("label_key", "label")))
        with meta_cols[1]:
            ds_text_key = st.text_input("Server text field", value=str(default_doc_server.get("replace_keys", {}).get("text", "content")))
        with meta_cols[2]:
            ds_id_key = st.text_input("Document ID key", value=str(default_doc_server.get("other_id", "id")))
        document_server_override = {
            "url": ds_url,
            "port": int(ds_port),
            "index": ds_index,
            "size": int(ds_size),
            "label_key": ds_label or None,
            "replace_keys": {"text": ds_text_key},
            "other_id": ds_id_key or None,
        }

    st.markdown("#### Pipeline settings")
    skip_present = st.checkbox(
        "Reuse present artifacts",
        value=True,
        help="Whether steps that were already completed should be reused."
    )
    return_statistics = st.checkbox(
        "Wait and return statistics",
        value=False,
        help="Usually leave this disabled for long runs.",
    )
    skip_steps = st.multiselect("Skip steps", PIPELINE_STEPS)

    config = json_editor(
        "Advanced pipeline JSON configuration",
        default_config,
        key="pipeline_config_text",
    )

    if st.button("Start pipeline", type="primary"):
        try:
            if input_mode == "Upload ZIP corpus":
                if uploaded_zip is None:
                    st.warning("Please upload a ZIP file first.")
                    return
                data_file = uploaded_file_tuple(uploaded_zip)
                files = {
                    key: uploaded_file_tuple(value)
                    for key, value in config_uploads.items()
                    if value is not None
                }
                result = client.start_pipeline_upload(
                    process,
                    language,
                    data_file,
                    files,
                    skip_present,
                    skip_steps,
                    return_statistics,
                )
            else:
                if config is None:
                    return
                config["name"] = process
                config["language"] = language
                if document_server_override is not None:
                    config["document_server"] = document_server_override
                result = client.start_pipeline_json(
                    process,
                    language,
                    config,
                    skip_present,
                    skip_steps,
                    return_statistics,
                )
            st.success("Pipeline request submitted.")
            st.json(result)
            st.session_state.watch_pipeline = True
        except Exception as exc:
            show_error(exc)

    st.markdown("#### Pipeline status")
    auto_refresh = st.checkbox(
        "Poll status",
        value=st.session_state.get("watch_pipeline", False),
        help="Should the pipeline be periodically queried for its status via API?",
    )
    interval = st.slider("Poll interval (seconds)", 2, 30, 5)
    placeholder = st.empty()
    if auto_refresh:
        with placeholder.container():
            status_panel(process)
        time.sleep(interval)
        st.rerun()
    else:
        status_panel(process)


def graph_svg(graph_payload: dict[str, Any], width: int = 900, height: int = 650) -> str:
    graph = nx.Graph()
    for node in graph_payload.get("nodes", []):
        graph.add_node(str(node.get("id")), **node)
    for item in graph_payload.get("adjacency", []):
        source = str(item.get("id"))
        for neighbor in item.get("neighbors", []):
            target = str(neighbor.get("id"))
            if source != target:
                graph.add_edge(source, target, weight=neighbor.get("weight"))
    if not graph.nodes:
        return "<p>No graph nodes.</p>"
    pos = nx.spring_layout(graph, seed=42, k=1 / math.sqrt(max(len(graph), 1)))
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    def scale(value: float, low: float, high: float, size: int, pad: int = 40) -> float:
        return pad + (value - low) / ((high - low) or 1) * (size - 2 * pad)
    coords = {
        node: (scale(x, min(xs), max(xs), width), scale(y, min(ys), max(ys), height))
        for node, (x, y) in pos.items()
    }
    edges = []
    for source, target, attrs in graph.edges(data=True):
        x1, y1 = coords[source]
        x2, y2 = coords[target]
        title = html.escape(str(attrs.get("weight", "")))
        edges.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#999" stroke-width="1"><title>{title}</title></line>')
    nodes = []
    for node, attrs in graph.nodes(data=True):
        x, y = coords[node]
        label = html.escape(str(attrs.get("label") or attrs.get("name") or node))
        short = label[:28] + ("…" if len(label) > 28 else "")
        nodes.append(f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#4e79a7"><title>{label}</title></circle><text x="{x + 12:.1f}" y="{y + 4:.1f}" font-size="11">{short}</text></g>')
    return f'<div style="overflow:auto"><svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(edges)}{"".join(nodes)}</svg></div>'


def graphs_tab(process: str) -> None:
    st.subheader("Inspect graphs")
    client = get_client()
    if st.button("Load graph statistics"):
        try:
            st.session_state.graph_stats = client.artifact("/graph/statistics", process)
        except Exception as exc:
            show_error(exc)
    if stats := st.session_state.get("graph_stats"):
        st.json(stats)
        graphs = stats.get("conceptGraphs", []) if isinstance(stats, dict) else []
        ids = [g.get("id") for g in graphs]
    else:
        ids = list(range(5))
    graph_id = st.number_input("Graph ID", min_value=0, value=int(ids[0] if ids else 0), step=1)
    mode = st.radio("View mode", ["Local SVG", "API interactive HTML", "Raw JSON"], horizontal=True)
    if st.button("Load graph"):
        try:
            if mode == "API interactive HTML":
                st.session_state.graph_html = client.graph_html(process, int(graph_id))
                st.session_state.graph_payload = None
            else:
                st.session_state.graph_payload = client.artifact(f"/graph/{int(graph_id)}", process)
                st.session_state.graph_html = None
        except Exception as exc:
            show_error(exc)
    if st.session_state.get("graph_html"):
        components.html(st.session_state.graph_html, height=850, scrolling=True)
    if payload := st.session_state.get("graph_payload"):
        if mode == "Local SVG":
            components.html(graph_svg(payload), height=700, scrolling=True)
        with st.expander("Graph JSON", expanded=(mode == "Raw JSON")):
            st.json(payload)


def rag_tab(process: str, language: str) -> None:
    st.subheader("RAG")
    default = {
        "language": language,
        "api_key": "",
        "chatter": {"chatter": "src.rag.chatters.blablador.BlabladorChatter"},
        "prompt_template": {"profile": language},
        "vectorstore_server": {"url": "http://localhost", "port": 8882},
    }
    config = json_editor("RAG configuration", st.session_state.get("rag_config", default), key="rag_config_text", height=260)
    api_key = st.text_input("Provider API key (session-only, optional)", type="password")
    force = st.checkbox("Force reinitialize vector store", value=False)
    if st.button("Initialize RAG"):
        if config is None:
            return
        if api_key:
            config["api_key"] = api_key
        try:
            st.json(get_client().init_rag(process, config, force))
        except Exception as exc:
            show_error(exc)
    question = st.text_area("Question")
    doc_ids = st.text_input("Optional document IDs, comma-separated")
    limit = st.number_input("Chunk limit", min_value=1, max_value=100, value=15)
    if st.button("Ask RAG", type="primary") and question.strip():
        try:
            ids = [item.strip() for item in doc_ids.split(",") if item.strip()]
            result = get_client().ask_rag(process, question.strip(), ids, int(limit))
            st.markdown("#### Answer")
            st.write(result.get("answer", result))
            info = result.get("info")
            if info:
                with st.expander("Sources"):
                    st.json(json.loads(info) if isinstance(info, str) else info)
        except Exception as exc:
            show_error(exc)


def query_expansion_tab(language: str) -> None:
    st.subheader("Query expansion")
    term = st.text_input("Term to expand")
    try:
        profiles_payload = get_client().query_expansion_profiles()
        profiles = profiles_payload.get("profiles", [])
    except Exception as exc:
        st.error(f"Could not load query-expansion profiles from API: {exc}")
        profiles = []
    profile_names = [profile.get("name") for profile in profiles if profile.get("name")]
    if not profile_names:
        st.warning("No query-expansion domain profiles are available from the API.")
        return
    default_profile_index = profile_names.index(language) if language in profile_names else 0
    profile = st.selectbox(
        "Domain profile",
        profile_names,
        index=default_profile_index,
        help="Profiles are loaded by the API from conf/query-expansion/profiles/.",
    )
    selected_profile = next(
        item for item in profiles if item.get("name") == profile
    )
    profile_categories = [
        category["id"] for category in selected_profile.get("categories", [])
    ]
    default_selected_categories = [
        category
        for category in selected_profile.get("default_categories", [])
        if category in profile_categories
    ] or profile_categories[:3]
    selected = st.multiselect(
        "Categories",
        profile_categories,
        default=default_selected_categories,
        help="Semantic categories from the selected domain profile.",
    )
    selected_relations = st.multiselect(
        "Relations",
        RELATIONS,
        default=[
            "equivalent_to",
            "may_indicate",
            "treated_by",
            "investigated_by",
            "confirmed_by",
        ],
        help="Backend-neutral semantic relations the LLM may use.",
    )
    st.markdown("##### Relation definitions / mini-ontology")
    st.caption(
        "For each selected relation, define which source and target categories it "
        "may connect. This is semantic structure only; no search-engine behavior "
        "is configured here."
    )
    relation_definitions = relation_definition_controls(
        selected_relations, profile_categories
    )
    relation_categories = {
        category
        for definition in relation_definitions
        for category in (
            definition.get("source_categories", [])
            + definition.get("target_categories", [])
        )
    }
    missing_categories = sorted(relation_categories - set(selected))
    if missing_categories:
        st.warning(
            "Some relation definitions reference categories that are not selected: "
            f"{', '.join(missing_categories)}. Relations using those categories "
            "will be filtered unless you select the categories too."
        )
    with st.expander("Relation definitions JSON preview"):
        st.json(relation_definitions)
    limit = st.number_input("Limit per category", min_value=1, max_value=100, value=5)
    provider = st.selectbox("LLM provider", ["ollama", "blablador", "openai"])
    model = st.text_input(
        "Model",
        value="llama3.1" if provider == "ollama" else "alias-fast",
    )
    base_url = st.text_input(
        "Base URL",
        value="http://localhost:11434" if provider == "ollama" else "",
    )
    api_key = st.text_input(
        "Provider API key (session-only, optional)",
        type="password",
        key="qe_key",
    )
    include_llm_only = st.checkbox("Include LLM-only expansions", value=True)
    minimum_score = st.slider("Minimum grounding score", 0.0, 1.0, 0.0)
    reject_below = st.checkbox("Reject below minimum", value=False)
    sources_text = st.text_area(
        "Grounding sources JSON array",
        value='[]',
        help=(
            'Example: [{"name":"local-medical-terms","type":"local",'
            '"path":"conf/query-expansion/grounding/medical_terms.example.yml"}]'
        ),
        height=120,
    )
    custom_payload = st.checkbox("Edit complete JSON before sending")
    try:
        sources = json.loads(sources_text or "[]")
    except json.JSONDecodeError as exc:
        st.error(f"Invalid sources JSON: {exc}")
        sources = []
    payload = {
        "term": term,
        "language": language,
        "categories": selected,
        "relations": selected_relations,
        "relation_definitions": relation_definitions or [],
        "limit_per_category": int(limit),
        "llm": {"model": model, "options": {"provider": provider}},
        "sources": sources,
        "grounding": {
            "include_llm_only": include_llm_only,
            "minimum_score": minimum_score,
            "reject_below_minimum": reject_below,
        },
        "prompt": {"profile": profile or None},
    }
    if base_url:
        payload["llm"]["options"]["base_url"] = base_url
    if custom_payload:
        edited = json_editor("Query expansion request", payload, key="qe_payload_text")
        if edited is not None:
            payload = edited
    with st.expander("Request preview"):
        st.json(payload)
    if st.button("Expand query", type="primary"):
        if not payload.get("term"):
            st.warning("Enter a term first.")
            return
        try:
            st.session_state.qe_result = get_client().expand_query(
                payload,
                api_key=api_key or None,
            )
        except Exception as exc:
            show_error(exc)
    if result := st.session_state.get("qe_result"):
        tabs = st.tabs(["Expansions", "Concepts", "Relations", "Raw JSON"])
        with tabs[0]:
            st.json(result.get("expansions", result))
        with tabs[1]:
            st.json(result.get("concepts", []))
        with tabs[2]:
            st.json(result.get("relations", []))
        with tabs[3]:
            st.json(result)


def sidebar() -> tuple[str, str]:
    st.sidebar.header("Connection")
    st.session_state.setdefault("api_base_url", "http://localhost:9010")
    st.session_state.api_base_url = st.sidebar.text_input("API base URL", st.session_state.api_base_url)
    process = st.sidebar.text_input("Process", value=st.session_state.get("process", "default"))
    language = st.sidebar.selectbox("Language", ["en", "de"], index=0 if st.session_state.get("language", "en") == "en" else 1)
    st.session_state.process = process
    st.session_state.language = language
    if st.sidebar.button("Check connection"):
        try:
            get_client().health()
            st.sidebar.success("Connected")
        except Exception as exc:
            st.sidebar.error(str(exc))
    st.sidebar.divider()
    if st.sidebar.button("List processes"):
        try:
            st.session_state.processes = get_client().list_processes()
        except Exception as exc:
            st.session_state.processes = str(exc)
    if "processes" in st.session_state:
        st.sidebar.json(st.session_state.processes)
    hard_stop = st.sidebar.checkbox("Hard stop/delete", value=False)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Stop"):
            try:
                st.sidebar.write(get_client().stop_process(process, hard_stop))
            except Exception as exc:
                st.sidebar.error(str(exc))
    with col2:
        if st.button("Delete"):
            try:
                st.sidebar.write(get_client().delete_process(process, hard_stop))
            except Exception as exc:
                st.sidebar.error(str(exc))
    return process, language


def main() -> None:
    st.set_page_config(page_title="Concept Graphs GUI", layout="wide")
    st.title("Concept Graphs")
    process, language = sidebar()
    tabs = st.tabs(["Pipeline", "Graphs", "RAG", "Query Expansion"])
    with tabs[0]:
        pipeline_tab(process, language)
    with tabs[1]:
        graphs_tab(process)
    with tabs[2]:
        rag_tab(process, language)
    with tabs[3]:
        query_expansion_tab(language)


if __name__ == "__main__":
    main()
