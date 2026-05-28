"""Flask route registration helpers."""

from src.api.routes.artifacts import register_artifact_routes
from src.api.routes.graph_documents import register_graph_document_routes
from src.api.routes.pipeline import register_pipeline_routes
from src.api.routes.processes import register_process_routes
from src.api.routes.rag import register_rag_routes
from src.api.routes.static import register_static_routes
from src.api.routes.status import register_status_routes


def register_routes(main_objects):
    """Register all API routes on the configured Flask app."""
    register_static_routes(main_objects)
    register_artifact_routes(main_objects)
    register_graph_document_routes(main_objects)
    register_pipeline_routes(main_objects)
    register_process_routes(main_objects)
    register_rag_routes(main_objects)
    register_status_routes(main_objects)
