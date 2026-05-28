"""Flask route registration helpers."""

from src.api.routes.artifacts import register_artifact_routes
from src.api.routes.graph_documents import register_graph_document_routes
from src.api.routes.pipeline import register_pipeline_routes
from src.api.routes.processes import register_process_routes
from src.api.routes.rag import register_rag_routes
from src.api.routes.static import create_static_blueprint
from src.api.routes.status import create_status_blueprint


def register_routes(app_context):
    """Register all API routes on the configured Flask app."""
    app_context.app.register_blueprint(create_static_blueprint(app_context))
    register_artifact_routes(app_context)
    register_graph_document_routes(app_context)
    register_pipeline_routes(app_context)
    register_process_routes(app_context)
    register_rag_routes(app_context)
    app_context.app.register_blueprint(create_status_blueprint(app_context))
