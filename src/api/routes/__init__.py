"""Flask route registration helpers."""

from src.api.routes.artifacts import create_artifact_blueprint
from src.api.routes.graph_documents import create_graph_document_blueprint
from src.api.routes.pipeline import create_pipeline_blueprint
from src.api.routes.processes import create_process_blueprint
from src.api.routes.rag import create_rag_blueprint
from src.api.routes.static import create_static_blueprint
from src.api.routes.status import create_status_blueprint


def register_routes(app_context):
    """Register all API routes on the configured Flask app."""
    app_context.app.register_blueprint(create_static_blueprint(app_context.app))
    app_context.app.register_blueprint(
        create_artifact_blueprint(
            app_context.app, app_context.storage, app_context.pipeline
        )
    )
    app_context.app.register_blueprint(
        create_graph_document_blueprint(
            app_context.app,
            app_context.processes,
            app_context.pipeline,
            app_context.storage,
        )
    )
    app_context.app.register_blueprint(
        create_pipeline_blueprint(
            app_context.app,
            app_context.processes,
            app_context.pipeline,
            app_context.storage,
        )
    )
    app_context.app.register_blueprint(
        create_process_blueprint(
            app_context.app,
            app_context.processes,
            app_context.pipeline,
            app_context.storage,
        )
    )
    app_context.app.register_blueprint(
        create_rag_blueprint(
            app_context.rag,
            app_context.processes,
            app_context.storage,
            app_context.pipeline,
        )
    )
    app_context.app.register_blueprint(
        create_status_blueprint(app_context.app, app_context.rag)
    )
