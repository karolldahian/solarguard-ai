"""Capa de presentacion de SolarGuard AI (Streamlit)."""

from solarguard_ai.view.alertas import (
    render_ticket_result,
    render_ticket_section,
    validate_panel_id,
)
from solarguard_ai.view.dashboard import (
    build_class_distribution_chart,
    build_priority_distribution_chart,
    compute_dashboard_stats,
    render_dashboard,
)
from solarguard_ai.view.diagnostico import (
    render_configuration_error,
    render_diagnosis,
    render_model_unavailable,
    render_prediction,
    render_priority,
    render_probabilities,
)
from solarguard_ai.view.historial import (
    AnalysisRecord,
    add_analysis,
    build_analysis_record,
    compute_image_fingerprint,
    get_history,
    normalize_priority,
    register_analysis,
)
from solarguard_ai.view.modelo import (
    get_inference_service,
    get_priority_config,
    resolve_model_path,
    resolve_priority_config_path,
)
from solarguard_ai.view.pagina_principal import (
    render_header,
    render_image_metadata,
    render_image_preview,
    upload_image,
)

__all__ = [
    "AnalysisRecord",
    "add_analysis",
    "build_analysis_record",
    "build_class_distribution_chart",
    "build_priority_distribution_chart",
    "compute_dashboard_stats",
    "compute_image_fingerprint",
    "get_history",
    "get_inference_service",
    "get_priority_config",
    "normalize_priority",
    "register_analysis",
    "render_configuration_error",
    "render_dashboard",
    "render_diagnosis",
    "render_header",
    "render_image_metadata",
    "render_image_preview",
    "render_model_unavailable",
    "render_prediction",
    "render_priority",
    "render_probabilities",
    "render_ticket_result",
    "render_ticket_section",
    "resolve_model_path",
    "resolve_priority_config_path",
    "upload_image",
    "validate_panel_id",
]
