"""
CropShield AI - Utility Modules
================================

Explainability and visualization utilities.
"""

from .gradcam import (
    GradCAM,
    generate_gradcam,
    generate_gradcam_visualization,
    get_target_layer,
    visualize_gradcam_grid,
    compare_gradcam_predictions,
    save_gradcam,
    get_colormap_options
)

from .app_utils import (
    # Class mapping
    load_class_names,
    load_class_mapping,
    
    # Image handling
    bytesio_to_pil,
    uploaded_file_to_pil,
    pil_to_bytes,
    resize_image,
    
    # Prediction visualization
    display_predictions,
    format_prediction_table,
    
    # GradCAM visualization
    show_gradcam_overlay,
    show_gradcam_grid,
    show_gradcam_comparison,
    
    # Utilities
    create_confidence_indicator,
    save_prediction_history,
    get_crop_emoji,
)

__all__ = [
    # GradCAM
    'GradCAM',
    'generate_gradcam',
    'generate_gradcam_visualization',
    'get_target_layer',
    'visualize_gradcam_grid',
    'compare_gradcam_predictions',
    'save_gradcam',
    'get_colormap_options',
    
    # App utilities
    'load_class_names',
    'load_class_mapping',
    'bytesio_to_pil',
    'uploaded_file_to_pil',
    'pil_to_bytes',
    'resize_image',
    'display_predictions',
    'format_prediction_table',
    'show_gradcam_overlay',
    'show_gradcam_grid',
    'show_gradcam_comparison',
    'create_confidence_indicator',
    'save_prediction_history',
    'get_crop_emoji',
]
