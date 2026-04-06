"""Niivue viewer configuration and rendering utilities."""
import streamlit as st
from constants import (
    NIIVUE_HEIGHT, VIEW_MODES, OVERLAY_COLORMAPS, DEFAULT_OVERLAY_OPACITY,
    MESSAGES, ERROR_MESSAGES
)
from utils import load_mri_data
from niivue_component import niivue_viewer


class NiivueViewerConfig:
    """Configuration container for Niivue viewer settings."""
    
    def __init__(self, view_mode: str, overlay_colormap: str, 
                 show_crosshair: bool, radiological: bool,
                 show_colorbar: bool, interpolation: bool,
                 show_overlay: bool):
        """Initialize Niivue viewer configuration.
        
        Args:
            view_mode: Viewing perspective (multiplanar, axial, coronal, sagittal, 3d)
            overlay_colormap: Colormap for overlay (grey, cool, warm)
            show_crosshair: Whether to show crosshair
            radiological: Whether to use radiological convention
            show_colorbar: Whether to show colorbar
            interpolation: Whether to use interpolation
            show_overlay: Whether to show overlay image
        """
        self.view_mode = view_mode
        self.overlay_colormap = overlay_colormap
        self.show_crosshair = show_crosshair
        self.radiological = radiological
        self.show_colorbar = show_colorbar
        self.interpolation = interpolation
        self.show_overlay = show_overlay
    
    def to_settings_dict(self) -> dict:
        """Convert to Niivue settings dictionary."""
        return {
            "crosshair": self.show_crosshair,
            "radiological": self.radiological,
            "colorbar": self.show_colorbar,
            "interpolation": self.interpolation
        }
    
    def get_viewer_key(self) -> str:
        """Generate unique key for viewer state based on settings."""
        return f"niivue_{self.view_mode}_{self.overlay_colormap}_{self.show_overlay}"


class NiivueViewerManager:
    """Manages Niivue viewer rendering and configuration."""
    
    @staticmethod
    def render_controls_panel() -> NiivueViewerConfig:
        """Render Niivue controls panel and return configuration.
        
        Returns:
            NiivueViewerConfig object with user selections
        """
        st.header(MESSAGES['niivue_controls_header'])
        
        # View mode selection
        view_mode = st.selectbox(
            MESSAGES['view_mode_label'],
            VIEW_MODES,
            help="Select the viewing perspective"
        )
        
        # Overlay colormap selection
        overlay_colormap = st.selectbox(
            MESSAGES['overlay_colormap_label'],
            OVERLAY_COLORMAPS,
            help="Select the colormap for the overlay"
        )
        
        st.divider()
        st.subheader(MESSAGES['display_settings_header'])
        
        # Display settings checkboxes
        show_crosshair = st.checkbox(MESSAGES['crosshair_label'], value=False)
        radiological = st.checkbox(MESSAGES['radiological_label'], value=False)
        show_colorbar = st.checkbox(MESSAGES['colorbar_label'], value=True)
        interpolation = st.checkbox(MESSAGES['interpolation_label'], value=True)
        
        # Overlay toggle
        show_overlay = st.checkbox(MESSAGES['show_overlay_label'], value=False)
        
        return NiivueViewerConfig(
            view_mode=view_mode,
            overlay_colormap=overlay_colormap,
            show_crosshair=show_crosshair,
            radiological=radiological,
            show_colorbar=show_colorbar,
            interpolation=interpolation,
            show_overlay=show_overlay
        )
    
    @staticmethod
    def build_overlay_list(mri_data: dict, config: NiivueViewerConfig) -> list:
        """Build overlay configuration list based on settings.
        
        Args:
            mri_data: MRI data dictionary from load_mri_data()
            config: NiivueViewerConfig with overlay settings
            
        Returns:
            List of overlay configurations (empty if no overlay)
        """
        if not config.show_overlay or "overlay_mri_image_bytes" not in mri_data:
            return []
        
        return [{
            "data": mri_data["overlay_mri_image_bytes"],
            "name": "overlay",
            "colormap": config.overlay_colormap,
            "opacity": DEFAULT_OVERLAY_OPACITY,
        }]
    
    @staticmethod
    def build_viewer_kwargs(mri_data: dict, config: NiivueViewerConfig) -> dict:
        """Build kwargs dictionary for niivue_viewer component.
        
        Args:
            mri_data: MRI data dictionary from load_mri_data()
            config: NiivueViewerConfig with viewer settings
            
        Returns:
            Dictionary of kwargs for niivue_viewer()
        """
        base_mri_image_bytes = mri_data.get("base_mri_image_bytes")
        base_mri_image_path = mri_data.get("base_mri_image_path")
        
        base_mri_name = str(base_mri_image_path.name) if base_mri_image_path else "base_mri.nii"
        settings = config.to_settings_dict()
        overlays = NiivueViewerManager.build_overlay_list(mri_data, config)
        
        viewer_kwargs = {
            "nifti_data": base_mri_image_bytes,
            "filename": base_mri_name,
            "height": NIIVUE_HEIGHT,
            "key": config.get_viewer_key(),
            "view_mode": config.view_mode,
            "settings": settings,
            "styled": True,
        }
        
        if overlays:
            viewer_kwargs["overlays"] = overlays
        
        return viewer_kwargs
    
    @staticmethod
    def render_viewer(dataset_dir, qc_config, config: NiivueViewerConfig):
        """Render Niivue viewer in the main viewing area.
        
        Args:
            qc_config: QC configuration object
            config: NiivueViewerConfig with viewer settings
        """
        st.header(MESSAGES['niivue_header'])
        
        try:
            # Load MRI data
            mri_data = load_mri_data(dataset_dir, qc_config)
            
            if "base_mri_image_bytes" not in mri_data:
                st.info(ERROR_MESSAGES['base_mri_not_found'])
                return
            
            # Build and render viewer
            viewer_kwargs = NiivueViewerManager.build_viewer_kwargs(mri_data, config)
            niivue_viewer(**viewer_kwargs)
            
        except Exception as e:
            st.error(ERROR_MESSAGES['mri_load_error'].format(error=e))
