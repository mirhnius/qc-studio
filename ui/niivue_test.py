import os
import streamlit as st
from niivue_component import niivue_viewer
# from niivue_component import niivue_component


def niivue_viewer_from_path(filepath: str, height: int = 600, key: str | None = None) -> None:
    """Load a local NIFTI file from `filepath` and display it with `niivue_viewer`.

    This helper reads the file bytes and calls the existing component.

    Args:
        filepath: Path to a local .nii or .nii.gz file.
        height: Viewer height in pixels (default 600).
        key: Optional Streamlit key for the component instance.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        file_bytes = f.read()

    if key is None:
        key = f"niivue_viewer_path_{os.path.basename(filepath)}"

    niivue_viewer(
        nifti_data=file_bytes,        
        filename=os.path.basename(filepath),
        height=height,
        key=key,
    )


# Text input + button to let users specify a local filepath directly
# filepath_input = st.text_input("/home/nikhil/projects/neuroinformatics_tools/sandbox/qc-studio/sample_data/bids/sub-ED01/ses-BL/anat/sub-ED01_ses-BL_T1w.nii.gz")
filepath_input = "/home/nikhil/projects/neuroinformatics_tools/sandbox/qc-studio/sample_data/bids/sub-ED01/ses-BL/anat/sub-ED01_ses-BL_T1w.nii.gz"
# filepath_input = "/home/nikhil/projects/neuroinformatics_tools/sandbox/qc-studio/sample_data/derivatives/freesurfer/7.3.2/output/ses-BL/sub-ED01/surf/rh.midthickness"
# filepath_input = "/home/nikhil/projects/neuroinformatics_tools/sandbox/qc-studio/sample_data/derivatives/freesurfer/7.3.2/output/ses-BL/sub-ED01/mri/aparc+aseg.mgz"

try:
    niivue_viewer_from_path(filepath_input, height=600, key="niivue_viewer_path")

except Exception as e:
    st.error(f"Failed to load file: {e}")



