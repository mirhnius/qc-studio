from pathlib import Path
import pandas as pd
import streamlit as st
from niivue_component import niivue_viewer
from utils.config import parse_qc_config
from utils.data_loaders import load_svg_data
from utils.export import save_qc_results_to_csv
from models import QCRecord
from constants import (
    EXPERIENCE_LEVELS, FATIGUE_LEVELS, DEFAULT_PANELS, PANEL_CONFIG,
    QC_RATINGS, DEFAULT_QC_RATING, NIIVUE_HEIGHT, SVG_HEIGHT, VIEW_MODES,
    OVERLAY_COLORMAPS, DEFAULT_OVERLAY_OPACITY, EQUAL_RATIO,
    RATING_IQM_RATIO, RATER_INFO_RATIO, UPLOAD_FILE_TYPES, MESSAGES, ERROR_MESSAGES,
    SUCCESS_MESSAGES, INFO_MESSAGES, SUBSTITUTIONS_DICT
)
from managers.session_manager import SessionManager
from managers.niivue_viewer_manager import NiivueViewerManager, NiivueViewerConfig
from managers.panel_layout_manager import PanelLayoutManager
from pages.landing_page import show_landing_page
from pages.congratulations_page import show_congratulations_page
from components.qc_viewer import display_qc_viewers


def app(dataset_dir, participant_id, session_id, qc_pipeline, qc_task, qc_config_path, out_dir, total_participants, drop_duplicates, participant_list) -> None:
	"""Main Streamlit layout: landing page, QC viewers, and congratulations."""
	st.set_page_config(layout="wide")

	# Initialize session state
	SessionManager.init_session_state()

	# Check if we're on the landing page
	if not SessionManager.is_landing_page_complete():
		show_landing_page(qc_pipeline, qc_task, out_dir, participant_list)
		return

	# Check if we're on the final congratulations page
	if participant_id is None:
		show_congratulations_page(qc_task, out_dir, total_participants, drop_duplicates)
		return

	# parse qc config
	substitution_values = {
		'participant_id': participant_id,
		'session_id': session_id
	}
	qc_config = parse_qc_config(qc_config_path, qc_task, substitution_values) 

	# Display QC Viewers with integrated pagination in left sidebar
	display_qc_viewers(
		dataset_dir=dataset_dir,			
		qc_config=qc_config,
		participant_id=participant_id,
		session_id=session_id,
		qc_pipeline=qc_pipeline,
		qc_task=qc_task,
		total_participants=total_participants
	)
				
                
