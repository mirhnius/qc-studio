from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
from niivue_component import niivue_viewer
from utils import parse_qc_config, load_mri_data, load_svg_data, save_qc_results_to_csv
from models import MetricQC, QCRecord
from constants import (
    EXPERIENCE_LEVELS, FATIGUE_LEVELS, DEFAULT_PANELS, PANEL_CONFIG,
    QC_RATINGS, DEFAULT_QC_RATING, NIIVUE_HEIGHT, SVG_HEIGHT, VIEW_MODES,
    OVERLAY_COLORMAPS, DEFAULT_OVERLAY_OPACITY, EQUAL_RATIO,
    RATING_IQM_RATIO, RATER_INFO_RATIO, UPLOAD_FILE_TYPES, MESSAGES, ERROR_MESSAGES,
    SUCCESS_MESSAGES, INFO_MESSAGES, SUBSTITUTIONS_DICT
)
from session_manager import SessionManager
from niivue_viewer_manager import NiivueViewerManager, NiivueViewerConfig
from panel_layout_manager import PanelLayoutManager
from landing_page import show_landing_page
from congratulations_page import show_congratulations_page
from qc_viewer import display_qc_viewers
from pagination import display_qc_rating_and_pagination


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

	# Middle: QC Viewers (Niivue, SVG, IQM)
	middle = st.container()
	with middle:
		display_qc_viewers(
			dataset_dir=dataset_dir,			
			qc_config=qc_config,
			participant_id=participant_id,
			session_id=session_id,
			qc_pipeline=qc_pipeline,
			qc_task=qc_task,
			total_participants=total_participants
		)

	# Bottom: QC Rating and Pagination
	display_qc_rating_and_pagination(		
		participant_id=participant_id,
		session_id=session_id,
		qc_pipeline=qc_pipeline,
		qc_task=qc_task,
		total_participants=total_participants
	)
				
                
