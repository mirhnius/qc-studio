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

	# Top container: participant info
	top = st.container()
	with top:
		st.title(MESSAGES['qc_title'])
		
		# Display rater info summary
		col1, col2, col3 = st.columns(RATER_INFO_RATIO)
		with col1:
			st.metric("Rater", SessionManager.get_rater_id())
		with col2:
			st.metric("Experience", SessionManager.get_rater_experience().split('(')[0].strip())
		with col3:
			st.metric("Fatigue Level", SessionManager.get_rater_fatigue().split('☕')[0].strip())

		col_participant_info, col_pipe_info = st.columns(2)
		with col_participant_info:
			st.write(f"### Participant: {participant_id} | Session: {session_id}")
		with col_pipe_info:
			st.write(f"### Pipeline: {qc_pipeline} | Task: {qc_task}")
			
	
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
				
                
