"""QC viewer component for displaying MRI, SVG, and metrics panels."""
import streamlit as st
from constants import SVG_HEIGHT, MESSAGES, ERROR_MESSAGES, QC_RATINGS, NIIVUE_SECONDARY_RATIO
from utils import load_svg_data
from niivue_viewer_manager import NiivueViewerManager
from session_manager import SessionManager
from models import QCRecord
from datetime import datetime


def display_qc_viewers(
	dataset_dir,
	qc_config,
	participant_id: str = None,
	session_id: str = None,
	qc_pipeline: str = None,
	qc_task: str = None,
	total_participants: int = None
) -> None:
	"""Display QC viewers (Niivue, SVG, IQM panels) based on user selection.
	
	Layout strategy:
	- If all three panels (Niivue + SVG + IQM): 3-column (controls | Niivue | SVG), then IQM and rating in 2-columns
	- If Niivue + SVG selected: 3-column layout (controls | Niivue | SVG)
	- If SVG only selected: Full-width SVG
	- If Niivue + IQM selected: 3-column layout (controls | Niivue | IQM)
	- If Niivue only selected: Full-width Niivue
	
	Args:
		qc_config: QC configuration object
	"""
	st.container()
	
	# Get selected panels and normalize naming for backward compatibility
	selected_panels = SessionManager.get_selected_panels()
	selected_panels = {
		'niivue': selected_panels.get('niivue_col', selected_panels.get('niivue', True)),
		'svg': selected_panels.get('svg_col', selected_panels.get('svg', True)),
		'iqm': selected_panels.get('iqm_col', selected_panels.get('iqm', False))
	}
	
	show_niivue = selected_panels.get('niivue', True)
	show_svg = selected_panels.get('svg', True)
	show_iqm = selected_panels.get('iqm', False)
	
	# All three panels selected: 3-column on top, IQM and QC rating in 2-columns below
	if show_niivue and show_svg and show_iqm:
		_display_niivue_with_secondary_panel(dataset_dir, selected_panels, qc_config)
		st.divider()
		_display_iqm_and_rating_side_by_side(
			dataset_dir=dataset_dir,
			participant_id=participant_id,
			session_id=session_id,
			qc_pipeline=qc_pipeline,
			qc_task=qc_task,
			total_participants=total_participants
	)
	# 3-column layout: Niivue + SVG (no IQM)
	elif show_niivue and show_svg:
		_display_niivue_with_secondary_panel(dataset_dir, selected_panels, qc_config)
	# 3-column layout: Niivue + IQM (no SVG)
	elif show_niivue and show_iqm:
		_display_niivue_with_secondary_panel(dataset_dir, selected_panels, qc_config)
	# Full-width Niivue only
	elif show_niivue:
		_display_niivue_full_width(qc_config)
	# Full-width SVG only
	elif show_svg:
		_display_svg_panel(dataset_dir, qc_config)
	# Full-width IQM only
	elif show_iqm:
		_display_iqm_panel()


def _display_niivue_with_secondary_panel(dataset_dir, selected_panels: dict, qc_config) -> None:
	"""Display 3-column layout: controls | Niivue viewer | Secondary panel (SVG or IQM).
	
	Used when Niivue is selected with either SVG or IQM panel.
	
	Args:
		dataset_dir: Root dataset directory
		selected_panels: Dictionary of selected panels
		qc_config: QC configuration object
	"""
	ctrl_col, viewer_col, panel_col = st.columns(NIIVUE_SECONDARY_RATIO, gap="small")
	
	# Left column: Niivue controls
	with ctrl_col:
		niivue_config = NiivueViewerManager.render_controls_panel()
	
	# Middle column: Niivue viewer (header rendered by render_viewer)
	with viewer_col:
		NiivueViewerManager.render_viewer(dataset_dir, qc_config, niivue_config)
	
	# Right column: SVG or IQM panel
	with panel_col:
		if selected_panels.get('svg', False):
			_display_svg_panel(dataset_dir, qc_config)
		else:
			_display_iqm_panel()


def _display_niivue_full_width(qc_config) -> None:
	"""Display Niivue in full width with controls on the left.
	
	Args:
		qc_config: QC configuration object
	"""
	left_col, right_col = st.columns([0.32, 0.68], gap="small")
	
	with left_col:
		niivue_config = NiivueViewerManager.render_controls_panel()
	
	with right_col:
		NiivueViewerManager.render_viewer(dataset_dir, qc_config, niivue_config)


def _display_svg_panel(dataset_dir, qc_config) -> None:
	"""Display SVG/PNG/JPEG montage panel with tabs for multiple images.
	
	If multiple image files are available, renders them as separate tabs.
	If only one image file is available, displays it directly.
	
	Supports:
	- SVG: Rendered as HTML
	- PNG/JPEG: Displayed as images using st.image()
	
	Args:
		dataset_dir: Root dataset directory
		qc_config: QC configuration object
	"""
	st.header(MESSAGES['svg_header'])
	image_data = load_svg_data(dataset_dir, qc_config)
	
	if image_data:
		# If multiple images, create tabs
		if len(image_data) > 1:
			tabs = st.tabs(list(image_data.keys()))
			for tab, (filename, data) in zip(tabs, image_data.items()):
				with tab:
					_render_image(data, filename)
		else:
			# Single image - display directly
			filename, data = list(image_data.items())[0]
			_render_image(data, filename)
	else:
		st.info(ERROR_MESSAGES['svg_not_found'])


def _render_image(image_data: dict, filename: str) -> None:
	"""Render a single image (SVG, PNG, or JPEG) in Streamlit.
	
	Args:
		image_data: Dict with keys 'type' and 'content'
		filename: Name of the image file for display
	"""
	image_type = image_data.get("type")
	content = image_data.get("content")
	
	if image_type == "svg":
		# Render SVG as HTML
		st.components.v1.html(content, height=SVG_HEIGHT, scrolling=True)
	elif image_type in ["png", "jpeg"]:
		# Display PNG/JPEG as image
		st.image(content, use_container_width=True, caption=filename)
	else:
		st.warning(f"Unsupported image type: {image_type}")


def _display_iqm_and_rating_side_by_side(
	dataset_dir,
	participant_id: str = None,
	session_id: str = None,
	qc_pipeline: str = None,
	qc_task: str = None,
	total_participants: int = None
) -> None:
	"""Display IQM metrics panel in 2-column layout.
	
	Left column shows IQM metrics. Right column shows QC rating form.
	
	Args:
		dataset_dir: Root dataset directory
		participant_id: Current participant ID
		session_id: Current session ID
		qc_pipeline: QC pipeline name
		qc_task: QC task name
		total_participants: Total number of participants
	"""
	metrics_col, rating_col = st.columns([0.5, 0.5], gap="small")
	
	# Left column: IQM metrics
	with metrics_col:
		_display_iqm_panel()
	
	# Right column: QC rating form
	with rating_col:
		st.subheader(MESSAGES['qc_rating_header'])
		rating = st.radio(MESSAGES['qc_rating_prompt'], options=QC_RATINGS, index=0, key="side_by_side_rating")
		notes = st.text_area(MESSAGES['qc_notes_prompt'], value=SessionManager.get_notes(), key="side_by_side_notes", height=120)
		SessionManager.set_notes(notes)
		
		# Save button
		if st.button(MESSAGES['save_csv_button'], width='stretch', key="side_by_side_save"):
			_save_qc_record(
				participant_id=participant_id,
				session_id=session_id,
				qc_pipeline=qc_pipeline,
				qc_task=qc_task,
				rating=rating,
				notes=notes,
				total_participants=total_participants
			)


def _display_iqm_panel() -> None:
	"""Display IQM metrics panel."""
	st.subheader(MESSAGES['metrics_header'])
	st.write("Add QC metrics here (e.g., SNR, motion). This is a placeholder area.")


def _save_qc_record(participant_id: str, session_id: str, qc_pipeline: str, 
					 qc_task: str, rating: str, notes: str, total_participants: int) -> None:
	"""Save a QC record and mark as complete.
	
	Args:
		participant_id: Participant ID
		session_id: Session ID
		qc_pipeline: QC pipeline name
		qc_task: QC task name
		rating: QC rating value
		notes: QC notes
		total_participants: Total participants (used to detect end of QC)
	"""
	now = datetime.now()
	timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
	
	record = QCRecord(
		participant_id=participant_id,
		session_id=session_id,
		qc_task=qc_task,
		pipeline=qc_pipeline,
		timestamp=timestamp,
		rater_id=SessionManager.get_rater_id(),
		rater_experience=SessionManager.get_rater_experience(),
		rater_fatigue=SessionManager.get_rater_fatigue(),
		final_qc=rating,
		notes=notes,
	)
	
	SessionManager.add_qc_record(record)
	SessionManager.set_current_page(total_participants + 1)
	st.rerun()
