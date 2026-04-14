"""Landing page component for QC-Studio UI."""
import pandas as pd
import streamlit as st
from constants import (
    EXPERIENCE_LEVELS, FATIGUE_LEVELS, PANEL_CONFIG, UPLOAD_FILE_TYPES,
    MESSAGES, ERROR_MESSAGES, SUCCESS_MESSAGES, INFO_MESSAGES, SVG_HEIGHT,
    MIN_MONTAGE_GRID_SIZE, MAX_MONTAGE_GRID_SIZE
)
from session_manager import SessionManager
from models import QCRecord
from panel_layout_manager import PanelLayoutManager
from niivue_viewer_manager import NiivueViewerManager
from utils import load_svg_data


def show_landing_page(qc_pipeline, qc_task, out_dir, participant_list) -> None:
	"""Display the landing page with rater info, panel selection, and CSV upload.
	
	Args:
		qc_pipeline: QC pipeline name
		qc_task: QC task name
		out_dir: Output directory path
		participant_list: Path to participant list file
	"""
	st.title(MESSAGES['welcome_title'])

	# Load participant list to get total unique participants
	try:
		participants_df = pd.read_csv(participant_list, delimiter="\t")
		total_participants_in_ds = len(participants_df['participant_id'].unique())
		participant_ids_in_ds = set(participants_df['participant_id'].unique())
	except Exception as e:
		st.error(ERROR_MESSAGES['participant_list_load_error'].format(error=e))
		return

	st.subheader(f"QC Pipeline: {qc_pipeline} | QC Task: {qc_task} | n_ds_participants: {total_participants_in_ds}")
	
	st.markdown("---")
	
	# Three-column layout for rater info, panel selection, and CSV upload
	col1, col2, col3 = st.columns([1, 1, 1], gap="large")
	
	# Left column: Rater Information
	with col1:
		_render_rater_form()
	
	# Middle column: Panel Selection and Montage Settings
	with col2:
		selected_panels = PanelLayoutManager.render_panel_header_with_controls()
		st.divider()
		_render_montage_settings()
	
	# Right column: CSV Upload
	with col3:
		_render_csv_upload(participant_ids_in_ds, total_participants_in_ds)
	
	st.markdown("---")
	
	# Display panel layout preview based on selected panels
	_display_panel_layout_preview(selected_panels)


def _render_rater_form() -> None:
	"""Render rater information form in the landing page."""
	st.subheader(MESSAGES['rater_info_header'])
	with st.form("rater_form"):
		# Rater name/ID
		rater_id = st.text_input(
			MESSAGES['rater_id_prompt'],
			value=SessionManager.get_rater_id()
		)
		
		# Remove spaces from rater_id
		rater_id_clean = "".join(rater_id.split())
		
		# Experience level
		default_exp_idx = 0
		if SessionManager.get_rater_experience() in EXPERIENCE_LEVELS:
			default_exp_idx = EXPERIENCE_LEVELS.index(SessionManager.get_rater_experience())
		rater_experience = st.radio(
			MESSAGES['experience_prompt'],
			EXPERIENCE_LEVELS,
			index=default_exp_idx
		)
		
		# Fatigue level
		default_fatigue_idx = 0
		if SessionManager.get_rater_fatigue() in FATIGUE_LEVELS:
			default_fatigue_idx = FATIGUE_LEVELS.index(SessionManager.get_rater_fatigue())
		rater_fatigue = st.radio(
			MESSAGES['fatigue_prompt'],
			FATIGUE_LEVELS,
			index=default_fatigue_idx
		)
		
		submit_rater = st.form_submit_button(MESSAGES['rater_form_button'], use_container_width=True)
		
		if submit_rater:
			if not rater_id_clean:
				st.error(ERROR_MESSAGES['invalid_rater_id'])
			elif SessionManager.get_panel_count() == 0:
				st.error(ERROR_MESSAGES['no_panel_selected'])
			else:
				SessionManager.set_rater_id(rater_id_clean)
				SessionManager.set_rater_experience(rater_experience)
				SessionManager.set_rater_fatigue(rater_fatigue)
				SessionManager.set_landing_page_complete(True)
				st.rerun()


def _render_csv_upload(participant_ids_in_ds: set, total_participants_in_ds: int) -> None:
	"""Render CSV upload section in the landing page.
	
	Args:
		participant_ids_in_ds: Set of participant IDs in dataset
		total_participants_in_ds: Total number of participants in dataset
	"""
	st.subheader(MESSAGES['upload_header'])
	st.info(MESSAGES['upload_help'])
	
	uploaded_file = st.file_uploader(
		MESSAGES['csv_uploader_label'],
		type=UPLOAD_FILE_TYPES,
		key="qc_file_upload"
	)
	
	if uploaded_file is not None:
		try:
			# Read the uploaded file				
			df = pd.read_csv(uploaded_file, sep=None, engine='python')
			
			st.success(SUCCESS_MESSAGES['csv_loaded'].format(
				count=len(df),
				filename=uploaded_file.name
			))
			
			# Get unique participants in the uploaded CSV
			unique_participants_in_csv = df['participant_id'].nunique()
			participant_ids_in_csv = set(df['participant_id'].unique())
			
			# Validate: Check if CSV has participants not in the participant list
			invalid_participants = participant_ids_in_csv - participant_ids_in_ds
			if invalid_participants:
				st.error(ERROR_MESSAGES['no_participants'].format(
					count=len(invalid_participants),
					participants=', '.join(sorted(invalid_participants))
				))
				st.stop()
			
			# Check if CSV has more unique participants than dataset
			if unique_participants_in_csv > total_participants_in_ds:
				st.error(ERROR_MESSAGES['too_many_participants'].format(
					csv_count=unique_participants_in_csv,
					list_count=total_participants_in_ds
				))
				st.stop()
			
			# Load participant list and show comparison
			try:
				# Create comparison display
				col_comp1, col_comp2 = st.columns(2)
				with col_comp1:
					st.metric("Participants Reviewed", unique_participants_in_csv)
				with col_comp2:
					st.metric("Total Participants in ds", total_participants_in_ds)
				
				# Progress percentage
				progress_pct = (unique_participants_in_csv / total_participants_in_ds) * 100 if total_participants_in_ds > 0 else 0
				st.progress(min(progress_pct / 100, 1.0), text=f"{progress_pct:.1f}% complete")
				
			except Exception as e:
				st.warning(ERROR_MESSAGES['csv_comparison_error'].format(error=e))
			
			# Extract rater information from the first record
			if len(df) > 0:
				first_record = df.iloc[0]
				extracted_rater_id = str(first_record.get('rater_id', ''))
				extracted_experience = str(first_record.get('rater_experience', ''))
				extracted_fatigue = str(first_record.get('rater_fatigue', ''))
				
				# Update session state with extracted rater info
				SessionManager.set_rater_id(extracted_rater_id)
				SessionManager.set_rater_experience(extracted_experience)
				SessionManager.set_rater_fatigue(extracted_fatigue)
				
				st.info(INFO_MESSAGES['rater_info_extracted'])
				st.write(INFO_MESSAGES['rater_id_prefix'].format(id=extracted_rater_id))
				st.write(INFO_MESSAGES['experience_prefix'].format(exp=extracted_experience))
				st.write(INFO_MESSAGES['fatigue_prefix'].format(fatigue=extracted_fatigue))
			
			# Display preview
			st.subheader(INFO_MESSAGES['preview_header'])
			st.dataframe(df.head(10), use_container_width=True)
			
			# Option to load these records
			if st.button(INFO_MESSAGES['load_records_button'], use_container_width=True):
				# Convert dataframe rows to QCRecord objects
				loaded_records = []
				for _, row in df.iterrows():
					record = QCRecord(
						participant_id=str(row.get('participant_id', '')),
						session_id=str(row.get('session_id', '')),
						qc_task=str(row.get('qc_task', '')),
						pipeline=str(row.get('pipeline', '')),
						timestamp=str(row.get('timestamp', '')),
						rater_id=str(row.get('rater_id', '')),
						rater_experience=str(row.get('rater_experience', '')),
						rater_fatigue=str(row.get('rater_fatigue', '')),
						final_qc=str(row.get('final_qc', '')),
						notes=str(row.get('notes', '')) if pd.notna(row.get('notes')) else '',
					)
					loaded_records.append(record)
				
				SessionManager.set_qc_records(loaded_records)
				st.success(SUCCESS_MESSAGES['records_loaded'].format(count=len(loaded_records)))
				st.info(INFO_MESSAGES['proceed_with_form'])
				
		except Exception as e:
			st.error(ERROR_MESSAGES['file_load_error'].format(error=e))
	
	st.divider()
	st.markdown("""
	**ℹ️ Tips:**
	- Save your work frequently using the 'Save QC results to CSV' button
	- Your session data persists within this application
	- Upload a previous file to resume or review work
	""")


def _display_panel_layout_preview(selected_panels: dict) -> None:
	"""Display a preview of the panel layout based on selected panels.
	
	When Niivue is selected: Shows 3-column layout (controls | Niivue | SVG/IQM)
	When Niivue is not selected: Shows full-width layout
	
	Args:
		selected_panels: Dictionary of selected panels
	"""
	st.subheader("📐 Panel Layout Preview")
	
	show_niivue = selected_panels.get('niivue', False)
	show_svg = selected_panels.get('svg', False)
	show_iqm = selected_panels.get('iqm', False)
	
	# No panels selected
	if not (show_niivue or show_svg or show_iqm):
		st.info("👉 Select panels above to see the layout preview")
		return
	
	# 3-column layout: Niivue with another panel
	if show_niivue and (show_svg or show_iqm):
		st.write("**Layout:** 3-column (Controls | Niivue Viewer | Secondary Panel)")
		ctrl_col, viewer_col, panel_col = st.columns([0.2, 0.4, 0.4], gap="small")
		
		with ctrl_col:
			st.info("🎮 **Controls**\n\n- View Mode\n- Overlay\n- Colormap\n- Opacity")
		
		with viewer_col:
			st.info("🧠 **Niivue Viewer**\n\n3D MRI data will be displayed here")
		
		with panel_col:
			secondary = "📊 **SVG Montage**" if show_svg else "📈 **QC Metrics**"
			st.info(f"{secondary}\n\nSecondary visualization will be displayed here")
	
	# Full-width Niivue only
	elif show_niivue:
		st.write("**Layout:** 2-column (Controls | Niivue Viewer)")
		left_col, right_col = st.columns([0.32, 0.68], gap="small")
		
		with left_col:
			st.info("🎮 **Controls**\n\n- View Mode\n- Overlay\n- Colormap\n- Opacity")
		
		with right_col:
			st.info("🧠 **Niivue Viewer**\n\n3D MRI data will be displayed here")
	
	# Full-width SVG only
	elif show_svg:
		st.write("**Layout:** Full-width (SVG Montage)")
		st.info("📊 **SVG Montage**\n\nSVG visualization will be displayed across the full width")
	
	# Full-width IQM only
	elif show_iqm:
		st.write("**Layout:** Full-width (QC Metrics)")
		st.info("📈 **QC Metrics**\n\nQC metrics will be displayed across the full width")


def _render_montage_settings() -> None:
	"""Render montage grid configuration settings.
	
	Allows users to specify maximum rows and columns for the SVG montage grid.
	When both are set to None (auto), the montage will optimize for square aspect ratio.
	"""
	st.markdown("#### 🎨 SVG Montage Grid Settings")
	
	with st.form("montage_settings_form"):
		col1, col2 = st.columns(2)
		
		with col1:
			current_rows = SessionManager.get_montage_max_rows()
			montage_rows = st.number_input(
				"Max Rows (use checkbox for auto-calculation)",
				min_value=MIN_MONTAGE_GRID_SIZE,
				max_value=MAX_MONTAGE_GRID_SIZE,
				value=current_rows if current_rows else MIN_MONTAGE_GRID_SIZE,
				step=1,
				help="Maximum number of rows in the SVG montage grid"
			)
			use_auto_rows = st.checkbox("Auto-calculate rows", value=(current_rows is None))
		
		with col2:
			current_cols = SessionManager.get_montage_max_cols()
			montage_cols = st.number_input(
				"Max Columns (use checkbox for auto-calculation)",
				min_value=MIN_MONTAGE_GRID_SIZE,
				max_value=MAX_MONTAGE_GRID_SIZE,
				value=current_cols if current_cols else MIN_MONTAGE_GRID_SIZE,
				step=1,
				help="Maximum number of columns in the SVG montage grid"
			)
			use_auto_cols = st.checkbox("Auto-calculate columns", value=(current_cols is None))
		
		submit_montage = st.form_submit_button("Apply Montage Settings", use_container_width=True)
		
		if submit_montage:
			# Set to None if auto-calculate is checked, otherwise use the specified value
			rows_to_set = None if use_auto_rows else montage_rows
			cols_to_set = None if use_auto_cols else montage_cols
			
			SessionManager.set_montage_max_rows(rows_to_set)
			SessionManager.set_montage_max_cols(cols_to_set)
			
			# Display confirmation
			auto_text = "(auto)" if use_auto_rows else f"({montage_rows})"
			auto_text_cols = "(auto)" if use_auto_cols else f"({montage_cols})"
			st.success(f"✅ Montage settings updated: Max rows {auto_text}, Max columns {auto_text_cols}")
	
	# Show current settings
	current_rows = SessionManager.get_montage_max_rows()
	current_cols = SessionManager.get_montage_max_cols()
	rows_display = "Auto" if current_rows is None else str(current_rows)
	cols_display = "Auto" if current_cols is None else str(current_cols)
	st.info(f"📋 Current montage settings: Max rows = {rows_display}, Max columns = {cols_display}")
