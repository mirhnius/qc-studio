from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
from niivue_component import niivue_viewer
from utils import parse_qc_config, load_mri_data, load_svg_data, save_qc_results_to_csv
from models import MetricQC, QCRecord


def show_landing_page(qc_pipeline, qc_task, out_dir, participant_list) -> None:
	"""Display the landing page with rater info and CSV upload."""
	st.title("Welcome to Nipoppy QC-Studio! 🚀")

	# Load participant list to get total unique participants
	try:
		participants_df = pd.read_csv(participant_list, delimiter="\t")
		total_participants_in_ds = len(participants_df['participant_id'].unique())
		participant_ids_in_ds = set(participants_df['participant_id'].unique())
	except Exception as e:
		st.error(f"Error loading participant list: {e}")
		return

	st.subheader(f"QC Pipeline: {qc_pipeline} | QC Task: {qc_task} | n_ds_participants: {total_participants_in_ds}")
	
	st.markdown("---")
	
	# Three-column layout for rater info, panel selection, and CSV upload
	col1, col2, col3 = st.columns([1, 1, 1], gap="large")
	
	# Left column: Rater Information
	with col1:
		st.subheader("👤 Rater Information")
		with st.form("rater_form"):
			# Rater name/ID
			rater_id = st.text_input(
				"Enter your Rater Name or ID:",
				value=st.session_state.get('rater_id', '')
			)
			
			# Remove spaces from rater_id
			rater_id_clean = "".join(rater_id.split())
			
			# Experience level
			options_exp = ["Beginner (< 1 year experience)", "Intermediate (1-5 year experience)", "Expert (>5 year experience)"]
			default_exp_idx = 0
			if st.session_state.get('rater_experience') in options_exp:
				default_exp_idx = options_exp.index(st.session_state.get('rater_experience'))
			rater_experience = st.radio(
				"What is your QC experience level?",
				options_exp,
				index=default_exp_idx
			)
			
			# Fatigue level
			options_fatigue = ["Not at all", "A bit tired ☕", "Very tired ☕☕"]
			default_fatigue_idx = 0
			if st.session_state.get('rater_fatigue') in options_fatigue:
				default_fatigue_idx = options_fatigue.index(st.session_state.get('rater_fatigue'))
			rater_fatigue = st.radio(
				"How tired are you feeling?",
				options_fatigue,
				index=default_fatigue_idx
			)
			
			submit_rater = st.form_submit_button("✅ Continue to QC", use_container_width=True)
			
			if submit_rater:
				if not rater_id_clean:
					st.error("Please enter a valid Rater ID (no spaces).")
				elif sum(st.session_state['selected_panels'].values()) == 0:
					st.error("⚠️ You must select at least one display panel to proceed!")
				else:
					st.session_state['rater_id'] = rater_id_clean
					st.session_state['rater_experience'] = rater_experience
					st.session_state['rater_fatigue'] = rater_fatigue
					st.session_state['landing_page_complete'] = True
					st.rerun()
	
	# Middle column: Panel Selection
	with col2:
		st.subheader("🖼️ Display Panels")
		st.info("Select which panels to display during QC (at least one required).")
		
		# Initialize panel selections if not present
		if 'selected_panels' not in st.session_state:
			st.session_state['selected_panels'] = {
				'niivue_col': True,
				'svg_col': True,
				'iqm_col': False
			}
		
		# Panel selection checkboxes
		st.session_state['selected_panels']['niivue_col'] = st.checkbox(
			"🧠 3D MRI Viewer (Niivue)",
			value=st.session_state['selected_panels'].get('niivue_col', True),
			help="Display interactive 3D MRI viewer"
		)
		
		st.session_state['selected_panels']['svg_col'] = st.checkbox(
			"📊 SVG Montage",
			value=st.session_state['selected_panels'].get('svg_col', True),
			help="Display SVG montage visualization"
		)
		
		st.session_state['selected_panels']['iqm_col'] = st.checkbox(
			"📈 QC Metrics",
			value=st.session_state['selected_panels'].get('iqm_col', False),
			help="Display QC metrics panel"
		)
		
		# Validation info
		selected_count = sum(st.session_state['selected_panels'].values())
		if selected_count == 0:
			st.warning("⚠️ You must select at least one panel to proceed!")
		else:
			st.success(f"✅ {selected_count} panel(s) selected")
	
	# Right column: CSV Upload
	with col3:
		st.subheader("📤 Upload Existing QC File (Optional)")
		st.info("Upload a previously saved QC_status.csv file to resume your QC session or review previous results.")
		
		uploaded_file = st.file_uploader(
			"Choose a QC_status.csv file",
			type=["csv", "tsv"],
			key="qc_file_upload"
		)
		
		if uploaded_file is not None:
			try:
				# Read the uploaded file				
				df = pd.read_csv(uploaded_file, sep=None, engine='python')
				
				st.success(f"✅ Loaded {len(df)} QC records from {uploaded_file.name}")
				
				# Get unique participants in the uploaded CSV
				unique_participants_in_csv = df['participant_id'].nunique()
				participant_ids_in_csv = set(df['participant_id'].unique())
				
				# Validate: Check if CSV has participants not in the participant list
				invalid_participants = participant_ids_in_csv - participant_ids_in_ds
				if invalid_participants:
					st.error(f"❌ Error: The uploaded CSV contains {len(invalid_participants)} participant(s) not in the participant list: {', '.join(sorted(invalid_participants))}")
					st.stop()
				
				# Check if CSV has more unique participants than dataset (should not happen if validation above passes)
				if unique_participants_in_csv > total_participants_in_ds:
					st.error(f"❌ Error: The uploaded CSV has {unique_participants_in_csv} unique participants, but the participant list only has {total_participants_in_ds}.")
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
					st.warning(f"Could not display comparison: {e}")
				
				# Extract rater information from the first record
				if len(df) > 0:
					first_record = df.iloc[0]
					extracted_rater_id = str(first_record.get('rater_id', ''))
					extracted_experience = str(first_record.get('rater_experience', ''))
					extracted_fatigue = str(first_record.get('rater_fatigue', ''))
					
					# Update session state with extracted rater info
					st.session_state['rater_id'] = extracted_rater_id
					st.session_state['rater_experience'] = extracted_experience
					st.session_state['rater_fatigue'] = extracted_fatigue
					
					st.info(f"📋 Rater information extracted:")
					st.write(f"- **Rater ID:** {extracted_rater_id}")
					st.write(f"- **Experience:** {extracted_experience}")
					st.write(f"- **Fatigue Level:** {extracted_fatigue}")
				
				# Display preview
				st.subheader("Preview of Loaded Records")
				st.dataframe(df.head(10), use_container_width=True)
				
				# Option to load these records
				if st.button("📥 Load These Records", use_container_width=True):
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
					
					st.session_state['qc_records'] = loaded_records
					st.success(f"✅ Loaded {len(loaded_records)} QC records into session!")
					st.info("You can now proceed with the rater form on the left to continue QC.")
					
			except Exception as e:
				st.error(f"❌ Error loading file: {e}")
		
		st.divider()
		st.markdown("""
		**ℹ️ Tips:**
		- Save your work frequently using the 'Save QC results to CSV' button
		- Your session data persists within this application
		- Upload a previous file to resume or review work
		""")


def app(participant_id, session_id, qc_pipeline, qc_task, qc_config_path, out_dir, total_participants, drop_duplicates, participant_list) -> None:
	"""Main Streamlit layout: landing page, QC viewers, and congratulations."""
	st.set_page_config(layout="wide")

	# Initialize session state for landing page
	if 'landing_page_complete' not in st.session_state:
		st.session_state['landing_page_complete'] = False

	# Check if we're on the landing page
	if not st.session_state['landing_page_complete']:
		show_landing_page(qc_pipeline, qc_task, out_dir, participant_list)
		return

	# Check if we're on the final congratulations page
	if participant_id is None:
		# Final page: Show congratulations message
		st.title("🎉 QC Complete! Congratulations! 🎉")
		
		# Display rater info and summary statistics
		rater_id = st.session_state.get('rater_id', 'Unknown')
		record_list = st.session_state.get('qc_records', [])
		num_reviewed = len(record_list)
		
		st.markdown(f"""
		## {num_reviewed} participant(s) have been reviewed!
		
		Thank you for completing the quality control process. Your thorough review ensures the integrity of our data!
		
		✅ All QC records have been automatically saved.
		
		""")
		
		col1, col2 = st.columns([1, 1])
		with col1:
			st.subheader("Session Information")
			st.write(f"**Rater ID:** {rater_id}")
			st.write(f"**QC Task:** {qc_task}")
			st.write(f"**Total Participants Reviewed:** {len(record_list)}")
		
		with col2:
			st.subheader("QC Results Summary")
			# Count final_qc values
			if record_list:
				final_qc_counts = {}
				for record in record_list:
					qc_value = record.final_qc
					final_qc_counts[qc_value] = final_qc_counts.get(qc_value, 0) + 1
				
				for qc_status, count in sorted(final_qc_counts.items()):
					st.write(f"**{qc_status}:** {count}")
		
		col1, col2, col3 = st.columns([1, 1, 1])
		with col1:
			if st.button("💾 Export Final Results", width=300):
				rater_id = st.session_state.get('rater_id', 'unknown')
				out_file = Path(out_dir) / f"{rater_id}_QC_status.tsv"
				record_list = st.session_state.get('qc_records', [])
				if record_list:
					out_path = save_qc_results_to_csv(out_file, record_list, drop_duplicates)
					st.success(f"✅ All QC results exported to: {out_path}")
				else:
					st.info("No QC records to export.")
		with col2:
			if st.button("◀️ Previous", width=300):
				st.session_state['current_page'] -= 1
				st.rerun()
		with col3:
			if st.button("🔄 Start Over (Page 1)", width=300):
				st.session_state['current_page'] = 1
				st.rerun()
		return

	# Top container: participant info
	top = st.container()
	with top:
		st.title("Nipoppy QC-Studio: Quality Control")
		# st.subheader(f"Pipeline: {qc_pipeline} | Task: {qc_task}")
		# st.write(f"**Participant:** {participant_id} | **Session:** {session_id}")
		
		# Display rater info summary
		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("Rater", st.session_state.get('rater_id', 'Unknown'))
		with col2:
			st.metric("Experience", st.session_state.get('rater_experience', 'Unknown').split('(')[0].strip())
		with col3:
			st.metric("Fatigue Level", st.session_state.get('rater_fatigue', 'Unknown').split('☕')[0].strip())

		col_participant_info, col_pipe_info = st.columns(2)
		with col_participant_info:
			st.write(f"### Participant: {participant_id} | Session: {session_id}")
		with col_pipe_info:
			st.write(f"### Pipeline: {qc_pipeline} | Task: {qc_task}")
			
	
	# parse qc config
	qc_config = parse_qc_config(qc_config_path, qc_task) 
	# print(f"qc config: {qc_config_path}, {qc_config}")

	# Middle: two side-by-side viewers
	middle = st.container()
	with middle:
		# Determine which panels are selected
		selected_panels = st.session_state.get('selected_panels', {
			'niivue_col': True,
			'svg_col': True,
			'iqm_col': False
		})
		
		show_niivue = selected_panels.get('niivue_col', True)
		show_svg = selected_panels.get('svg_col', True)
		show_iqm = selected_panels.get('iqm_col', False)
		
		# Calculate column layout based on selected panels
		if show_niivue and show_svg and show_iqm:
			# All three panels: niivue (40%), svg (35%), iqm will be below
			show_iqm_below = True
			niivue_col, svg_col = st.columns([0.4, 0.6], gap="small")
		elif show_niivue and show_svg:
			# Two panels side-by-side
			niivue_col, svg_col = st.columns([0.4, 0.6], gap="small")
			show_iqm_below = False
		elif show_niivue and show_iqm:
			# Niivue and IQM
			niivue_col, iqm_col = st.columns([0.5, 0.5], gap="small")
			show_iqm_below = False
		elif show_svg and show_iqm:
			# SVG and IQM
			svg_col, iqm_col = st.columns([0.5, 0.5], gap="small")
			show_iqm_below = False
		elif show_niivue:
			# Only Niivue selected
			niivue_col = st.container()
			show_iqm_below = False
		elif show_svg:
			# Only SVG selected
			svg_col = st.container()
			show_iqm_below = False
		elif show_iqm:
			# Only IQM selected
			iqm_col = st.container()
			show_iqm_below = False
		else:
			# Fallback - should not reach here if validation works
			show_iqm_below = False

		# Display Niivue panel if selected
		if show_niivue:
			with niivue_col:
				# Create a narrow controls column and a main viewer area inside the niivue column
				cfg_col, view_col = st.columns([0.32, 0.68], gap="small")

				with cfg_col:
					st.header("Niivue Controls")
					# Persistent controls column (sidebar-like)
					view_mode = st.selectbox(
						"View Mode",
						["multiplanar", "axial", "coronal", "sagittal", "3d"],
						help="Select the viewing perspective"
					)

					height = 600 #st.slider("Viewer Height (px)", 400, 1000, 600, 50)
					overlay_colormap = st.selectbox(
						"Overlay Colormap",
						["grey", "cool", "warm"],
						help="Select the colormap for the overlay"
					)

					st.divider()
					st.subheader("Display Settings")
					show_crosshair = st.checkbox("Show Crosshair", value=False)
					radiological = st.checkbox("Radiological Convention", value=False)
					show_colorbar = st.checkbox("Show Colorbar", value=True)
					interpolation = st.checkbox("Interpolation", value=True)

					# Toggle to show/hide overlay image in the Niivue column
					show_overlay = st.checkbox("Show overlay image", value=False)

				with view_col:
					st.header("3D MRI (Niivue)")
					# Show mri
					mri_data = load_mri_data(qc_config)
					if "base_mri_image_bytes" in mri_data:
						base_mri_image_bytes = mri_data["base_mri_image_bytes"]
						base_mri_name = str(qc_config.get("base_mri_image_path").name) if qc_config.get("base_mri_image_path") else "base_mri.nii"

						try:
							# Prepare settings dictionary
							settings = {
								"crosshair": show_crosshair,
								"radiological": radiological,
								"colorbar": show_colorbar,
								"interpolation": interpolation
							}

							# Prepare optional overlays only if user enabled and overlay bytes exist
							overlays = []
							if show_overlay and "overlay_mri_image_bytes" in mri_data:
								overlays.append(
									{
										"data": mri_data["overlay_mri_image_bytes"],
										"name": "overlay",
										"colormap": overlay_colormap,
										"opacity": 0.5,
									}
								)

							# Build kwargs for niivue_viewer; include overlays only when present
							overlay_state = f"{overlay_colormap}_{show_overlay}"
							viewer_key = f"niivue_{view_mode}_{overlay_state}"

							viewer_kwargs = {
								"nifti_data": base_mri_image_bytes,
								"filename": base_mri_name,
								"height": height,
								"key": viewer_key,
								"view_mode": view_mode,
								"settings": settings,
							}
							if overlays:
								viewer_kwargs["overlays"] = overlays
							
							viewer_kwargs["styled"] = True

							niivue_viewer(**viewer_kwargs)

						except Exception as e:
							st.error(f"Failed to load base MRI in Niivue viewer: {e}")
					else:
						st.info("Base MRI image not found or could not be loaded.")

		# Display SVG panel if selected
		if show_svg:
			with svg_col:
				st.header("SVG Montage")
				# Show SVG montage
				svg_data = load_svg_data(qc_config)
				if svg_data:
					st.components.v1.html(svg_data, height=600, scrolling=True)
				else:
					st.info("SVG montage not found or could not be loaded.")
		
		# Display IQM panel if selected and not below
		if show_iqm and not show_iqm_below:
			with iqm_col:
				st.subheader("QC Metrics")
				# Placeholder: user may compute or display metrics here
				st.write("Add QC metrics here (e.g., SNR, motion). This is a placeholder area.")
		
		# If IQM should be displayed below (when all three panels selected)
		if show_iqm_below:
			st.divider()
			iqm_full = st.container()
			with iqm_full:
				st.subheader("📈 QC Metrics")
				st.write("Add QC metrics here (e.g., SNR, motion). This is a placeholder area.")

	# Bottom: QC metrics and radio buttons
	bottom = st.container()
	with bottom:
		# Determine if iqm_col should be shown alongside rating_col
		selected_panels = st.session_state.get('selected_panels', {
			'niivue_col': True,
			'svg_col': True,
			'iqm_col': False
		})
		
		show_iqm_bottom = selected_panels.get('iqm_col', False) and (
			selected_panels.get('niivue_col', True) or 
			selected_panels.get('svg_col', True)
		)
		
		if show_iqm_bottom:
			rating_col, iqm_col = st.columns([0.4, 0.6], gap="small")
		else:
			rating_col = st.container()
		
		if show_iqm_bottom:
			with iqm_col:
				st.subheader("QC Metrics")
				# Placeholder: user may compute or display metrics here
				st.write("Add QC metrics here (e.g., SNR, motion). This is a placeholder area.")

		with rating_col:
			st.subheader("QC Rating")
			rating = st.radio("Rate this qc-task:", options=("PASS", "FAIL", "UNCERTAIN"), index=0)
			notes = st.text_area("Notes (optional):", value=st.session_state.get('notes', ''))
			st.session_state['notes'] = notes
			if st.button("💾 Save QC results to CSV", width=600):
				now = datetime.now()
				timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

				record = QCRecord(
					participant_id=participant_id,
					session_id=session_id,
					qc_task=qc_task,
					pipeline=qc_pipeline,
					timestamp=timestamp,
					rater_id=st.session_state.get('rater_id', 'Unknown'),
					rater_experience=st.session_state.get('rater_experience', 'Unknown'),
					rater_fatigue=st.session_state.get('rater_fatigue', 'Unknown'),
					final_qc=rating,
					notes=notes,
				)

				st.session_state['qc_records'].append(record)
				record_list = st.session_state['qc_records']
				st.session_state['current_page'] = total_participants + 1
				st.rerun()

			# Pagination controls
			st.divider()
			st.write(f"Participant {st.session_state['current_page']} of {total_participants}")
			col1, col2, col3 = st.columns([1, 1, 1])
			with col1:
				if st.button("Previous"):
					st.session_state['current_page'] -= 1
					st.rerun()
			with col2:
				if st.button("Confirm and Next"):
					now = datetime.now()
					timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
					record = QCRecord(
						participant_id=participant_id,
						session_id=session_id,
						qc_task=qc_task,
						pipeline=qc_pipeline,
						timestamp=timestamp,
						rater_id=st.session_state.get('rater_id', 'Unknown'),
						rater_experience=st.session_state.get('rater_experience', 'Unknown'),
						rater_fatigue=st.session_state.get('rater_fatigue', 'Unknown'),
						final_qc=rating,
						notes=notes,
					)
					st.session_state['qc_records'].append(record)
					record_list = st.session_state['qc_records']
					# out_path = save_qc_results_to_csv(out_file, record_list)
					# st.success(f"QC results saved to: {out_path}")
					st.session_state['current_page'] += 1
					st.rerun()
			with col3:
				if st.button("Next"):
					st.session_state['current_page'] += 1
					st.rerun()
			
			st.divider()
			if st.button("🏠 Back to Landing Page", use_container_width=True):
				st.session_state['landing_page_complete'] = False
				st.rerun()
				
                
