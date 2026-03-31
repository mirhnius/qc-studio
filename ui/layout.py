import os
from pathlib import Path
from datetime import datetime
import streamlit as st
from niivue_component import niivue_viewer
from utils import parse_qc_config, load_mri_data, load_svg_data, save_qc_results_to_csv
from models import MetricQC, QCRecord


def app(participant_id, session_id, qc_pipeline, qc_task, qc_config_path, out_dir) -> None:
	"""Main Streamlit layout: top inputs, middle two viewers, bottom QC controls."""
	st.set_page_config(layout="wide")

	# Top container: inputs
	top = st.container()
	with top:
		st.title("Welcome to Nipoppy QC-Studio! 🚀")
		# qc_pipeline = "fMRIPrep"
		# qc_task = "sdc-wf"
		st.subheader(f"QC Pipeline: {qc_pipeline}, QC task: {qc_task}")

		# show participant and session
		st.write(f"Participant ID: {participant_id} | Session ID: {session_id}")        

		# Rater info 
		rater_id = st.text_input("Rater name or ID: 🧑" )
		st.write("You entered:", rater_id)
		
        # Remove spaces
		rater_id = "".join(rater_id.split())

		# Split into two columns for collecting rater specific info
		exp_col, fatigue_col = st.columns([0.5, 0.5], gap="small")
		
		with exp_col:
			# Input rater experience as radio buttons
			options = ["Beginner (< 1 year experience)", "Intermediate (1-5 year experience)", "Expert (>5 year experience)"]
			# add radio buttons
			# experience_level = st.radio()
			rater_experience = st.radio("What is your QC experience level:", options)
			st.write("Experience level:", rater_experience)
			
		with fatigue_col:
			# Input rater experience as radio buttons
			options = ["Not at all", "A bit tired ☕", "Very tired ☕☕"]
			# add radio buttons
			# experience_level = st.radio()
			rater_fatigue = st.radio("How tired are you feeling:", options)
			st.write("Fatigue level:", rater_fatigue)
		

	# parse qc config
	qc_config = parse_qc_config(qc_config_path, qc_task) 
	# print(f"qc config: {qc_config_path}, {qc_config}")

	# Middle: two side-by-side viewers
	middle = st.container()
	with middle:
		niivue_col, svg_col = st.columns([0.4, 0.6], gap="small")

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

		with svg_col:
			st.header("SVG Montage")
			# Show SVG montage
			svg_data = load_svg_data(qc_config)
			if svg_data:
				st.components.v1.html(svg_data, height=600, scrolling=True)
			else:
				st.info("SVG montage not found or could not be loaded.")

	# Bottom: QC metrics and radio buttons
	bottom = st.container()
	with bottom:
		# st.header("QC: Rating & Metrics")
		rating_col, iqm_col = st.columns([0.4, 0.6], gap="small")
		with iqm_col:
			st.subheader("QC Metrics")
			# Placeholder: user may compute or display metrics here
			st.write("Add QC metrics here (e.g., SNR, motion). This is a placeholder area.")

		with rating_col:
			st.subheader("QC Rating")
			rating = st.radio("Rate this qc-task:", options=("PASS", "FAIL", "UNCERTAIN"), index=0)
			notes = st.text_area("Notes (optional):")
			if st.button("💾 Save QC results to CSV", width=600):
				now = datetime.now()
				timestamp = now.strftime("%Y-%m-%d %H:%M:%S")				
				out_file = Path(out_dir) / f"{rater_id}_QC_status.tsv"

				record = QCRecord(
					participant_id=participant_id,
					session_id=session_id,
					qc_task=qc_task,
					pipeline=qc_pipeline,
					timestamp=timestamp,
					rater_id=rater_id,
					rater_experience=rater_experience,
					rater_fatigue=rater_fatigue,
					final_qc=rating,
					notes=notes,
				)

                # TODO: handle list of records (i.e. multiple subjects and/or qc-tasks)
				# For now just save a single record
                
				record_list = [record]
				out_path = save_qc_results_to_csv(out_file, record_list)
				st.success(f"QC results saved to: {out_path}")
				
                
