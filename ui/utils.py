import json
from pathlib import Path
import pandas as pd
from models import QCConfig
from constants import SUBSTITUTIONS_DICT

def parse_qc_config(qc_json, qc_task, substitution_values) -> dict:
	"""Parse a QC JSON file using the QCConfig Pydantic model.

	Returns a dict with keys:
	  - 'base_mri_image_path': Path | None
	  - 'overlay_mri_image_path': Path | None
	  - 'svg_montage_path': Path | None
	  - 'iqm_path': Path | None

	If the file is missing, invalid, or the requested qc_task is not present,
	all values will be None. Uses `QCConfig` from `models` for validation.
	"""

	qc_json_path = Path(qc_json) if qc_json else None
	print(f"Parsing QC config: {qc_json_path}, task: {qc_task}")

	try:
		# Pydantic v2 deprecates `parse_file`; read file and validate JSON string.
		raw_text = qc_json_path.read_text()

		# Make all the substitutions in the raw text before parsing with Pydantic
		for key, substitution in SUBSTITUTIONS_DICT.items():
			if substitution in raw_text:
				raw_text = raw_text.replace(substitution, substitution_values.get(key))

		qcconf = QCConfig.model_validate_json(raw_text)
	except Exception:
		# validation/parsing failed
		return {
			"base_mri_image_path": None,
			"overlay_mri_image_path": None,
			"svg_montage_path": None,
			"iqm_path": None,
		}

	# qcconf.root is a dict: qc_task -> QCTask (RootModel)
	qctask = qcconf.root.get(qc_task)
	if not qctask:
		return {
			"base_mri_image_path": None,
			"overlay_mri_image_path": None,
			"svg_montage_path": None,
			"iqm_path": None,
		}

	# qctask is a QCTask model; its fields are Path or None already
	return {
		"base_mri_image_path": qctask.base_mri_image_path,
		"overlay_mri_image_path": qctask.overlay_mri_image_path,
		"svg_montage_path": qctask.svg_montage_path,
		"iqm_path": qctask.iqm_path,
	}


def load_mri_data(dataset_dir, path_dict: dict) -> dict:
	"""Load base and overlay MRI image files as bytes."""

	base_mri_path = Path(dataset_dir).joinpath(path_dict.get("base_mri_image_path"))
	overlay_mri_path = Path(dataset_dir).joinpath(path_dict.get("overlay_mri_image_path"))
	file_bytes_dict = {}

	if base_mri_path and Path(base_mri_path).is_file():
		file_bytes_dict["base_mri_image_bytes"] = Path(base_mri_path).read_bytes()
		file_bytes_dict["base_mri_image_path"] = base_mri_path

	if overlay_mri_path and Path(overlay_mri_path).is_file():
		file_bytes_dict["overlay_mri_image_bytes"] = Path(overlay_mri_path).read_bytes()

	return file_bytes_dict


def load_svg_data(dataset_dir, path_dict: dict) -> dict | None:
	"""Load SVG montage file(s) content as dict.
	
	Loads one or more SVG files from paths specified in path_dict.
	Supports both single path and list of paths for backward compatibility.
	Handles string representations of path lists (e.g., from JSON parsing).
	
	Args:
		dataset_dir: Base directory path
		path_dict: Dictionary containing 'svg_montage_path' key with:
			- Single Path/str (legacy), or
			- List of Path/str objects (new), or
			- String representation of list (from JSON)
	
	Returns:
		Dict with format: {filename: svg_content_string}
		Returns None if no valid SVG files are found.
	
	Example:
		{
			"montage_1.svg": "<svg>...</svg>",
			"montage_2.svg": "<svg>...</svg>"
		}
	"""
	svg_paths = path_dict.get("svg_montage_path")
	if not svg_paths:
		return None
	
	print(f"Original SVG paths from config: {svg_paths} (type: {type(svg_paths)})")

	if isinstance(svg_paths, Path):
		svg_paths = [svg_paths]
	elif not isinstance(svg_paths, list):
		# Fallback for other types
		svg_paths = [svg_paths]

	svg_data = {}
	
	print(f"Loading SVG data from paths: {svg_paths}")
	for svg_path in svg_paths:
		full_path = Path(dataset_dir).joinpath(str(svg_path))
		print(f"Attempting to load SVG from: {full_path}")
		if full_path and full_path.is_file():
			try:
				with open(full_path, "r") as f:
					# Use filename as key for tab labels
					filename = full_path.name
					svg_data[filename] = f.read()
			except Exception:
				continue
	
	print(f"Loaded SVG data for paths: {list(svg_data.keys())}")
	return svg_data if svg_data else None


def load_iqm_data(dataset_dir, path_dict: dict) -> dict | None:
	"""Load IQM JSON file content as dict."""
	iqm_path = Path(dataset_dir).joinpath(path_dict.get("iqm_path"))
	if iqm_path and iqm_path.is_file():
		try:
			with open(iqm_path, "r") as f:
				return json.load(f)
		except Exception:
			return None
	return None


# TODO : integrate with layout.py
def save_qc_results_to_csv(out_file, qc_records, drop_duplicates=True):
	"""
	Save QC results from Streamlit session state to a CSV file.

	This function is resilient to both `QCRecord` model instances and plain
	dicts. It will extract the canonical fields from the updated `QCRecord`:
	- qc_task, participant_id, session_id, task_id, run_id, pipeline,
	  timestamp, rater_id, rater_experience, rater_fatigue, final_qc

	If a record also contains a `metrics` list (items compatible with
	`MetricQC`), those metrics will be flattened into columns as
	`<metric_name>_value` and `<metric_name>` (for qc string), and
	`QC_notes` (if present) will be placed in a `notes` column.

	Parameters
	----------
	out_file : str or Path
		Path where the CSV will be saved.
	qc_records : list
		List of `QCRecord` objects (or dicts) stored.
	"""
	out_file = Path(out_file)
	out_file.parent.mkdir(parents=True, exist_ok=True)

	rows = []

	for rec in qc_records:
		# support both model instances and plain dicts
		if hasattr(rec, "model_dump"):
			# pydantic v2 model -> convert to dict for uniform access
			rec_dict = rec.model_dump()
		elif hasattr(rec, "dict"):
			# pydantic v1 fallback
			rec_dict = rec.dict()
		elif isinstance(rec, dict):
			rec_dict = rec
		else:
			# Handle this better with exceptions
			print("Unknown record format")

		row = {
			"qc_task": rec_dict.get("qc_task"),
			"participant_id": rec_dict.get("participant_id"),
			"session_id": rec_dict.get("session_id"),
			"task_id": rec_dict.get("task_id"),
			"run_id": rec_dict.get("run_id"),
			"pipeline": rec_dict.get("pipeline"),
			"timestamp": rec_dict.get("timestamp"),
			"rater_id": rec_dict.get("rater_id"),
			"rater_experience": rec_dict.get("rater_experience"),
			"rater_fatigue": rec_dict.get("rater_fatigue"),
			"final_qc": rec_dict.get("final_qc"),
			"notes": rec_dict.get("notes"),
		}
		rows.append(row)

	# Define expected columns
	expected_columns = [
		"qc_task", "participant_id", "session_id", "task_id", "run_id",
		"pipeline", "timestamp", "rater_id", "rater_experience",
		"rater_fatigue", "final_qc", "notes"
	]

	# Create dataframe with proper columns even if empty
	if rows:
		df = pd.DataFrame(rows)
	else:
		df = pd.DataFrame(columns=expected_columns)

	if out_file.exists():
		df_existing = pd.read_csv(out_file, sep="\t")
		df = pd.concat([df_existing, df], ignore_index=True)

		# Drop duplicates based on core identity columns
		if drop_duplicates:
			subset_keys = ["participant_id", "session_id", "pipeline", "qc_task"]
			existing_keys = [k for k in subset_keys if k in df.columns]
			if existing_keys:
				df = df.drop_duplicates(subset=existing_keys, keep="last")

	# Only sort if dataframe is not empty
	if not df.empty:
		sort_key = "participant_id" if "participant_id" in df.columns else df.columns[0]
		df = df.sort_values(by=[sort_key]).reset_index(drop=True)
	
	df.to_csv(out_file, index=False, sep='\t')

	return out_file
