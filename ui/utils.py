import json
from pathlib import Path
import pandas as pd
from models import QCConfig, QCRecord


def parse_qc_config(qc_json, qc_task) -> dict:
    """
    Parse a QC JSON file using the QCConfig Pydantic model.

    Returns a dict with keys:
      - 'base_mri_image_path': Path | None
      - 'overlay_mri_image_path': Path | None
      - 'svg_montage_path': list[Path] | None
      - 'iqm_path': list[Path] | None

    If the file is missing, invalid, or the requested qc_task is not present,
    all values will be None.
    """
    qc_json_path = Path(qc_json) if qc_json else None

    try:
        raw_text = qc_json_path.read_text()
        qcconf = QCConfig.model_validate_json(raw_text)
    except Exception:
        return {
            "base_mri_image_path": None,
            "overlay_mri_image_path": None,
            "svg_montage_path": None,
            "iqm_path": None,
        }

    qctask = qcconf.root.get(qc_task)
    if not qctask:
        return {
            "base_mri_image_path": None,
            "overlay_mri_image_path": None,
            "svg_montage_path": None,
            "iqm_path": None,
        }

    return {
        "base_mri_image_path": qctask.base_mri_image_path,
        "overlay_mri_image_path": qctask.overlay_mri_image_path,
        "svg_montage_path": qctask.svg_montage_path,
        "iqm_path": qctask.iqm_path,
    }


def load_mri_data(path_dict: dict) -> dict:
    """Load base and overlay MRI image files as bytes."""
    base_mri_path = path_dict.get("base_mri_image_path")
    overlay_mri_path = path_dict.get("overlay_mri_image_path")

    file_bytes_dict = {}

    if base_mri_path and Path(base_mri_path).is_file():
        file_bytes_dict["base_mri_image_bytes"] = Path(base_mri_path).read_bytes()

    if overlay_mri_path and Path(overlay_mri_path).is_file():
        file_bytes_dict["overlay_mri_image_bytes"] = Path(overlay_mri_path).read_bytes()

    return file_bytes_dict


def load_svg_data(path_dict: dict) -> list[str]:
    """
    Load SVG montage file(s) content as strings.
    Returns a list (possibly empty).
    """
    svg_paths = path_dict.get("svg_montage_path") or []
    out = []

    for p in svg_paths:
        p = Path(p)
        if p.is_file():
            try:
                out.append(p.read_text())
            except Exception:
                pass

    return out


def load_iqm_data(path_dict: dict):
    """
    Load IQM files.
    - TSV files are returned as pandas DataFrames
    - JSON files are returned as dicts

    Returns a list of loaded objects (possibly empty).
    """
    iqm_paths = path_dict.get("iqm_path") or []
    out = []

    for p in iqm_paths:
        p = Path(p)
        if not p.is_file():
            continue

        suffix = p.suffix.lower()

        if suffix == ".tsv":
            try:
                out.append(pd.read_csv(p, sep="\t"))
            except Exception:
                pass
        elif suffix == ".json":
            try:
                out.append(json.loads(p.read_text()))
            except Exception:
                pass
        else:
            try:
                out.append(p.read_text())
            except Exception:
                pass

    return out


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
