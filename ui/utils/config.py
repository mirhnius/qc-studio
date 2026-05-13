"""QC configuration parsing utilities."""
from pathlib import Path
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
	# print(f"Parsing QC config: {qc_json_path}, task: {qc_task}")

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
