import json
from pathlib import Path
import pandas as pd
from models import QCConfig


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


def save_qc_results_to_csv(out_file, qc_records):
    """
    Save QC results to a CSV/TSV file. Accepts QCRecord objects or dicts.
    Overwrites rows by identity keys.

    Output columns:
      qc_task, participant_id, session_id, task_id, run_id, pipeline,
      timestamp, rater_id, rater_experience, rater_fatigue, final_qc, notes
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for rec in qc_records:
        if hasattr(rec, "model_dump"):
            rec_dict = rec.model_dump()
        elif hasattr(rec, "dict"):
            rec_dict = rec.dict()
        elif isinstance(rec, dict):
            rec_dict = rec
        else:
            continue

        rows.append(
            {
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
        )

    df_new = pd.DataFrame(rows)

    if out_file.exists():
        try:
            df_old = pd.read_csv(out_file)
        except Exception:
            df_old = pd.DataFrame()

        if not df_old.empty:
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
    else:
        df = df_new

    key_cols = [
        "participant_id",
        "session_id",
        "qc_task",
        "task_id",
        "run_id",
        "rater_id",
    ]
    existing_keys = [c for c in key_cols if c in df.columns]
    if existing_keys:
        df = df.drop_duplicates(subset=existing_keys, keep="last")

    if "participant_id" in df.columns:
        df = df.sort_values(by=["participant_id"]).reset_index(drop=True)

    df.to_csv(out_file, index=False)
    return out_file
