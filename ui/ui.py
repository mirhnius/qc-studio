# %%
import math
import os
import pickle
import re
import json
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from functools import wraps

import pandas as pd
import streamlit as st
from models import MetricQC, QCRecord

import os
import streamlit as st
from niivue_component import niivue_viewer


# Debug: show keys on every rerun
# st.write("Session state before initialized:", st.session_state)

st.markdown('<div id="top_of_page"></div>', unsafe_allow_html=True)

def parse_args(args=None):
    parser = ArgumentParser("fmriprep QC")

    parser.add_argument(
        "--fmri_dir",
        help=(
            "The root directory of fMRI preprocessing derivatives. "
            "For example, /SCanD_project/data/local/derivatives/fmriprep/23.2.3."
        ),
        required=False,
    )
    parser.add_argument(
        "--participant_labels",
        help=("List of participants to QC"),
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        dest="out_dir",
        help="Directory to save session state and QC results",
        required=True,
    )
    parser.add_argument(
        "--svg_list_json",
        dest="svg_list_json",
        help=("Optional path to a JSON file containing a list of SVG file paths. "
              "If provided, SVGs will be taken from this list instead of scanning --fmri_dir."),
        required=False,
    )
    parser.add_argument(
        "--mri_list_json",
        dest="mri_list_json",
        help=("Optional path to a JSON file containing a list of MRI file paths."),
        required=False,
    )
    return parser.parse_args(args)


args = parse_args()
# fs_metric = args.freesurfer_metric
participant_labels = args.participant_labels
fmri_dir = args.fmri_dir
out_dir = args.out_dir
svg_list_json = args.svg_list_json
mri_list_json = args.mri_list_json


participants_df = pd.read_csv(participant_labels, delimiter="\t")

st.title("fMRIPrep QC")
rater_name = st.text_input("Rater name:")
# Show the value dynamically
st.write("You entered:", rater_name)
 
def load_json(qc_files_json: str) -> Optional[List[Path]]:
    """
    Load a JSON file containing a list of SVG file paths.
    Validates that the top-level structure is a list.
    Returns a list of Path objects or None if no file provided.
    """
    # loaded_svg_list: Optional[List[Path]] = None
    # if svg_list_json:
    qc_files_json = Path(qc_files_json)
    if not qc_files_json.exists():
        raise FileNotFoundError(f"JSON not found: {qc_files_json}")
    try:
        with open(qc_files_json, "r") as fh:
            raw = json.load(fh)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON file {qc_files_json}: {e}")

    if not isinstance(raw, list):
        raise ValueError("JSON must contain a top-level list of file paths")

    # Convert to Path objects (do not verify existence here; downstream handles missing files)
    loaded_file_list = [Path(p) for p in raw]

    return loaded_file_list

def niivue_viewer_from_path(filepath: str, height: int = 600, key: str | None = None) -> None:
    """Load a local NIFTI file from `filepath` and display it with `niivue_viewer`.

    This helper reads the file bytes and calls the existing component.

    Args:
        filepath: Path to a local .nii or .nii.gz file.
        height: Viewer height in pixels (default 600).
        key: Optional Streamlit key for the component instance.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        file_bytes = f.read()

    if key is None:
        key = f"niivue_viewer_path_{os.path.basename(filepath)}"

    niivue_viewer(
        nifti_data=file_bytes,
        filename=os.path.basename(filepath),
        height=height,        
        key=key,
    )


# --- Niivue viewer panel (right-side) ---
# Provides an uploader and a local filepath input to display the niivue viewer
cols = st.columns([1])

with cols[0]:
    st.markdown("### Niivue Viewer")
    # st.container(height=600, border=True).markdown("This is a container with a fixed height of 600px.")
    
    # read mri_list_json if provided
    viewer_path = load_json(mri_list_json)[0] if mri_list_json else None
    print("viewer_path:", viewer_path)

    try:
        niivue_viewer_from_path(viewer_path, height=800, key=f"niivue_viewer_path_panel_{os.path.basename(viewer_path)}")
    except FileNotFoundError:
        st.error(f"File not found: {viewer_path}")
    except Exception as e:
        st.error(f"Failed to load file: {e}")


def init_session_state():
    defaults = {
        "current_page": 1,
        "batch_size": 1,
        "current_batch_qc": {},
    }
    # Initialize defaults if not already set
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# Pagination
def get_current_batch(metrics_df, current_page, batch_size):
    total_rows = len(metrics_df)
    start_idx = (current_page - 1) * batch_size
    end_idx = min(start_idx + batch_size, total_rows)
    current_batch = metrics_df.iloc[start_idx:end_idx]
    return total_rows, current_batch


def save_qc_results_to_csv(out_file, qc_records):
    """
    Save QC results from Streamlit session state to a CSV file.

    Parameters
    ----------
    out_file : str or Path
        Path where the CSV will be saved.
    qc_records : list
        List of QCRecord objects (or dicts) stored.
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Flatten metrics dynamically
    rows = []
    # qc_records = [QCRecord(**rec) for rec in qc_records_dict.values()]

    for rec in qc_records:
        row = {
            "subject": f"sub-{rec.subject_id}",
            "session": rec.session_id,
            "task": str(rec.task_id),
            "run": str(rec.run_id),
            "pipeline": rec.pipeline,
            "complete_timestamp": rec.complete_timestamp,
        }

        for m in rec.metrics:
            metric_name = m.name.lower().replace("-", "_")
            if m.value is not None:
                row[f"{metric_name}_value"] = m.value
            if m.qc is not None:
                row[f"{metric_name}"] = m.qc

        row.update(
            {
                "require_rerun": rec.require_rerun,
                "rater": rec.rater,
                "final_qc": rec.final_qc,
                "notes": next(
                    (m.notes for m in rec.metrics if m.name == "QC_notes"), None
                ),
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if out_file.exists():
        df_existing = pd.read_csv(out_file)
        df = pd.concat([df_existing, df], ignore_index=True)
        # Drop duplicates based on all columns or a subset
        df = df.drop_duplicates(
            subset=["subject", "session", "task", "run", "pipeline"], keep="last"
        )
    df = df.sort_values(by=["subject"]).reset_index(drop=True)
    df.to_csv(out_file, index=False)

    return out_file

def collect_qc_svgs(fmri_dir: str, sub_id: str, pattern: str) -> List[Path]:
    """
    Collect SVG QC figures.
    Returns:
      - For per-run QC:
          {
            "type": "perrun",
            "data": { ses -> task -> run -> [paths] }
          }
      
      - For global QC:
          {
            "type": "global",
            "data": [list of Path]
          }
    """
    base = Path(fmri_dir) / f"sub-{sub_id}" / "figures"
    search_pattern = f"sub-{sub_id}*{pattern}.svg"
    svg_paths = sorted(Path(p) for p in glob(str(base / search_pattern)))

    if not svg_paths:
        st.warning(f" No {pattern} SVGs found for subject {sub_id}")
        return {
        "type": "perrun",
        "data": {}
    }
    # Try extracting entities to see if any file contains session/run/task
    has_session = False
    has_run = False
    has_task = False
    for p in svg_paths:
        e = get_entities(p)
        if e.get("session"): has_session = True
        if e.get("run"): has_run = True
        if e.get("task"): has_task = True

    # If none of them exist → GLOBAL metric
    if not (has_session or has_task or has_run):
        return {
            "type": "global",
            "data": svg_paths
        }
    
    # Otherwise → PER-RUN metric
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in svg_paths:
        e = get_entities(p)

        ses = e.get("session") or "ses-NA"
        task = e.get("task") or "task-NA"
        run = e.get("run") or "run-NA"

        data[ses][task][run].append(p)

    return {
        "type": "perrun",
        "data": data
    }

def get_entities(filepath: Path) -> Dict[str, Optional[str]]:
    fname = filepath.name

    sub = re.search(r"sub-([a-zA-Z0-9]+)", fname)
    ses = re.search(r"ses-([a-zA-Z0-9]+)", fname)
    task = re.search(r"task-([a-zA-Z0-9]+)", fname)
    run = re.search(r"run-([0-9]+)", fname)

    return {
        "subject": sub.group(1) if sub else None,
        "session": f"ses-{ses.group(1)}" if ses else None,
        "task": task.group(1) if task else None,
        "run": f"run-{run.group(1)}" if run else None,
    }

total_rows, current_batch = get_current_batch(
    participants_df, st.session_state.current_page, st.session_state.batch_size
)

# Save to CSV
now = datetime.now()
out_file = Path(out_dir) / f"fMRIPrep_QC_status.csv"

# --- Decorator ---
def global_fallback(func):
    """
    Decorator to automatically try the global-only key (subject, None, None, None)
    if the original key returns None.
    """
    @wraps(func)
    def wrapper(sub_id, ses_id=None, task_id=None, run_id=None, metric=None):
        value = func(sub_id, ses_id, task_id, run_id,  metric)
        if value is None and metric is not None:
            # fallback to global-only
            value = func(sub_id, None, None, None, metric)
        return value
    return wrapper

def get_metrics_from_csv(qc_results: Path, metrics_to_load=None):
    """
    Load QC metrics from a CSV file. Returns a tuple:
        - list of metrics loaded
        - a function get_val(sub_id, ses_id, run_id, metric) to query the metrics

    Works with composite keys: (subject, session, run)
    """
    if metrics_to_load is None:
        metrics_to_load = [
            "surface_segmentation_qc",
            "dseg_qc",
            "sdc_qc",
            "coreg_qc",
            "require_rerun",
            "notes"
        ]

    if not qc_results.exists():
        data_dict = {}
    else:
        df_existing = pd.read_csv(qc_results)
        columns_available = [col for col in metrics_to_load if col in df_existing.columns]
        if "subject" not in df_existing.columns or not columns_available:
            data_dict = {}
        else:
            # Build dictionary with composite keys (subject, session, run)
            data_dict = {}
            for _, row in df_existing.iterrows():
                subject = row['subject']
                ses = row['session']
                if pd.isna(ses):
                    ses = None
                else:
                    ses = str(ses)
                    if not ses.startswith("ses-"): ses = f"ses-{ses}"
                run = row['run']
                if pd.isna(run):
                    run = None
                task = row['task']
                if pd.isna(task):
                    task =  None
                key = (subject, ses, task, run)
                # metrics = {col: row[col] for col in columns_available}
                metrics = {}
                for col in columns_available:
                    val= row[col]
                    if pd.isna(val):
                        metrics[col] = None
                    else:
                        metrics[col] = val
                data_dict[key] = metrics
                for col in ["dseg_qc"]:  # global-only metrics
                    global_key = (subject, None, None, None)
                    if global_key not in data_dict:
                        data_dict[global_key] = {}
                    data_dict[global_key][col] = row.get(col)

    # --- Define get_val with decorator ---
    @global_fallback
    def get_val(sub_id, ses_id=None, task_id=None, run_id=None, metric=None):
        if sub_id is None or metric is None:
            return None
        sub_key = sub_id if sub_id.startswith("sub-") else f"sub-{sub_id}"
        key = (sub_key, ses_id, task_id, run_id)
        return data_dict.get(key, {}).get(metric, None)

    return data_dict, get_val

# Load the existing qc_results if exist
data_dict, get_val = get_metrics_from_csv(out_file)

# If svg_list_json is provided, load the list of SVG paths
# loaded_svg_list: Optional[List[Path]] = None
if svg_list_json:
    print("Loading SVG list from JSON:", svg_list_json)
    loaded_svg_list = load_json(svg_list_json)
    print("Loaded SVG list from JSON:", loaded_svg_list)


def collect_qc_svgs_from_list(svg_paths_list: List[Path], sub_id: str, pattern: str) -> Dict:
    """
    Collect SVG QC figures from a provided list of paths instead of scanning the filesystem.
    Mirrors the behavior of `collect_qc_svgs` but operates on the `svg_paths_list`.
    """
    # Filter candidate SVGs for the subject and pattern
    svg_paths = [p for p in svg_paths_list if f"sub-{sub_id}" in str(p) and pattern in str(p) and str(p).endswith(".svg")]

    if not svg_paths:
        st.warning(f" No {pattern} SVGs found for subject {sub_id} (from JSON list)")
        return {
            "type": "perrun",
            "data": {}
        }

    # Try extracting entities to see if any file contains session/run/task
    has_session = False
    has_run = False
    has_task = False
    for p in svg_paths:
        e = get_entities(p)
        if e.get("session"):
            has_session = True
        if e.get("run"):
            has_run = True
        if e.get("task"):
            has_task = True

    if not (has_session or has_task or has_run):
        return {
            "type": "global",
            "data": [Path(p) for p in svg_paths]
        }

    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in svg_paths:
        e = get_entities(p)

        ses = e.get("session") or "ses-NA"
        task = e.get("task") or "task-NA"
        run = e.get("run") or "run-NA"

        data[ses][task][run].append(Path(p))

    return {
        "type": "perrun",
        "data": data
    }


def display_svg_group(
    svg_list: list[Path],
    sub_id: str,
    qc_name: str,
    metric_name: str,
    subject_metrics: list,
    ses=None,
    task=None,
    run=None
):
    """
    Display one or more SVGs with a QC radio button.
    ses, task, run are optional; include them to make Streamlit keys unique.
    """
    # with st.container():
    st.set_page_config(layout="wide")
    st.markdown(f"<h4> sub-{sub_id} - {qc_name} QC", unsafe_allow_html=True)
    options = ("PASS", "FAIL", "UNCERTAIN")

    for svg_path in svg_list:
        if not svg_path.exists():
            st.warning(f"Missing SVG: {svg_path.name}")
            continue
        with st.container():
            with open(svg_path, "r") as f:
                st.markdown(f.read(), unsafe_allow_html=True)
            st.write(f"**{svg_path.name}**")

        # Build unique Streamlit key
        key = f"{sub_id}"
        if ses: key += f"_{ses}"
        if task: key += f"_{task}"
        if run: key += f"_{run}"
        key += f"_{metric_name}"

        # Load existing value from CSV if present
        stored_val = get_val(
            sub_id=f"sub-{sub_id}",
            ses_id=ses,
            task_id=task,
            run_id=run,
            metric=metric_name
        )

        qc_choice = st.radio(
            f"{qc_name} QC:",
            options,
            key=key,
            label_visibility="collapsed",
            index=options.index(stored_val) if stored_val in options else None,
        )

        subject_metrics.append(MetricQC(
            name=metric_name,
            qc=qc_choice
        ))


# Collect all the current batch subject metrics
qc_records = []
for _, row in current_batch.iterrows():
    subj = row["participant_id"]
    parts = subj.split("_")
    sub_id = parts[0].split("-")[1]


    per_run_bundles = {}

    for pattern, qc_name, metric_name in [
        ("sdc_bold", "Susceptible Distortion Correction", "sdc_qc"),
        ("coreg_bold", "Coregistration", "coreg_qc")
    ]:
        if loaded_svg_list is not None:
            svg_bundle = collect_qc_svgs_from_list(loaded_svg_list, sub_id, pattern)
        else:
            svg_bundle = collect_qc_svgs(fmri_dir, sub_id, pattern)
        for ses, task_dict in svg_bundle["data"].items():
            for task, run_dict in task_dict.items():
                for run, svg_list in run_dict.items():
                    key = (ses, task ,run)
                    per_run_bundles.setdefault(key, []).append(
                        (svg_list, qc_name, metric_name)
                    )
    for (ses, task, run), metrics_list in sorted(per_run_bundles.items()):
        st.markdown(f"### Session: {ses} | Task: {task} | Run: {run}")
        run_metrics = []
        for svg_list, qc_name, metric_name in metrics_list:     
            display_svg_group(
                svg_list=svg_list,
                sub_id=sub_id,
                qc_name=qc_name,
                metric_name=metric_name,
                subject_metrics=run_metrics,
                ses=ses,
                task=task,
                run=run
            )
        # --- Require rerun QC radio ---
        metric = "require_rerun"
        stored_val = get_val(f"sub-{sub_id}", ses, task, run, metric)
        # stored_val = None
        options = ("YES", "NO")
        require_rerun = st.radio(
            f"Require rerun?",
            options,
            key=f"{sub_id}_{ses}_{task}_{run}_rerun",
            index=options.index(stored_val) if stored_val in options else None,
        )
        if require_rerun is None:
            final_qc = None
        else:
            final_qc = "FAIL" if require_rerun == "YES" else "PASS"
        # Notes
        metric = "notes"
        # stored_val = get_val(f"sub-{sub_id}", ses, task, run, metric)
        notes = st.text_input(f"***NOTES***", key=f"{sub_id}_{ses}_{task}_{run}_notes", value=stored_val)
        run_metrics.append(MetricQC(name="QC_notes", notes=notes))
        # all_metrics = global_metrics + run_metrics
    # st.write(subject_metrics)
    
    
    # Create QCRecord
        record = QCRecord(
            subject_id=sub_id,
            session_id=ses,
            task_id=task,
            run_id=run,
            pipeline="fmriprep-23.2.3",
            complete_timestamp= None,
            rater=rater_name,
            require_rerun=require_rerun,
            final_qc=final_qc,
            metrics=run_metrics,
        )
        qc_records.append(record)
    # st.session_state.current_batch_qc[sub_id] = record.model_dump()

# Pagination Controls - MOVED TO TOP
bottom_menu = st.columns((1, 2, 1))
# Update batch size first
with bottom_menu[2]:
    new_batch_size = st.selectbox(
        "Page Size",
        options=[1, 10, 20],
        index=(
            [1, 10, 20].index(st.session_state.batch_size)
            if st.session_state.batch_size in [1, 10, 20]
            else 0
        ),
    )

    # If batch size changed, reset to page 1
    if new_batch_size != st.session_state.batch_size:
        st.session_state.batch_size = new_batch_size
        st.session_state.current_page = 1
        st.rerun()

# Calculate total pages with current batch size
total_pages = max(1, math.ceil(total_rows / st.session_state.batch_size))
# Navigation controls

with bottom_menu[1]:
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    if col1.button("⬅️"):
        if st.session_state.current_page > 1:
            st.session_state.current_page -= 1
            st.rerun()  # Force rerun to update immediately

    new_page = col2.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=st.session_state.current_page,
        step=1,
    )

    # Update current page if changed
    if new_page != st.session_state.current_page:
        st.session_state.current_page = new_page
        st.rerun()

    if col3.button("➡️"):
        out_path = save_qc_results_to_csv(out_file, qc_records)
        if st.session_state.current_page < total_pages:
            st.session_state.current_page += 1
            st.rerun()  # Force rerun to update immediately

# add another button to save the results without changing page
with bottom_menu[2]:
    if st.button("💾 Save QC results to CSV"):
        out_path = save_qc_results_to_csv(out_file, qc_records)
        st.success(f"QC results saved to: {out_path}")

with bottom_menu[0]:
    st.markdown(f"Page **{st.session_state.current_page}** of **{total_pages}**")

# st.write("The current session state is:", qc_records)

st.write("The current session state is:", len(st.session_state))
# if st.button("Save QC results to CSV"):
#     out_path = save_qc_results_to_csv(out_file, qc_records)
#     st.success(f"QC results saved to: {out_path}")

st.markdown(
    """
    <a href="#top_of_page">
        <button style="
            background-color: white; 
            border: 1px solid #d1d5db; 
            padding: 0.5rem 1rem; 
            border-radius: 0.5rem; 
            cursor: pointer;">
            ⬆️ Scroll to Top
        </button>
    </a>
    """,
    unsafe_allow_html=True
)

# %%
