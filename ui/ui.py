# %%
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from layout import app

import streamlit as st

def parse_args(args=None):
    parser = ArgumentParser("QC-Studio")

    parser.add_argument(
        "--participant_list",
        dest="participant_list",
        help=("List of participants to QC"),
        required=True,
    )
    parser.add_argument(
        "--session_list",
        dest="session_list",
        help=("List of sessions to QC"),
        default="Baseline",
        required=False,
    )
    parser.add_argument(
        "--qc_pipeline",
        help=("Pipeline output to QC"),
        dest="qc_pipeline",
        required=True,
    )
    parser.add_argument(
        "--qc_task",
        help=("Specific workflow output to QC"),
        dest="qc_task",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        dest="out_dir",
        help="Directory to save session state and QC results",
        required=True,
    )
    parser.add_argument(
        "--qc_json",
        dest="qc_json",
        help=("Path to a JSON containing a list of image file paths to be displayed."),
        required=True,
    )

    return parser.parse_args(args)


args = parse_args()

participant_list = args.participant_list
session_list = args.session_list
qc_pipeline = args.qc_pipeline
qc_task = args.qc_task
qc_json = args.qc_json
out_dir = args.out_dir

current_dir = os.path.dirname(os.path.abspath(__file__))
qc_config_path = os.path.join(current_dir, qc_json)

participants_df = pd.read_csv(participant_list, delimiter="\t")

participant_ids = participants_df['participant_id'].tolist()
total_participants = len(participant_ids)

def init_session_state():
    defaults = {
        "current_page": 1,
        "batch_size": 1,
        "current_batch_qc": {},
        "qc_records": [],
        "rater_id": "",
        "rater_experience": None,
        "rater_fatigue": None,
        "notes": "",
    }
    # Initialize defaults if not already set
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

current_page = st.session_state['current_page']
if current_page < 1:
    st.session_state['current_page'] = 1
    current_page = 1

if current_page > total_participants:
    participant_id = None
else:
    participant_id = participant_ids[current_page - 1]

session_id = "ses-01"

drop_duplicates = False
app(
    participant_id=participant_id,
    session_id=session_id,
    qc_pipeline=qc_pipeline,
    qc_task=qc_task,
    qc_config_path=qc_config_path,
    out_dir=out_dir,
    total_participants=total_participants,
    drop_duplicates=drop_duplicates,
    participant_list=participant_list
)




# %%
