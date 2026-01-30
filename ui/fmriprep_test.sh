pipeline_script="ui.py"
data_dir="../sample_data/derivatives/fmriprep/23.1.3/output/"
svg_list_json="sample_fmriprep_svg_qc.json"
mri_list_json="sample_fmriprep_mri_qc.json"
participant_labels_tsv="qc_participants.tsv"
output_dir="../output"
port_number="8501"

streamlit run $pipeline_script --server.port=$port_number -- \
  --svg_list_json $svg_list_json \
  --mri_list_json $mri_list_json \
  --participant_labels $participant_labels_tsv \
  --output_dir $output_dir
