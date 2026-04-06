pipeline_script="ui/ui.py"
qc_pipeline="fmriprep"
qc_task="anat_wf_qc"
qc_json="../pipelines/sample_qc.json"
dataset_dir="sample_data"
participant_list="sample_data/qc_participants.tsv"
output_dir="./output"
port_number="8501"

streamlit run $pipeline_script --server.port=$port_number -- \
  --qc_json $qc_json \
  --qc_task $qc_task \
  --qc_pipeline $qc_pipeline \
  --dataset_dir $dataset_dir \
  --participant_list $participant_list \
  --output_dir $output_dir
