pipeline_script="ui/ui.py"
qc_pipeline="fsqc"
qc_task="FS_volume_wf_qc"
qc_json="../pipelines/fsqc/qc.json"
dataset_dir="/home/nikhil/projects/Parkinsons/qpn/releases/enigma/local/"
participant_list="pipelines/fsqc/participants.tsv"
output_dir="./output"
port_number="8501"

streamlit run $pipeline_script --server.port=$port_number -- \
  --qc_json $qc_json \
  --qc_task $qc_task \
  --qc_pipeline $qc_pipeline \
  --dataset_dir $dataset_dir \
  --participant_list $participant_list \
  --output_dir $output_dir
