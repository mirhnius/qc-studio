# qc-studio

Prototyping repo for creating visual QC workflow that can be integrated with Nipoppy. 

## Goals 
- Create a web-app to visualize pipeline output including reports, svg / png images, and 3D MRI files
- Pre-define a set of files in a `<pipeline-name>_qc.json` config file that will be visualized in the web-app

## Dependencies 
- Streamlit: `pip install streamlit`
- Niivue plugin: `pip install --index-url https://test.pypi.org/simple/ --no-deps niivue-streamlit`


## Sample workflow (used files from `sample_data`)
- List 3D MRI files (i.e. pipeline output) in the `sample_fmriprep_mri_qc.json`
- List svg files (i.e. pipeline reports) in the `sample_fmriprep_svg_qc.json`
- Run test script

```
cd ui
./fmriprep_test.sh

```

## Pipelines to support (future)
- mriqc
- freesurfer
- fmriprep
- qsiprep
- qsirecon
- [agitation](https://github.com/Neuro-iX/agitation?tab=readme-ov-file)
