# qc-studio

Prototyping repo for creating visual QC workflow that can be integrated with Nipoppy. [See design overview here](dev_plan.md).

## Goals 
- Create a web-app to visualize pipeline output including reports, svg / png images, and 3D MRI files
- Pre-define a set of files in a `<pipeline-name>_qc.json` config file that will be visualized in the web-app

## Dependencies 
- Streamlit: `pip install streamlit`
- Niivue plugin: `pip install --index-url https://test.pypi.org/simple/ --no-deps niivue-streamlit`

## Try example workflow (used files from `sample_data`)
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
