# Human decision trace — participant uploads

This app accepts the original EELS `.npy` file, or AFM `.ibw` and `.h5` files, directly from each participant's browser. Participants enter a code, upload authorized raw files, choose among 20 microscopy patches, see the selected response, and review their history. The app does not require `manual_dataset.pkl.gz`, `export_data.py`, or raw data checked into Git.

## Deploy

1. Push `app.py`, `raw_data.py`, `requirements.txt`, `.streamlit/config.toml`, and `.gitignore` to the GitHub repository.
2. In Streamlit Community Cloud, **Manage app > Settings > Secrets**, create a `[participant_codes]` table based on `.streamlit/secrets.toml.example` and replace every sample code with a real random passcode. Save the organizer-only code-to-name map outside GitHub.
3. Deploy `app.py`. Participants open the HTTPS link, enter their codes and upload the files.

## Critical hosting and experimental limitations

- Streamlit Community Cloud may not support very large file uploads or enough memory/disk for these files. The AFM HDF5 file is about **1.27 GB** and the EELS `.npy` around **557 MB**. The upload limit is raised to 1600 MB in `.streamlit/config.toml`, but this does **not** guarantee the hosting service will accept/process them. If uploads or app memory fail, use a more capable host or smaller approved, scientifically valid data partitions.
- Each participant receives *all* responses by uploading the complete source data. The UI hides unrevealed spectra, but the study is **not blind**. Participants must have permission to access these files.
- The SQLite decision database is stored on the **ephemeral local Streamlit server disk**, which can be lost on restart. The app provides downloadable coded `json.gz` traces after each decision. For robust group research, connect a persistent database before collecting official traces.
- Candidate schedules are identical for participants who upload the same files: 20 per step, 100 steps, seed 42. The AFM image-to-force-map mapping uses the same grid convention as the SV-DKL notebook, but physical alignment is not independently verified.
- Dataset session keys reflect type and EELS dataset ID, not cryptographic file identity. The organizer must ensure participants upload the exact same files. The app does not provide verified data provenance.
- Shared trace files include coded IDs and selected measurements; organizer name mapping must be kept separately. Timestamps can potentially reidentify participants; remove before publication if needed.

Local launch: `python -m pip install -r requirements.txt && python -m streamlit run app.py`.
