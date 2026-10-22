# Human Scientific Measurement Selection — full Pickle traces

A Streamlit app for a manual decision study. Each participant uploads authorized EELS `.npy` or AFM `.ibw` + `.h5` files, sees 20 image patches per decision, selects one, sees its measured response, and can review the accumulated image–measurement history.

## Install and run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For Streamlit Community Cloud, deploy `app.py` from the GitHub repository and set the `[participant_codes]` section using the Streamlit Cloud app **Secrets** editor. Do not commit `.streamlit/secrets.toml` or raw scientific files.

Example private secrets (use randomly generated codes, not these placeholders):

```toml
[participant_codes]
P01 = "REPLACE_WITH_RANDOM_ACCESS_CODE_1"
P02 = "REPLACE_WITH_RANDOM_ACCESS_CODE_2"
```

Preserve the mapping from `P01`–`P06` to real operator identities only in your organizer records, not in shared traces. Participant codes should be distributed privately.

## File uploads

- **AFM:** Original IBW image and HDF5 force map. The image uses `Channel_001` transposed; a measurement at `i:j` is `ForceMap/0/i:j`. The display shows channel 2 against channel 7. The Pickle trace additionally saves the entire original 8-channel (or available-channel) force curve for every selected measurement.
- **EELS:** Original `.npy` dictionary collection; select dataset ID. The display and trace use the corresponding spectral-image voxel and energy axis.

The same fixed candidate seed is used across participants; different raw upload content is fingerprinted to avoid mixing histories inadvertently. All options presented per completed decision are retained in the output `.pkl`.

## Main downloadable output

`trace_P0x_<dataset-key>.pkl`: Python dictionary with participant pseudonym, dataset ID/fingerprint, microscopy grid shape, 20 image patches per completed step, all candidate indices/IDs/coordinates, selected position, selected response and axis, full selected AFM force curves or EELS spectra, selection timestamps, and reproducible scheduling metadata. Selected-position indexing is 0-based; step number is 1-based.

Read locally:

```python
import pickle
with open('trace_P01_....pkl', 'rb') as f:
    trace = pickle.load(f)
print(trace.keys())
print(trace['decisions'][0].keys())
```

**Security:** Only load Pickle files from trusted sources. A Pickle file can execute code when unpickled. Uploaded original raw data contain all scientific responses, so this is not a blinded-data study.

**Persistence caveat:** The app's SQLite database is stored on the app's local filesystem. Streamlit Community Cloud may reset this filesystem; participants should download their trace after every session. For durable central recordkeeping across server restarts, connect an external persistent database before running a formal group study. The download is generated from the current dataset upload plus recorded selections.

**Large-upload caveat:** `.streamlit/config.toml` sets a 1600 MB upload limit, but Streamlit Cloud's RAM and storage resources may be too limited for a 1.3 GB HDF5 or 557 MB NPY upload. These limits are independent of the file-upload size setting.
