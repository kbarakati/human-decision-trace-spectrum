"""Participant-upload human image–spectrum decision study.
Run: python -m streamlit run app.py
"""
import gzip
import hashlib
import io
import json
import os
import pickle
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from raw_data import load_eels, load_nonlinearity, schedule, get_response, get_full_response
st.set_page_config(page_title="Human Image–Spectrum Decisions", layout="wide")
BASE = Path(__file__).resolve().parent
STATE = BASE / "private"
STATE.mkdir(exist_ok=True)
DB_PATH = STATE / "study.sqlite3"
N_CANDIDATES = 20
N_STEPS = 100
SEED = 42
PATCH_SIZE = 16
def codes():
    """Code-to-pseudonym map is a server secret, never part of shared traces."""
    try:
        secret_codes = dict(st.secrets.get("participant_codes", {}))
    except (FileNotFoundError, Exception):
        secret_codes = {}
    if secret_codes:
        return {str(code): str(pid) for pid, code in secret_codes.items()}
    env_codes = os.getenv("PARTICIPANT_CODES_JSON", "")
    if env_codes:
        return {str(code): str(pid) for pid, code in json.loads(env_codes).items()}
    return {}
def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("""CREATE TABLE IF NOT EXISTS decisions (
        participant_id TEXT NOT NULL,
        dataset_key TEXT NOT NULL,
        step INTEGER NOT NULL,
        candidate_position INTEGER NOT NULL,
        selected_index INTEGER NOT NULL,
        timestamp_utc TEXT NOT NULL,
        PRIMARY KEY (participant_id, dataset_key, step)
    )""")
    return con
def history(pid, dataset_key):
    with db() as con:
        return con.execute(
            "SELECT step,candidate_position,selected_index,timestamp_utc "
            "FROM decisions WHERE participant_id=? AND dataset_key=? ORDER BY step",
            (pid, dataset_key)
        ).fetchall()
def record(pid, dataset_key, step, pos, idx):
    with db() as con:
        con.execute("BEGIN IMMEDIATE")
        count = con.execute(
            "SELECT COUNT(*) FROM decisions WHERE participant_id=? AND dataset_key=?",
            (pid, dataset_key)
        ).fetchone()[0]
        if count != step - 1:
            con.rollback()
            return False
        con.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?)",
            (pid, dataset_key, step, pos, int(idx), datetime.now(timezone.utc).isoformat())
        )
        con.commit()
    return True
def make_image(patch, bundle):
    lo, hi = bundle["vmin"], bundle["vmax"]
    scaled = np.clip((patch - lo) / max(hi - lo, 1e-10), 0, 1)
    return Image.fromarray(np.uint8(scaled * 255)).resize((160, 160), Image.Resampling.NEAREST)
def plot_spectrum(raw, idx):
    axis, signal = get_response(raw, int(idx))
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.plot(axis, signal, lw=1)
    ax.set_xlabel(raw["axis_label"])
    ax.set_ylabel(raw["response_label"])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
def save_upload(upload, dest):
    """Write uploaded file in chunks and compute a stable dataset fingerprint."""
    digest = hashlib.sha256()
    upload.seek(0)
    with open(dest, "wb") as out:
        while True:
            chunk = upload.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()
def create_dataset(mode, files, eels_id):
    workspace = tempfile.TemporaryDirectory(prefix="manual_science_")
    folder = Path(workspace.name)
    if mode == "EELS":
        npy = folder / "measurements.npy"
        fingerprint = save_upload(files[0], npy)
        raw = load_eels(npy, eels_id)
        dataset_key = f"eels_{eels_id}"
    else:
        ibw = folder / "image.ibw"
        h5 = folder / "force_map.h5"
        image_hash = save_upload(files[0], ibw)
        force_hash = save_upload(files[1], h5)
        fingerprint = hashlib.sha256((image_hash + force_hash).encode()).hexdigest()
        raw = load_nonlinearity(ibw, h5)
        dataset_key = "nonlinearity"
    bundle = schedule(raw, n_candidates=N_CANDIDATES, n_steps=N_STEPS,
                      seed=SEED, patch_size=PATCH_SIZE)
    # Separate histories when a different upload is used.
    dataset_key += "_" + fingerprint[:16]
    return workspace, raw, bundle, dataset_key, fingerprint
def build_trace(pid, key, bundle, raw, decisions, fingerprint):
    """Analysis-ready human decisions, including every presented patch."""
    outputs = []
    for step, pos, idx, stamp in decisions:
        start = (step - 1) * N_CANDIDATES
        stop = start + N_CANDIDATES
        slot = start + pos
        expected_index = int(bundle["indices"][slot])
        if int(idx) != expected_index:
            raise ValueError(f"Stored decision {step} does not match candidate schedule")
        full = get_full_response(raw, idx)
        outputs.append({
            "step": int(step),
            "timestamp_utc": stamp,
            "candidate_indices": np.asarray(bundle["indices"][start:stop], dtype=np.int64),
            "candidate_ids": list(bundle["ids"][start:stop]),
            "candidate_coordinates": np.asarray(bundle["coordinates"][start:stop]).copy(),
            "candidate_patches": np.asarray(bundle["patches"][start:stop], dtype=np.float32).copy(),
            "selected_position": int(pos),  # 0-based, as in SV-DKL traces
            "selected_index": int(idx),
            "selected_id": str(bundle["ids"][slot]),
            "selected_coordinates": np.asarray(bundle["coordinates"][slot]).copy(),
            "selected_patch": np.asarray(bundle["patches"][slot], dtype=np.float32).copy(),
            "selected_axis": np.asarray(full["axis"]).copy(),
            "selected_spectrum": np.asarray(full["signal"]).copy(),
            "selected_raw_measurement": np.asarray(full["raw"]).copy(),
            "measurement_channels": full["channels"],
        })
    return {
        "schema_version": "human-sv-dkl-1.0",
        "participant_id": str(pid),  # pseudonymous only; no identity mapping
        "dataset": raw["dataset"],
        "dataset_key": key,
        "dataset_sha256": fingerprint,
        "eels_dataset_id": raw.get("dataset_id"),
        "grid_shape": tuple(map(int, raw["grid_shape"])),
        "image_shape": tuple(map(int, raw["image"].shape)),
        "patch_size": PATCH_SIZE,
        "n_candidates": N_CANDIDATES,
        "n_steps_planned": bundle["n_steps"],
        "steps_completed": len(outputs),
        "candidate_seed": SEED,
        "candidate_schedule": "same fixed shuffled pool for all participants",
        "axis_label": raw["axis_label"],
        "response_label": raw["response_label"],
        "decision_indexing": "step is 1-based; selected_position and measurement indices are 0-based",
        "decisions": outputs,
    }
def export_trace_pkl(pid, key, bundle, raw, decisions, fingerprint):
    return pickle.dumps(build_trace(pid, key, bundle, raw, decisions, fingerprint),
                        protocol=pickle.HIGHEST_PROTOCOL)
st.title("Manual Scientific Measurement Selection")
st.caption("20 image patches → choose one → reveal its measured response → review your accumulated history")
st.warning("Only upload research files you are authorized to use. Uploaded raw files contain all responses; this is not a blinded-data study.")
if "participant_id" not in st.session_state:
    with st.form("login"):
        code = st.text_input("Participant access code", type="password")
        go = st.form_submit_button("Enter study")
    if go:
        mapping = codes()
        if not mapping:
            st.error("Organizer must configure participant_codes in Streamlit app secrets before the study can begin.")
        elif code.strip() not in mapping:
            st.error("Invalid participant code")
        else:
            st.session_state.participant_id = mapping[code.strip()]
            st.rerun()
    st.stop()
pid = st.session_state.participant_id
st.sidebar.write(f"Participant: **{pid}**")
if st.sidebar.button("Sign out"):
    st.session_state.pop("participant_id", None)
    st.session_state.pop("dataset", None)
    st.session_state.pop("trace_download", None)
    st.session_state.pop("trace_download_identity", None)
    st.rerun()
if "dataset" not in st.session_state:
    mode = st.radio("Dataset", ["AFM nonlinearity", "EELS"], horizontal=True)
    if mode == "EELS":
        eels_id = st.selectbox("EELS dataset ID", [0, 1, 2, 3])
        npy = st.file_uploader("Upload Plasmonic_sets_hackathon.npy", type="npy")
        files = [npy] if npy is not None else []
    else:
        eels_id = None
        ibw = st.file_uploader("Upload AFM image (.ibw)", type="ibw")
        h5 = st.file_uploader("Upload AFM force map (.h5)", type="h5")
        files = [ibw, h5] if ibw is not None and h5 is not None else []
    st.caption("All participants should upload the same original files. Files are processed on the server and are not committed to GitHub. Very large uploads may exceed Streamlit Cloud memory; for those, use preprocessed or memory-mapped inputs in raw_data.py.")
    if st.button("Load dataset", type="primary", disabled=not bool(files)):
        try:
            with st.spinner("Reading scientific measurements and preparing candidates..."):
                workspace, raw, bundle, key, fingerprint = create_dataset(
                    "EELS" if mode == "EELS" else "AFM", files, eels_id
                )
            st.session_state.dataset = {"workspace": workspace, "raw": raw, "bundle": bundle, "key": key, "fingerprint": fingerprint}
            st.rerun()
        except Exception as err:
            st.error(f"Unable to read files: {err}")
    st.stop()
loaded = st.session_state.dataset
raw, bundle, key = loaded["raw"], loaded["bundle"], loaded["key"]
fingerprint = loaded["fingerprint"]
if st.sidebar.button("Load different files"):
    st.session_state.pop("dataset", None)
    st.session_state.pop("trace_download", None)
    st.session_state.pop("trace_download_identity", None)
    st.rerun()
hist = history(pid, key)
st.progress(min(len(hist) / bundle["n_steps"], 1.0))
st.write(f"**Dataset:** {key} | **Completed:** {len(hist)}/{bundle['n_steps']} | **Candidate seed:** {SEED}")
if hist:
    st.subheader("Your accumulated image–spectrum history")
    # Render only one previous measurement rather than up to 100 plots on each rerun.
    options = list(range(len(hist) - 1, -1, -1))
    selected_history = st.selectbox(
        "Inspect a previous measurement",
        options,
        format_func=lambda k: f"Round {hist[k][0]} · Location {bundle['ids'][(hist[k][0]-1)*N_CANDIDATES+hist[k][1]]}",
    )
    old_step, pos, idx, _stamp = hist[selected_history]
    slot = (old_step - 1)*N_CANDIDATES + pos
    a, b = st.columns([1, 3])
    with a:
        st.image(make_image(bundle["patches"][slot], bundle))
    with b:
        plot_spectrum(raw, idx)
    # Export is opt-in: never build all selected spectra on each interaction.
    if st.button("Prepare decision trace download"):
        with st.spinner("Building the decision trace..."):
            try:
                st.session_state["trace_download"] = export_trace_pkl(
                    pid, key, bundle, raw, hist, fingerprint
                )
                st.session_state["trace_download_identity"] = (key, len(hist))
            except (MemoryError, OSError, ValueError) as exc:
                st.error(f"Could not export trace: {exc}")
    if st.session_state.get("trace_download_identity") == (key, len(hist)):
        st.download_button(
            "Download complete coded decision trace (.pkl)",
            data=st.session_state["trace_download"],
            file_name=f"trace_{pid}_{key}.pkl",
            mime="application/octet-stream",
            on_click="ignore",
        )
next_step = len(hist)+1
if next_step > bundle["n_steps"]:
    st.success("Experiment complete. Save your coded trace using the download button above.")
    st.stop()
st.subheader(f"Decision {next_step}: Choose one of 20 image patches")
st.info("Selecting reveals only that location's measurement and adds it to your history.")
start = (next_step - 1)*N_CANDIDATES
for row in range(0, N_CANDIDATES, 5):
    cols = st.columns(5)
    for j, col in enumerate(cols):
        pos = row+j
        slot = start+pos
        with col:
            st.image(make_image(bundle["patches"][slot], bundle),
                     caption=f"Candidate {pos+1} · {bundle['ids'][slot]}",
                     width="stretch")
            if st.button(f"Select {pos+1}", key=f"selection_{key}_{next_step}_{pos}", width="stretch"):
                if not record(pid, key, next_step, pos, bundle["indices"][slot]):
                    st.warning("Round already recorded. Refreshing.")
                st.rerun()
