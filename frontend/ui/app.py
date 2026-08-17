import os
import time
import tempfile
import traceback
import shutil
from threading import Thread
from queue import Queue, Empty
from datetime import timedelta
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'ui.*' imports work when running as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from ui.config import (
    DEFAULT_PROMPT,
    PROMPT_MAX_CHARS,
    TARGET_MAX_SIZE_GB,
    POLL_SECONDS,
    MAX_WAIT_MIN,
)
from ui.config import get_initial_settings, persist_settings
from ui.aws_client import get_s3_client, discover_bucket_from_stack, object_exists
from ui.upload import (
    save_uploaded_to_disk,
    check_free_space,
    multipart_upload,
    copy_s3_object_to_input,
    s3_key_for_upload,
)
from ui.polling import result_key_for_input
from ui.logs import latest_log_line


st.set_page_config(page_title="Highlight Uploader", layout="centered")


def _human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def _parse_s3_uri(uri: str):
    """Parse s3://bucket/key into (bucket, key). Raises ValueError on bad input."""
    if not uri.startswith("s3://"):
        raise ValueError("Must start with s3://")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError("Provide a full s3://bucket/key")
    return bucket, key


_STATUS_ICON = {
    "queued": "⏳",
    "uploading": "⬆️",
    "processing": "⚙️",
    "done": "✅",
    "failed": "❌",
    "timeout": "⌛",
}


def _stage_to_disk(job: dict, tmpdir: str) -> str:
    """Stage a browser upload or a local-path job onto local disk for multipart upload.
    Returns the temp path. S3-URI jobs never reach here (they are copied server-side).
    """
    temp_path = os.path.join(tmpdir, os.path.basename(job["name"]))
    if job["kind"] == "local":
        lp = Path(job["src"])
        # Hardlink when possible to avoid copying multi-GB files; fall back to copy.
        try:
            os.link(str(lp), temp_path)
        except Exception:
            shutil.copy2(str(lp), temp_path)
    else:  # browser upload
        uf = job["src"]
        uf.seek(0)
        save_uploaded_to_disk(uf, temp_path)
    return temp_path


def _upload_with_progress(s3, bucket, temp_path, prompt, prog, label, dry_run):
    """Run a single multipart upload in a worker thread while updating a Streamlit
    progress bar from the main thread. Returns (key, error_text)."""
    q: Queue = Queue()
    result = {"key": None, "error": None}

    def on_progress(ps):
        # Called from s3transfer threads; never touch Streamlit here. Queue instead.
        try:
            q.put(ps, block=False)
        except Exception:
            pass

    def worker():
        try:
            result["key"] = multipart_upload(
                s3,
                bucket=bucket,
                src_path=temp_path,
                prompt=prompt,
                on_progress=on_progress,
                dry_run=dry_run,
            )
        except Exception:
            result["error"] = traceback.format_exc()

    t = Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        try:
            ps = q.get(timeout=0.2)
            eta = f"ETA {timedelta(seconds=int(ps.eta))}" if ps.eta else "Estimating…"
            prog.progress(min(ps.pct / 100.0, 1.0), text=f"{label} • {ps.pct:.1f}% • {eta}")
        except Empty:
            pass
        time.sleep(0.05)
    t.join(timeout=1)
    return result["key"], result["error"]


# -------- Sidebar settings --------
with st.sidebar:
    st.header("Settings")
    s = get_initial_settings()
    bucket = s.bucket_name
    region = s.region
    stack_name = s.stack_name
    # Dry run is no longer configurable from the UI; honor env/.env default silently
    dry_run = s.dry_run

    if not bucket:
        st.caption("Bucket not found in env; attempting CloudFormation discovery…")
        discovered = discover_bucket_from_stack(region, stack_name)
        if discovered:
            bucket = discovered
    bucket = st.text_input("S3 Bucket", value=bucket or "")
    region = st.text_input("AWS Region", value=region or "")
    stack_name = st.text_input(
        "Stack name (for CloudWatch log groups)", value=stack_name or "HighlightProcessorStack"
    )
    if st.button("Save settings"):
        persist_settings(bucket.strip() or None, region.strip() or None, stack_name.strip() or None)
        st.success("Saved. Restart not required.")

st.title("Automatic Highlight Reel – Local UI")

# -------- Upload Section --------
st.header("Upload Videos")
st.caption("Upload one or more videos at once. Every selected video is processed with the same prompt below.")

# Each source below contributes to a single flat list of jobs.
# job = {"kind": "upload"|"local"|"s3uri", "name": str, "src": <obj>, "size": Optional[int]}
jobs: list[dict] = []

col_quick, col_large = st.columns(2)

with col_quick:
    st.subheader("Quick upload")
    st.caption("Best for smaller videos. Drag and drop several at once. For multi‑GB files, use 'Large upload'.")
    uploaded_files = st.file_uploader(
        "Drag and drop or browse videos",
        type=["mp4", "mov", "mkv", "avi"],
        accept_multiple_files=True,
    )
    for uf in uploaded_files or []:
        jobs.append({"kind": "upload", "name": uf.name, "src": uf, "size": uf.size})

with col_large:
    st.subheader("Large upload (recommended for big files)")
    st.caption("Avoid browser bottlenecks by using local paths or existing S3 objects — one per line.")
    large_method = st.selectbox(
        "Choose a large upload method",
        ["Local file paths on this machine", "Existing S3 objects (s3://bucket/key)"],
        index=0,
    )

    if large_method == "Local file paths on this machine":
        local_paths = st.text_area(
            "Local video paths (one absolute path per line)", value="", height=100
        )
        use_local = st.checkbox("Use these local paths", value=False)
        if use_local:
            for line in local_paths.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    lp = Path(line).expanduser().resolve()
                    if lp.exists() and lp.is_file():
                        jobs.append(
                            {"kind": "local", "name": lp.name, "src": str(lp), "size": lp.stat().st_size}
                        )
                    else:
                        st.warning(f"Skipping (not a file): {line}")
                except Exception as e:
                    st.warning(f"Skipping invalid path '{line}': {e}")
    else:
        s3_uris = st.text_area(
            "S3 URIs (one s3://bucket/path/file.mp4 per line)", value="", height=100
        )
        use_s3_uri = st.checkbox("Use these S3 objects", value=False)
        if use_s3_uri:
            for line in s3_uris.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    src_bucket, src_key = _parse_s3_uri(line)
                    jobs.append(
                        {"kind": "s3uri", "name": os.path.basename(src_key), "src": (src_bucket, src_key), "size": None}
                    )
                except Exception as e:
                    st.warning(f"Skipping invalid S3 URI '{line}': {e}")

# Summary of everything selected across sources.
if jobs:
    total_known = sum(j["size"] for j in jobs if j["size"])
    st.info(f"Selected {len(jobs)} video(s) — total {_human_size(total_known)} (known sizes)")
    with st.expander("Selected videos", expanded=len(jobs) <= 10):
        for j in jobs:
            size = _human_size(j["size"]) if j["size"] else "size unknown"
            st.write(f"• {j['name']} — {size}  ·  _{j['kind']}_")

st.divider()

# -------- Process Section --------
st.header("Process Videos")

prompt = st.text_area(
    "Custom prompt (applies to all videos, optional)",
    value=DEFAULT_PROMPT,
    max_chars=PROMPT_MAX_CHARS,
    height=80,
)

col1, col2 = st.columns(2)
start = col1.button("Start upload & process", type="primary", disabled=not jobs)
reset = col2.button("Reset")

if reset:
    st.session_state.clear()
    st.rerun()

status = st.empty()
overall = st.progress(0.0, text="Idle")
list_box = st.empty()
log_box = st.empty()


def _render_status_list(records: list[dict]) -> None:
    lines = []
    for r in records:
        icon = _STATUS_ICON.get(r["status"], "⏳")
        note = f" — {r['note']}" if r.get("note") else ""
        lines.append(f"{icon} **{r['label']}** — {r['status']}{note}")
    list_box.markdown("\n\n".join(lines))


if start and jobs:
    if not bucket:
        st.error("Bucket is required. Set it in Settings.")
        st.stop()

    # Two videos with the same basename would collide on the same S3 input key and
    # overwrite each other. Stop early with a clear message rather than lose one.
    target_keys = [s3_key_for_upload(j["name"]) for j in jobs]
    dupes = {k for k in target_keys if target_keys.count(k) > 1}
    if dupes:
        st.error(
            "Some videos share the same filename and would overwrite each other in S3:\n"
            + "\n".join(f"• {os.path.basename(k)}" for k in sorted(dupes))
            + "\nRename them so each has a unique filename."
        )
        st.stop()

    s3 = get_s3_client(region or None)
    tmpdir = tempfile.mkdtemp(prefix="hl_upload_")

    # ---- Upload phase: process each job sequentially, one progress bar each. ----
    records: list[dict] = []
    max_bytes = TARGET_MAX_SIZE_GB * 1024 * 1024 * 1024
    status.info("Uploading videos to S3…")

    for i, job in enumerate(jobs):
        label = job["name"]
        prog = st.progress(0.0, text=f"Queued: {label}")
        rec = {"label": label, "input_key": None, "result_key": None, "status": "uploading", "note": None}
        records.append(rec)
        try:
            if job["size"] and job["size"] > max_bytes:
                raise ValueError(f"File exceeds {TARGET_MAX_SIZE_GB} GB limit.")

            if job["kind"] == "s3uri":
                src_bucket, src_key = job["src"]
                prog.progress(0.5, text=f"Copying in S3: {label}")
                input_key = copy_s3_object_to_input(
                    s3,
                    dest_bucket=bucket,
                    source_bucket=src_bucket,
                    source_key=src_key,
                    prompt=prompt.strip() or None,
                )
            else:
                temp_path = _stage_to_disk(job, tmpdir)
                if not check_free_space(temp_path, os.path.getsize(temp_path) * 2):
                    raise RuntimeError("Insufficient disk space for staging upload.")
                input_key, err = _upload_with_progress(
                    s3, bucket, temp_path, prompt.strip() or None, prog, f"Uploading {label}", dry_run
                )
                if err:
                    raise RuntimeError(err)

            prog.progress(1.0, text=f"Uploaded: {label}")
            rec["input_key"] = input_key
            rec["result_key"] = result_key_for_input(input_key)
            rec["status"] = "processing"
        except Exception as e:
            prog.progress(1.0, text=f"Failed: {label}")
            rec["status"] = "failed"
            rec["note"] = str(e).strip().splitlines()[-1] if str(e).strip() else "upload failed"

        overall.progress((i + 1) / len(jobs), text=f"Uploaded {i + 1}/{len(jobs)}")

    _render_status_list(records)

    # ---- Processing phase: poll each result object until done or timeout. ----
    pending = [r for r in records if r["status"] == "processing"]
    if pending:
        status.info(f"Processing {len(pending)} video(s) in backend… This can take several minutes each.")
        stack = stack_name or os.getenv("STACK_NAME", "HighlightProcessorStack")
        log_groups = [
            f"/aws/lambda/{stack}-VideoTriggerLambda",
            f"/ecs/video-processor-{stack}",
        ]
        deadline = time.time() + MAX_WAIT_MIN * 60

        while any(r["status"] == "processing" for r in records):
            for r in records:
                if r["status"] != "processing":
                    continue
                try:
                    if object_exists(s3, bucket, r["result_key"]):
                        r["status"] = "done"
                except Exception:
                    pass

            line = latest_log_line(region or None, log_groups)
            if line:
                log_box.caption(f"Last log: {line.strip()}")

            done = sum(1 for r in records if r["status"] in ("done", "failed", "timeout"))
            overall.progress(done / len(records), text=f"Completed {done}/{len(records)}")
            _render_status_list(records)

            if all(r["status"] != "processing" for r in records):
                break
            if time.time() > deadline:
                for r in records:
                    if r["status"] == "processing":
                        r["status"] = "timeout"
                        r["note"] = "timed out waiting for result; check CloudWatch logs"
                _render_status_list(records)
                break
            time.sleep(POLL_SECONDS)

    # ---- Results: preview + download per finished video. ----
    n_done = sum(1 for r in records if r["status"] == "done")
    if n_done == len(records):
        status.success(f"All {n_done} highlight video(s) ready!")
    else:
        status.warning(f"{n_done}/{len(records)} completed. See details below.")

    st.subheader("Results")
    for r in records:
        icon = _STATUS_ICON.get(r["status"], "⏳")
        with st.expander(f"{icon} {r['label']} — {r['status']}", expanded=r["status"] == "done"):
            if r["status"] == "done":
                s3_path = f"s3://{bucket}/{r['result_key']}"
                st.write(s3_path)
                try:
                    # Presigned URL avoids loading every result fully into memory.
                    url = s3.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": bucket, "Key": r["result_key"]},
                        ExpiresIn=3600,
                    )
                    st.video(url)
                    st.markdown(f"[Download {os.path.basename(r['result_key'])}]({url})")
                except Exception:
                    st.info("Preview not available. Use the S3 path above.")
            elif r["status"] == "failed":
                st.error(r.get("note") or "Upload failed.")
            elif r["status"] == "timeout":
                st.warning(r.get("note") or "Timed out. Verify permissions and check CloudWatch logs.")
            else:
                st.info("Still processing.")
