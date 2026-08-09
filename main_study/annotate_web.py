"""
annotate_web.py
---------------
Browser-based annotation tool for V2P study trials.

    python annotate_web.py
    python annotate_web.py --port 8765
    python annotate_web.py --no-browser

CHANGELOG (this revision):
  - do_GET / do_POST now wrap all logic in try/except and return a JSON
    error body (with traceback) instead of letting the handler die
    silently. Previously an unhandled exception on a real-data edge case
    (malformed CSV row, unexpected PID format, etc.) would kill the
    request without a response, which the frontend fetch() never
    resolves/rejects cleanly -> page appears to "load forever".
  - Frontend fetch() now has an 8s AbortController timeout with a visible
    error message, as a second line of defense against silent hangs.
  - Gaze_alarm restructured to mirror Stopped_at_Alarm: instead of a vague
    glanced/looked intensity scale, it now tracks WHICH alarm triggered a
    gaze shift (no / f / vha / both for Adaptive, no / vha for Baseline).
    The separate Look_Reaction field was removed -- Stopped_at_Alarm +
    Gaze_alarm + free-text Note cover what's needed without an overlapping
    third category.
  - PID Summary / Overview panels now break down Gaze_alarm distribution
    instead of the removed Look_Reaction.
"""

import os, csv, re, glob, shutil, json, argparse, threading, webbrowser, traceback
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

LOG_DIR      = "logs"
OUTPUT_FILE  = "trial_annotations.csv"
PARTICIPANT_FILE = "participant_profile.csv"   # per-PID Walker_speed/Walking_style (not per-trial)
DEFAULT_PORT = 8765

HEADER_RE   = re.compile(r"#\s*PID:(\S+)\s+Mode:(\S+)\s+Run:(\S+)\s+Outcome:(\S+)")
# Order matches how fields are reviewed: outcome, general gaze, alarm
# reaction/intensity, whether they stopped, whether/which alarm made them
# look, free-text note last.
ALL_COLUMNS = [
    "Actual_Outcome", "Gaze_general", "Alarm_reaction",
    "Stopped_at_Alarm", "Gaze_alarm",
    "Note",
]
PARTICIPANT_FIELDS = ["Walker_speed", "Walking_style"]
FIELDNAMES  = ["PID", "Mode", "Run", "Logged_Outcome"] + ALL_COLUMNS
TRIAL_ORDER = ["t1", "t2", "t3", "t4", "t5", "c1", "c2", "s1", "s2"]

# Migration map: old Looked_at_Screen values -> new Gaze_general values
LEGACY_GAZE_MAP = {
    "screen_glanced": "phone", "screen_looked": "phone",
    "traffic_glanced": "traffic", "traffic_looked": "traffic",
    "both": "both", "glanced": "phone", "looked": "phone",
    "focused": "phone", "yes": "phone",
}

OPTIONS = {
    "outcome":          ["collision", "response_stop", "response_run",
                         "response_run_back", "response_continue_walk",
                         "not_in_time", "safe_stop", "safe_turn"],
    "stopped_adaptive": ["no", "f", "vha", "both"],
    "stopped_baseline": ["no", "vha"],
    "gaze_general":     ["phone", "traffic", "both", "neither"],
    # Which alarm triggered a shift in gaze (mirrors stopped_adaptive/baseline,
    # since "did they look because of f / vha / both" is the meaningful
    # distinction here, not a vague glanced/looked intensity scale).
    "gaze_alarm_adaptive": ["no", "f", "vha", "both"],
    "gaze_alarm_baseline": ["no", "vha"],
    "walker_speed":     ["slow", "normal", "fast"],
    "walking_style":    ["constant", "dynamic"],
    "alarm_reaction":   ["none", "mild", "strong", "jumped"],
}


# ─────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────

def parse_header(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            line = f.readline()
    except Exception:
        return None
    m = HEADER_RE.search(line)
    if not m:
        return None
    return {"PID": m.group(1), "Mode": m.group(2),
            "Run": m.group(3), "Logged_Outcome": m.group(4)}


def load_run_order(folder):
    """Read <folder>/run_order.csv (written by generate_run_order()) and return
    {mode: [run_codes_in_session_order]}. Only the 'B'/'A' lines are used (the
    authoritative, cleaned order); '*_raw' lines are ignored. Returns {} if the
    file is missing or unparsable -- callers should treat that as 'no order
    info available' rather than an error."""
    path = os.path.join(folder, "run_order.csv")
    if not os.path.exists(path):
        return {}
    order = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or "_raw" in line:
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                label, rest = parts
                mode = {"B": "b", "A": "a"}.get(label)
                if not mode:
                    continue
                order[mode] = [r.strip() for r in rest.split(",") if r.strip()]
    except Exception as e:
        print(f"[WARN] Could not read run_order.csv in {folder}: {e}")
    return order


def load_all_trials():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "P*", "*.csv")))
    trials = []
    run_order_cache = {}  # folder path -> {mode: [run_codes]}, avoid re-reading per file
    for fp in files:
        d = parse_header(fp)
        if d:
            folder = os.path.dirname(fp)
            if folder not in run_order_cache:
                run_order_cache[folder] = load_run_order(folder)
            order_list = run_order_cache[folder].get(d["Mode"], [])
            d["Session_Order"] = order_list.index(d["Run"]) if d["Run"] in order_list else None
            trials.append(d)

    # Fallback: if no log files found, reconstruct trial list from annotations CSV
    if not trials and os.path.exists(OUTPUT_FILE):
        print(f"[INFO] No log files found under '{LOG_DIR}/'. "
              f"Building trial list from {OUTPUT_FILE}.")
        try:
            with open(OUTPUT_FILE, "r", newline="", encoding="utf-8-sig") as f:
                raw = f.read()
            try:
                dialect = csv.Sniffer().sniff(raw[:2048], delimiters=",;\t")
            except Exception:
                dialect = csv.excel
            reader = csv.DictReader(raw.splitlines(), dialect=dialect)
            for row in reader:
                def g(k):
                    return (row.get(k) or row.get(k.lower()) or "").strip()
                pid  = g("PID")
                mode = g("Mode")
                run  = g("Run")
                logged = g("Logged_Outcome")
                if pid and mode and run:
                    # No run_order.csv available in this fallback path (no
                    # per-participant folder to read it from).
                    trials.append({"PID": pid, "Mode": mode,
                                   "Run": run, "Logged_Outcome": logged,
                                   "Session_Order": None})
        except Exception as e:
            print(f"[WARN] Could not build fallback trial list: {e}")

    def key(t):
        return (int(re.sub(r"\D", "", t["PID"]) or 0),
                0 if t["Mode"] == "a" else 1,
                TRIAL_ORDER.index(t["Run"]) if t["Run"] in TRIAL_ORDER else 99)
    return sorted(trials, key=key)


def clean_header_name(name):
    """Strip leading/trailing whitespace and any trailing run of non-word
    characters (semicolons, commas, stray punctuation) from a CSV header
    name. Files edited by hand or exported by other tools sometimes end up
    with a header like 'Note;' instead of 'Note' -- an exact-match lookup
    against 'Note' then silently finds nothing and that whole column's data
    gets dropped on the next save. This makes column-name matching robust
    to that kind of trailing junk."""
    return re.sub(r"[^\w]+$", "", name.strip())


def load_annotations():
    data = {}
    if not os.path.exists(OUTPUT_FILE):
        return data
    try:
        with open(OUTPUT_FILE, "r", newline="", encoding="utf-8-sig") as f:
            raw = f.read()
        try:
            dialect = csv.Sniffer().sniff(raw[:2048], delimiters=",;\t")
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(raw.splitlines(), dialect=dialect)
        rows   = list(reader)
        first_col = (reader.fieldnames or [""])[0]
        if rows and not rows[0].get("Mode") and "," in (rows[0].get(first_col) or ""):
            # Sniffer picked the wrong delimiter and the whole header/row ended
            # up crammed into a single field. Recover the REAL column names
            # from that crammed header itself (not from the current ALL_COLUMNS),
            # so this stays correct regardless of which schema version the file
            # was written with. Using a hardcoded, possibly-longer column list
            # here previously caused a silent off-by-one that dropped the last
            # column (Note) once a new field was added to ALL_COLUMNS.
            col_names = [c.strip() for c in first_col.split(",")]
            new_rows  = []
            for row in rows:
                parts = (row.get(first_col) or "").split(",", len(col_names)-1)
                new_rows.append(dict(zip(col_names, [p.strip().strip('"') for p in parts])))
            rows = new_rows

        # Normalize header names (e.g. 'Note;' -> 'Note') so lookups below
        # match regardless of stray trailing punctuation in the source file.
        rows = [{clean_header_name(k): v for k, v in row.items()} for row in rows]

        for row in rows:
            def g(*keys):
                for k in keys:
                    v = (row.get(k) or row.get(k.lower()) or "").strip()
                    v = v.rstrip(";").strip()  # trailing ';' is a known export/edit artifact, never real content
                    if v: return v
                return ""
            pid, mode, run = g("PID"), g("Mode"), g("Run")
            if not (pid and mode and run):
                continue
            ann = {col: g(col) for col in ALL_COLUMNS}
            # Migrate old Looked_at_Screen -> Gaze_general
            if not ann["Gaze_general"]:
                old = g("Looked_at_Screen", "Looked_At_Screen", "looked_at_screen")
                if old:
                    ann["Gaze_general"] = LEGACY_GAZE_MAP.get(old.lower(), "")
            data[(pid, mode, run)] = ann
    except Exception as e:
        print(f"[WARN] {e}")
    return data


def apply_defaults(trials, annotations):
    """Pre-fill Actual_Outcome=logged and Stopped=no for unannotated catch/safe
    trials. Independently pre-fill Alarm_reaction=none and Gaze_alarm=no for
    safe_turn trials (any mode) and safe_stop trials (baseline only) --
    adaptive safe_stop is excluded since an early warning could still
    plausibly fire there. Each fill only ever touches a genuinely empty
    field, never overwrites an existing annotation."""
    changed = False
    for t in trials:
        if not t["Run"].startswith(("c", "s")):
            continue
        key = (t["PID"], t["Mode"], t["Run"])
        ann = annotations.get(key, {col: "" for col in ALL_COLUMNS})
        trial_changed = False
        if not ann.get("Actual_Outcome"):
            ann["Actual_Outcome"]   = t["Logged_Outcome"]
            ann["Stopped_at_Alarm"] = "no"
            trial_changed = True
        outcome = ann.get("Actual_Outcome") or t["Logged_Outcome"]
        no_alarm_expected = outcome == "safe_turn" or (outcome == "safe_stop" and t["Mode"] == "b")
        if not ann.get("Alarm_reaction") and no_alarm_expected:
            ann["Alarm_reaction"] = "none"
            trial_changed = True
        if not ann.get("Gaze_alarm") and no_alarm_expected:
            ann["Gaze_alarm"] = "no"
            trial_changed = True
        if trial_changed:
            annotations[key] = ann
            changed = True
    return changed


def save_all(trials, annotations):
    if os.path.exists(OUTPUT_FILE):
        shutil.copy2(OUTPUT_FILE, OUTPUT_FILE + ".bak")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for t in trials:
            key = (t["PID"], t["Mode"], t["Run"])
            ann = annotations.get(key, {col: "" for col in ALL_COLUMNS})
            writer.writerow({"PID": t["PID"], "Mode": t["Mode"],
                             "Run": t["Run"], "Logged_Outcome": t["Logged_Outcome"],
                             **ann})


def load_participant_profiles():
    """Read participant_profile.csv -> {pid: {Walker_speed, Walking_style}}."""
    data = {}
    if not os.path.exists(PARTICIPANT_FILE):
        return data
    try:
        with open(PARTICIPANT_FILE, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = (row.get("PID") or "").strip()
                if not pid:
                    continue
                data[pid] = {col: (row.get(col) or "").strip() for col in PARTICIPANT_FIELDS}
    except Exception as e:
        print(f"[WARN] Could not read {PARTICIPANT_FILE}: {e}")
    return data


def save_participant_profiles(profiles):
    if os.path.exists(PARTICIPANT_FILE):
        shutil.copy2(PARTICIPANT_FILE, PARTICIPANT_FILE + ".bak")
    with open(PARTICIPANT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["PID"] + PARTICIPANT_FIELDS)
        writer.writeheader()
        for pid, vals in sorted(profiles.items(), key=lambda kv: int(re.sub(r"\D","",kv[0]) or 0)):
            writer.writerow({"PID": pid, **{c: vals.get(c, "") for c in PARTICIPANT_FIELDS}})


def migrate_participant_speed_style():
    """One-time migration: Walker_speed/Walking_style used to live on every
    trial row in trial_annotations.csv. Now they're per-participant, in
    PARTICIPANT_FILE. If that file doesn't exist yet but the old trial file
    does, pull the first non-empty value per PID out of the old file before
    it's silently ignored (ALL_COLUMNS no longer includes those columns, so
    without this they'd just be dropped on the next save)."""
    if os.path.exists(PARTICIPANT_FILE) or not os.path.exists(OUTPUT_FILE):
        return
    try:
        with open(OUTPUT_FILE, "r", newline="", encoding="utf-8-sig") as f:
            raw = f.read()
        try:
            dialect = csv.Sniffer().sniff(raw[:2048], delimiters=",;\t")
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(raw.splitlines(), dialect=dialect)
        rows = list(reader)
        first_col = (reader.fieldnames or [""])[0]
        if rows and not rows[0].get("Mode") and "," in (rows[0].get(first_col) or ""):
            # same crammed-header recovery as load_annotations()
            col_names = [c.strip() for c in first_col.split(",")]
            rows = [dict(zip(col_names,
                    [p.strip().strip('"') for p in (row.get(first_col) or "").split(",", len(col_names)-1)]))
                    for row in rows]
        rows = [{clean_header_name(k): v for k, v in row.items()} for row in rows]
        profiles = {}
        for row in rows:
            def g(*keys):
                for k in keys:
                    v = (row.get(k) or row.get(k.lower()) or "").strip()
                    v = v.rstrip(";").strip()
                    if v: return v
                return ""
            pid = g("PID")
            if not pid or pid in profiles:
                continue
            speed, style = g("Walker_speed"), g("Walking_style")
            if speed or style:
                profiles[pid] = {"Walker_speed": speed, "Walking_style": style}
        if profiles:
            save_participant_profiles(profiles)
            print(f"[INFO] Migrated Walker_speed/Walking_style for {len(profiles)} "
                  f"participant(s) from {OUTPUT_FILE} into {PARTICIPANT_FILE}.")
    except Exception as e:
        print(f"[WARN] Could not migrate participant speed/style: {e}")


# ─────────────────────────────────────────────────────────
# HTTP SERVER
# ─────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.trials      = load_all_trials()
        self.annotations = load_annotations()
        migrate_participant_speed_style()
        self.participants = load_participant_profiles()
        if apply_defaults(self.trials, self.annotations):
            try:
                save_all(self.trials, self.annotations)
            except PermissionError:
                print(f"\n  [WARN] Could not write '{OUTPUT_FILE}' -- it looks like the "
                      f"file is currently open in Excel or another program.\n"
                      f"         Close it and restart this script, or your edits in the "
                      f"browser won't be saved to disk.\n"
                      f"         Continuing to start the server anyway so you can look "
                      f"around; annotations will only be kept in memory until the file "
                      f"can be written.\n")
            except OSError as e:
                print(f"\n  [WARN] Could not write '{OUTPUT_FILE}': {e}\n"
                      f"         Continuing to start the server anyway; annotations will "
                      f"only be kept in memory until this is resolved.\n")

state = AppState()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    # ── NEW: catch-all error responder ─────────────────────
    # Without this, an unhandled exception mid-request kills the
    # connection with no response at all, and the frontend fetch()
    # just hangs (looks like "infinite loading" in the browser).
    def send_error_json(self, exc):
        try:
            body = json.dumps({
                "error": str(exc),
                "trace": traceback.format_exc(),
            }).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except Exception:
            pass  # connection already broken, nothing more we can do

    # Client-side disconnects (browser closed the tab, reloaded mid-request,
    # antivirus/firewall interference, etc.) are normal network noise, not
    # bugs in our logic. Trying to write an error response back on a socket
    # the client already tore down just raises a second exception, so we
    # log a one-liner and stop -- no traceback, no response attempt.
    CLIENT_DISCONNECT_ERRORS = (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)

    def do_GET(self):
        try:
            path = urlparse(self.path).path
            if path == "/":
                self.send_html(HTML)
            elif path == "/api/data":
                pids = sorted({t["PID"] for t in state.trials},
                              key=lambda p: int(re.sub(r"\D","",p) or 0))
                by_pid = {}
                for t in state.trials:
                    pid, mode = t["PID"], t["Mode"]
                    key  = (pid, mode, t["Run"])
                    ann  = state.annotations.get(key, {col: "" for col in ALL_COLUMNS})
                    by_pid.setdefault(pid, {}).setdefault(mode, []).append({**t, **ann})
                self.send_json({"pids": pids, "trials": by_pid, "options": OPTIONS,
                                 "participants": state.participants})
            else:
                self.send_response(404); self.end_headers()
        except self.CLIENT_DISCONNECT_ERRORS:
            print(f"[INFO] Client disconnected during GET {self.path} (harmless, e.g. reload/abort)")
        except Exception as e:
            print(f"[ERROR] GET {self.path}: {e}\n{traceback.format_exc()}")
            self.send_error_json(e)

    def do_POST(self):
        try:
            path = urlparse(self.path).path
            if path == "/api/save":
                length = int(self.headers.get("Content-Length", 0))
                body   = json.loads(self.rfile.read(length))
                pid, mode, run = body["pid"], body["mode"], body["run"]
                key = (pid, mode, run)
                ann = state.annotations.get(key, {col: "" for col in ALL_COLUMNS}).copy()
                for col in ALL_COLUMNS:
                    if col in body:
                        ann[col] = body[col]
                state.annotations[key] = ann
                save_all(state.trials, state.annotations)
                self.send_json({"ok": True})
            elif path == "/api/save_participant":
                length = int(self.headers.get("Content-Length", 0))
                body   = json.loads(self.rfile.read(length))
                pid = body["pid"]
                prof = state.participants.get(pid, {c: "" for c in PARTICIPANT_FIELDS}).copy()
                for col in PARTICIPANT_FIELDS:
                    if col in body:
                        prof[col] = body[col]
                state.participants[pid] = prof
                save_participant_profiles(state.participants)
                self.send_json({"ok": True})
            else:
                self.send_response(404); self.end_headers()
        except self.CLIENT_DISCONNECT_ERRORS:
            print(f"[INFO] Client disconnected during POST {self.path} (harmless, e.g. reload/abort)")
        except Exception as e:
            print(f"[ERROR] POST {self.path}: {e}\n{traceback.format_exc()}")
            self.send_error_json(e)


# ─────────────────────────────────────────────────────────
# HTML / CSS / JS
# ─────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>V2P Annotation Tool</title>
<style>
:root {
  --bg:#f0f2f5; --card:#fff; --border:#dde1e7;
  --accent-a:#2563eb; --accent-b:#7c3aed;
  --text:#111827; --muted:#6b7280; --muted2:#9ca3af;
  --green:#16a34a; --row-empty:#fffbeb; --row-done:#f0fdf4; --row-diff:#eff6ff;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--bg);color:var(--text);font-size:14px}

header{background:#fff;border-bottom:1px solid var(--border);
       padding:14px 24px;display:flex;align-items:center;gap:16px}
header h1{font-size:17px;font-weight:700}
#subtitle{color:var(--muted);font-size:13px}

.pid-tabs{display:flex;flex-wrap:wrap;gap:6px 6px;padding:16px 24px 0}
.pid-tab{padding:7px 18px;border-radius:8px 8px 0 0;border:1px solid var(--border);
         border-bottom:none;background:#e5e7eb;cursor:pointer;font-weight:600;
         font-size:13px;color:var(--muted);transition:all .12s}
.pid-tab.active{background:#fff;color:var(--text);border-bottom:1px solid #fff;
                margin-bottom:-1px;z-index:1;position:relative}
.pid-tab:hover:not(.active){background:#d1d5db}
.pid-tab.overview-tab{background:#eef2ff;color:#4338ca}
.pid-tab.overview-tab.active{background:#fff;color:#4338ca}
.pid-tab.overview-tab:hover:not(.active){background:#e0e7ff}

.content{padding:0 24px 40px}
.section{background:#fff;border:1px solid var(--border);
         border-radius:0 8px 8px 8px;margin-bottom:20px;overflow:hidden}
.sec-hdr{padding:11px 16px;font-weight:700;font-size:12px;
         letter-spacing:.06em;text-transform:uppercase}
.sec-hdr.adaptive{background:#eff6ff;color:var(--accent-a);border-bottom:1px solid #bfdbfe}
.sec-hdr.baseline{background:#f5f3ff;color:var(--accent-b);border-bottom:1px solid #ddd6fe}

.stats-bar{display:flex;flex-wrap:wrap;gap:16px;padding:8px 16px;
           background:#f9fafb;border-bottom:1px solid var(--border);
           font-size:12px;color:var(--muted)}
.stats-bar strong{color:var(--text)}

table{width:100%;border-collapse:collapse}
th{padding:7px 8px;text-align:left;font-weight:600;font-size:11px;
   color:var(--muted);background:#f9fafb;border-bottom:1px solid var(--border);
   white-space:nowrap}
td{padding:5px 6px;border-bottom:1px solid #f3f4f6;vertical-align:middle}
tr:last-child td{border-bottom:none}
tr.row-empty{background:var(--row-empty)}
tr.row-done {background:var(--row-done)}
tr.row-diff {background:var(--row-diff)}

/* profile sub-row */
tr.sub-row td{background:#f8fafc;border-bottom:1px solid #e5e7eb;padding:4px 8px 6px 28px}
tr.sub-row:last-child td{border-bottom:none}
.profile-bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.profile-bar .lbl{font-size:11px;color:var(--muted2);white-space:nowrap}

.badge{display:inline-block;padding:2px 8px;border-radius:4px;
       font-weight:700;font-size:11px;font-family:monospace}
.badge-t{background:#fee2e2;color:#991b1b}
.badge-c{background:#fef3c7;color:#92400e}
.badge-s{background:#dcfce7;color:#166534}

.logged-pill{display:inline-block;padding:2px 7px;background:#f3f4f6;
             border-radius:4px;font-size:12px;color:var(--muted)}

select,input[type=text]{
  border:1px solid var(--border);border-radius:5px;padding:4px 6px;
  font-size:12px;background:#fff;color:var(--text);outline:none;transition:border-color .12s}
select:focus,input[type=text]:focus{border-color:#6366f1}
select.ph{color:var(--muted)}

.s-outcome{width:142px} .s-stopped{width:106px}
.s-gaze-g{width:110px}  .s-gaze-a{width:90px}
.s-speed{width:80px}    .s-style{width:88px}
.s-react{width:84px}    .inp-note{width:240px}

.dot{font-size:14px;display:block;text-align:center}
.dot-yes{color:var(--green)}.dot-no{color:#d1d5db}

/* PID summary panel */
.summary-card{background:#fff;border:1px solid var(--border);border-radius:8px;
              margin-bottom:20px;overflow:hidden}
.summary-hdr{padding:11px 16px;font-weight:700;font-size:12px;letter-spacing:.06em;
             text-transform:uppercase;background:#f9fafb;color:var(--text);
             border-bottom:1px solid var(--border)}
.summary-body{display:grid;grid-template-columns:1fr 1fr;gap:0}
.summary-col{padding:14px 16px}
.summary-col + .summary-col{border-left:1px solid var(--border)}
.summary-col h3{font-size:12px;font-weight:700;margin-bottom:8px}
.summary-col.adaptive h3{color:var(--accent-a)}
.summary-col.baseline h3{color:var(--accent-b)}
.summary-progress{font-size:12px;color:var(--muted);margin-bottom:10px}
.summary-progress .bar{height:6px;background:#e5e7eb;border-radius:3px;overflow:hidden;margin-top:3px}
.summary-progress .fill{height:100%;background:var(--green)}
.summary-row{display:flex;justify-content:space-between;font-size:12px;
            padding:3px 0;border-bottom:1px dashed #f0f0f0}
.summary-row:last-child{border-bottom:none}
.summary-row .k{color:var(--muted)}
.summary-row .v{font-weight:600}
.summary-empty{grid-column:1/-1;padding:16px;text-align:center;color:var(--muted);font-size:12px}

#toast{position:fixed;bottom:24px;right:24px;background:#111827;color:#fff;
       padding:9px 16px;border-radius:7px;font-size:13px;opacity:0;
       transition:opacity .18s;pointer-events:none;z-index:999}
#toast.show{opacity:1}
.empty-msg{padding:40px;text-align:center;color:var(--muted)}
</style>
</head>
<body>

<header>
  <div>
    <h1>V2P · Annotation Tool</h1>
    <div id="subtitle">Loading…</div>
  </div>
</header>
<div id="pid-tabs" class="pid-tabs"></div>
<div class="content" id="content">
  <div class="empty-msg">Loading trials…</div>
</div>
<div id="toast"></div>

<script>
const OUT_LABELS = {
  collision:"collision", response_stop:"response stop",
  response_run:"response run", response_run_back:"response run back",
  response_continue_walk:"continued walking (no response)",
  not_in_time:"not in time", safe_stop:"safe stop", safe_turn:"safe turn",
};
const STOP_LABELS  = {no:"no", f:"f alarm", vha:"vha alarm", both:"both alarms"};
const GAZE_G_LABELS = {
  phone:"mainly phone", traffic:"mainly traffic",
  both:"both", neither:"neither",
};
// Gaze (alarm) now shares the same value domain as Stopped_at_Alarm
// (no / f / vha / both -- which alarm triggered a gaze shift), so it
// reuses STOP_LABELS directly instead of a separate label map.
const SPEED_LABELS  = {slow:"slow", normal:"normal", fast:"fast"};
const STYLE_LABELS  = {constant:"constant speed", dynamic:"dynamic"};
const REACT_LABELS  = {none:"none", mild:"mild", strong:"strong", jumped:"jumped"};

let app=null, pid=null, saveTimer=null, toastTimer=null, sortMode="type";

async function init() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);
  try {
    const r = await fetch("/api/data", { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!r.ok) {
      const errBody = await r.json().catch(() => ({}));
      throw new Error(errBody.error || ("HTTP " + r.status));
    }
    app = await r.json();
    app.participants = app.participants || {};
    renderTabs();
    if (app.pids.length) {
      chooseView("__overview__");
    } else {
      document.getElementById("content").innerHTML =
        '<div class="empty-msg">No trials found.<br><br>' +
        'Make sure you run <code>python annotate_web.py</code> from your study folder ' +
        '(the one containing <code>logs/</code> or <code>trial_annotations.csv</code>).</div>';
    }
    updateSubtitle();
  } catch(e) {
    clearTimeout(timeoutId);
    const msg = e.name === "AbortError"
      ? "Server did not respond within 8s. Check the terminal running annotate_web.py for a traceback."
      : e.message;
    document.getElementById("content").innerHTML =
      '<div class="empty-msg">Connection error: ' + msg + '<br><br>' +
      '<button onclick="init()" style="margin-top:8px;padding:6px 14px;cursor:pointer;">Retry</button></div>';
  }
}

function updateSubtitle() {
  let total=0, done=0;
  for (const p of app.pids)
    for (const m of Object.values(app.trials[p]||{}))
      for (const t of m) { total++; if(t.Actual_Outcome) done++; }
  document.getElementById("subtitle").textContent =
    done + " / " + total + " trials annotated";
}

function renderTabs() {
  const el = document.getElementById("pid-tabs");
  el.innerHTML = "";
  const overviewBtn = document.createElement("button");
  overviewBtn.className = "pid-tab overview-tab";
  overviewBtn.textContent = "\u{1F4CA} Overview";
  overviewBtn.dataset.pid = "__overview__";
  overviewBtn.onclick = () => chooseView("__overview__");
  el.appendChild(overviewBtn);
  for (const p of app.pids) {
    const b = document.createElement("button");
    b.className = "pid-tab"; b.textContent = "P"+p; b.dataset.pid = p;
    b.onclick = () => chooseView(p);
    el.appendChild(b);
  }
}

function chooseView(p) {
  pid = p;
  document.querySelectorAll(".pid-tab")
    .forEach(b => b.classList.toggle("active", b.dataset.pid === p));
  if (p === "__overview__") {
    renderOverview();
  } else {
    renderContent();
  }
}

function badgeCls(run) {
  return run.startsWith("t") ? "badge-t" : run.startsWith("c") ? "badge-c" : "badge-s";
}
function rowCls(t) {
  if (!t.Actual_Outcome) return "row-empty";
  return t.Actual_Outcome !== t.Logged_Outcome ? "row-diff" : "row-done";
}

function mkSel(opts, labels, current, cls, onChange) {
  const sel = document.createElement("select");
  sel.className = cls + (current ? "" : " ph");
  const ph = document.createElement("option");
  ph.value=""; ph.textContent="— select —";
  if (!current) ph.selected=true;
  sel.appendChild(ph);
  for (const v of opts) {
    const o = document.createElement("option");
    o.value=v; o.textContent=labels[v]||v;
    if (v===current) o.selected=true;
    sel.appendChild(o);
  }
  sel.onchange = () => {
    sel.classList.toggle("ph", !sel.value);
    onChange(sel.value);
  };
  return sel;
}

// ── NEW: PID Summary panel ─────────────────────────────────
function renderPidSummary(container) {
  const modes = app.trials[pid] || {};
  const card = document.createElement("div");
  card.className = "summary-card";

  const hdr = document.createElement("div");
  hdr.className = "summary-hdr";
  hdr.textContent = "Summary · P" + pid;
  card.appendChild(hdr);

  const body = document.createElement("div");
  body.className = "summary-body";

  const modeOrder = [["a","adaptive","Adaptive"], ["b","baseline","Baseline"]];
  let any = false;

  for (const [modeKey, cls, label] of modeOrder) {
    const trials = modes[modeKey];
    if (!trials?.length) continue;
    any = true;

    const col = document.createElement("div");
    col.className = "summary-col " + cls;

    const h3 = document.createElement("h3");
    h3.textContent = label;
    col.appendChild(h3);

    const done = trials.filter(t => t.Actual_Outcome).length;
    const pct = Math.round((done / trials.length) * 100);
    const prog = document.createElement("div");
    prog.className = "summary-progress";
    prog.innerHTML = done + " / " + trials.length + " annotated (" + pct + "%)" +
      '<div class="bar"><div class="fill" style="width:' + pct + '%"></div></div>';
    col.appendChild(prog);

    // Outcome distribution
    const outcomeCounts = {};
    let stoppedCount = 0;
    const gazeAlarmCounts = {};
    for (const t of trials) {
      if (t.Actual_Outcome)
        outcomeCounts[t.Actual_Outcome] = (outcomeCounts[t.Actual_Outcome]||0)+1;
      if (t.Stopped_at_Alarm && t.Stopped_at_Alarm !== "no")
        stoppedCount++;
      if (t.Gaze_alarm)
        gazeAlarmCounts[t.Gaze_alarm] = (gazeAlarmCounts[t.Gaze_alarm]||0)+1;
    }

    addSummaryRow(col, "Stopped at alarm", stoppedCount + " / " + trials.length);
    for (const [k,v] of Object.entries(outcomeCounts))
      addSummaryRow(col, OUT_LABELS[k]||k, v);
    if (Object.keys(gazeAlarmCounts).length) {
      const divider = document.createElement("div");
      divider.style.cssText = "margin-top:6px;padding-top:6px;border-top:1px solid #f0f0f0;";
      col.appendChild(divider);
      for (const [k,v] of Object.entries(gazeAlarmCounts))
        addSummaryRow(col, "Gaze @ " + (STOP_LABELS[k]||k), v);
    }

    body.appendChild(col);
  }

  if (!any) {
    const empty = document.createElement("div");
    empty.className = "summary-empty";
    empty.textContent = "No trials for this participant yet.";
    body.appendChild(empty);
  }

  card.appendChild(body);
  container.appendChild(card);
}
function addSummaryRow(col, k, v) {
  const row = document.createElement("div");
  row.className = "summary-row";
  const ks = document.createElement("span"); ks.className="k"; ks.textContent=k;
  const vs = document.createElement("span"); vs.className="v"; vs.textContent=v;
  row.append(ks, vs);
  col.appendChild(row);
}

// ── NEW: Overview across all participants ──────────────────
function renderOverview() {
  const el = document.getElementById("content");
  el.innerHTML = "";

  const card = document.createElement("div");
  card.className = "summary-card";
  const hdr = document.createElement("div");
  hdr.className = "summary-hdr";
  hdr.textContent = "Overview \u00b7 All Participants (" + app.pids.length + ")";
  card.appendChild(hdr);

  const body = document.createElement("div");
  body.className = "summary-body";

  const modeOrder = [["a","adaptive","Adaptive"], ["b","baseline","Baseline"]];
  for (const [modeKey, cls, label] of modeOrder) {
    const col = document.createElement("div");
    col.className = "summary-col " + cls;
    const h3 = document.createElement("h3");
    h3.textContent = label;
    col.appendChild(h3);

    let total=0, done=0, stoppedCount=0;
    const outcomeCounts = {};
    const gazeAlarmCounts = {};
    for (const p of app.pids) {
      const trials = (app.trials[p]||{})[modeKey] || [];
      for (const t of trials) {
        total++;
        if (t.Actual_Outcome) {
          done++;
          outcomeCounts[t.Actual_Outcome] = (outcomeCounts[t.Actual_Outcome]||0)+1;
        }
        if (t.Stopped_at_Alarm && t.Stopped_at_Alarm !== "no") stoppedCount++;
        if (t.Gaze_alarm) gazeAlarmCounts[t.Gaze_alarm] = (gazeAlarmCounts[t.Gaze_alarm]||0)+1;
      }
    }
    const pct = total ? Math.round((done/total)*100) : 0;
    const prog = document.createElement("div");
    prog.className = "summary-progress";
    prog.innerHTML = done + " / " + total + " annotated (" + pct + "%)" +
      '<div class="bar"><div class="fill" style="width:' + pct + '%"></div></div>';
    col.appendChild(prog);

    addSummaryRow(col, "Stopped at alarm", stoppedCount + " / " + total);
    for (const [k,v] of Object.entries(outcomeCounts))
      addSummaryRow(col, OUT_LABELS[k]||k, v);
    if (Object.keys(gazeAlarmCounts).length) {
      const divider = document.createElement("div");
      divider.style.cssText = "margin-top:6px;padding-top:6px;border-top:1px solid #f0f0f0;";
      col.appendChild(divider);
      for (const [k,v] of Object.entries(gazeAlarmCounts))
        addSummaryRow(col, "Gaze @ " + (STOP_LABELS[k]||k), v);
    }
    body.appendChild(col);
  }
  card.appendChild(body);
  el.appendChild(card);

  // Per-participant completion table
  const tblSection = document.createElement("div");
  tblSection.className = "section";
  const tblHdr = document.createElement("div");
  tblHdr.className = "sec-hdr";
  tblHdr.style.cssText = "background:#f9fafb;color:var(--text);border-bottom:1px solid var(--border);";
  tblHdr.textContent = "Per-Participant Completion";
  tblSection.appendChild(tblHdr);

  const tbl = document.createElement("table");
  tbl.innerHTML = `<thead><tr>
    <th>PID</th><th>Adaptive</th><th>Baseline</th><th>Total</th>
  </tr></thead>`;
  const tbody = document.createElement("tbody");
  for (const p of app.pids) {
    const aTrials = (app.trials[p]||{})["a"]||[];
    const bTrials = (app.trials[p]||{})["b"]||[];
    const aDone = aTrials.filter(t=>t.Actual_Outcome).length;
    const bDone = bTrials.filter(t=>t.Actual_Outcome).length;
    const totalDone = aDone+bDone, totalAll = aTrials.length+bTrials.length;
    const complete = totalAll>0 && totalDone===totalAll;
    const tr = document.createElement("tr");
    tr.style.cursor = "pointer";
    tr.title = "Click to open P" + p;
    tr.onclick = () => chooseView(p);
    tr.innerHTML = `
      <td><span class="badge" style="background:#eef2ff;color:#4338ca;">P${p}</span></td>
      <td>${aDone} / ${aTrials.length}</td>
      <td>${bDone} / ${bTrials.length}</td>
      <td><strong>${totalDone} / ${totalAll}</strong>${complete ? " \u2713" : ""}</td>
    `;
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);
  tblSection.appendChild(tbl);
  el.appendChild(tblSection);
}

// ── Trial ordering: by trial type (t1..t5,c1,c2,s1,s2) or by the actual
// session order from generate_run_order()'s run_order.csv (exact index,
// no ties -- 'unknown' only means this trial's Run code wasn't found in
// that file, e.g. missing/corrupt run_order.csv).
function orderTrials(trials) {
  if (sortMode !== "session") return trials.map(t => ({ t, unknown: false }));
  const known = trials.filter(t => t.Session_Order != null);
  if (!known.length) return trials.map(t => ({ t, unknown: false }));
  const sorted = [...trials].sort((a, b) => {
    const ao = a.Session_Order, bo = b.Session_Order;
    if (ao == null && bo == null) return 0;
    if (ao == null) return 1;   // unknown position sorts last
    if (bo == null) return -1;
    return ao - bo;
  });
  return sorted.map(t => ({ t, unknown: t.Session_Order == null }));
}

function renderSortToggle(container) {
  const modes = app.trials[pid] || {};
  const anyOrderInfo = ["a","b"].some(m => (modes[m]||[]).some(t => t.Session_Order != null));
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:12px;font-size:12px;color:var(--muted);";
  const lbl = document.createElement("span"); lbl.textContent = "Order:";
  const mkBtn = (mode, label) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = "padding:4px 10px;border-radius:5px;border:1px solid var(--border);" +
      "font-size:12px;cursor:pointer;background:" + (sortMode===mode ? "#111827" : "#fff") +
      ";color:" + (sortMode===mode ? "#fff" : "var(--text)") + ";";
    b.disabled = mode==="session" && !anyOrderInfo;
    if (b.disabled) { b.style.opacity = "0.4"; b.style.cursor = "not-allowed"; b.title = "No run_order.csv found for this participant"; }
    b.onclick = () => { sortMode = mode; renderContent(); };
    return b;
  };
  wrap.append(lbl, mkBtn("type","Trial type"), mkBtn("session","Session order (run_order.csv)"));
  if (sortMode === "session") {
    const hint = document.createElement("span");
    hint.textContent = "? = Run code not found in run_order.csv \u00b7 block order reflects counterbalancing (odd PID = Adaptive first)";
    hint.style.cssText = "color:var(--muted2);margin-left:4px;";
    wrap.appendChild(hint);
  }
  container.appendChild(wrap);
}

// ── Participant profile: Speed/Style now apply to the whole participant,
// not per trial. Mid-session changes go in a trial's Note instead.
function renderParticipantProfile(container) {
  if (!app.participants[pid]) app.participants[pid] = {};
  const prof = app.participants[pid];

  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;align-items:center;gap:10px;margin-bottom:12px;" +
    "padding:10px 14px;background:#f8fafc;border:1px solid var(--border);border-radius:8px;font-size:12px;";
  const title = document.createElement("span");
  title.textContent = "Participant profile:";
  title.style.cssText = "font-weight:600;color:var(--text);";
  wrap.appendChild(title);

  const lblSpd = document.createElement("span"); lblSpd.className="lbl"; lblSpd.textContent="Speed:";
  const spdSel = mkSel(app.options.walker_speed, SPEED_LABELS, prof.Walker_speed, "s-speed",
    val => { prof.Walker_speed = val; saveParticipant(pid); });

  const lblSty = document.createElement("span"); lblSty.className="lbl"; lblSty.textContent="Style:";
  const stySel = mkSel(app.options.walking_style, STYLE_LABELS, prof.Walking_style, "s-style",
    val => { prof.Walking_style = val; saveParticipant(pid); });

  const hint = document.createElement("span");
  hint.textContent = "(changes mid-session -> note it on the relevant trial)";
  hint.style.cssText = "color:var(--muted2);margin-left:4px;";

  wrap.append(lblSpd, spdSel, lblSty, stySel, hint);
  container.appendChild(wrap);
}

// Counterbalancing (per study design): odd PID = Adaptive first,
// even PID = Baseline first. In "Trial type" view we always show Adaptive
// above Baseline for consistency; in "Session order" view the block order
// itself should reflect which block the participant actually did first.
function blockOrder() {
  if (sortMode !== "session") return ["a", "b"];
  const n = parseInt(String(pid).replace(/\D/g, ""), 10);
  if (isNaN(n)) return ["a", "b"];
  return (n % 2 === 1) ? ["a", "b"] : ["b", "a"];
}

function renderContent() {
  const el = document.getElementById("content");
  el.innerHTML = "";
  const modes = app.trials[pid]||{};

  renderPidSummary(el);
  renderParticipantProfile(el);
  renderSortToggle(el);

  for (const mode of blockOrder()) {
    const trialsRaw = modes[mode];
    if (!trialsRaw?.length) continue;
    const ordered = orderTrials(trialsRaw);

    const section = document.createElement("div");
    section.className = "section";

    const hdr = document.createElement("div");
    hdr.className = "sec-hdr " + (mode==="a" ? "adaptive" : "baseline");
    hdr.dataset.mode = mode;
    section.appendChild(hdr);
    refreshHdr(hdr, trialsRaw, mode);

    const stats = document.createElement("div");
    stats.className = "stats-bar"; stats.dataset.mode = mode;
    section.appendChild(stats);
    refreshStats(stats, trialsRaw);

    const tbl = document.createElement("table");
    tbl.innerHTML = `<thead><tr>
      <th>#</th><th>Trial</th><th>Logged</th><th>Actual Outcome</th>
      <th>Gaze (general)</th><th>Alarm reaction</th><th>Stopped (alarm)</th>
      <th>Looked up (alarm)</th><th>Note</th><th></th>
    </tr></thead>`;
    const tbody = document.createElement("tbody");

    for (const [seq, {t, unknown}] of ordered.entries()) {

      const tr = document.createElement("tr");
      tr.className = rowCls(t);

      // Sequence number (position in the currently selected order)
      const tdSeq = document.createElement("td");
      tdSeq.style.cssText = "color:var(--muted2);font-size:11px;";
      tdSeq.textContent = (seq+1) + (unknown ? " ?" : "");
      if (unknown) tdSeq.title = "Run code not found in run_order.csv";
      tr.appendChild(tdSeq);

      // Trial badge
      const tdB = document.createElement("td");
      const bdg = document.createElement("span");
      bdg.className = "badge "+badgeCls(t.Run); bdg.textContent = t.Run;
      tdB.appendChild(bdg); tr.appendChild(tdB);

      // Logged
      const tdL = document.createElement("td");
      const pill = document.createElement("span");
      pill.className="logged-pill"; pill.textContent=t.Logged_Outcome;
      tdL.appendChild(pill); tr.appendChild(tdL);

      // Actual Outcome
      const tdO = document.createElement("td");
      const outOpts  = ["__keep__", ...app.options.outcome];
      const outLabels = {__keep__: "↩ same as logged", ...OUT_LABELS};
      let reaSel;   // Alarm reaction select, created below -- referenced here so
      let gazeaSel; // changing Outcome can live-update both (safe_turn/safe_stop default)
      const outSel = mkSel(outOpts, outLabels, t.Actual_Outcome, "s-outcome",
        val => {
          t.Actual_Outcome = val==="__keep__" ? t.Logged_Outcome : val;
          tr.className = rowCls(t);
          updateDot(tr, t);
          // Mirror the backend default: safe_turn (any mode) / safe_stop
          // (baseline only) implies no alarm reaction and no alarm-triggered
          // gaze shift -- but only fill fields still empty, never overwrite.
          const noAlarmExpected = t.Actual_Outcome === "safe_turn" ||
            (t.Actual_Outcome === "safe_stop" && mode === "b");
          if (!t.Alarm_reaction && noAlarmExpected) {
            t.Alarm_reaction = "none";
            if (reaSel) { reaSel.value = "none"; reaSel.classList.remove("ph"); }
          }
          if (!t.Gaze_alarm && noAlarmExpected) {
            t.Gaze_alarm = "no";
            if (gazeaSel) { gazeaSel.value = "no"; gazeaSel.classList.remove("ph"); }
          }
          refreshHdrByMode(mode); refreshStatsByMode(mode); updateSubtitle();
          refreshSummary();
          scheduleSave(t);
        });
      tdO.appendChild(outSel); tr.appendChild(tdO);

      // Gaze - general
      const tdGg = document.createElement("td");
      const gazegSel = mkSel(app.options.gaze_general, GAZE_G_LABELS, t.Gaze_general, "s-gaze-g",
        val => { t.Gaze_general=val; scheduleSave(t); });
      tdGg.appendChild(gazegSel); tr.appendChild(tdGg);

      // Alarm reaction
      const tdRea = document.createElement("td");
      reaSel = mkSel(app.options.alarm_reaction, REACT_LABELS, t.Alarm_reaction, "s-react",
        val => { t.Alarm_reaction=val; scheduleSave(t); });
      tdRea.appendChild(reaSel); tr.appendChild(tdRea);

      // Stopped at Alarm
      const tdS = document.createElement("td");
      const stopOpts = mode==="a" ? app.options.stopped_adaptive : app.options.stopped_baseline;
      const stopSel  = mkSel(stopOpts, STOP_LABELS, t.Stopped_at_Alarm, "s-stopped",
        val => { t.Stopped_at_Alarm=val; refreshSummary(); scheduleSave(t); });
      tdS.appendChild(stopSel); tr.appendChild(tdS);

      // Looked up (alarm) -- which alarm triggered a gaze shift (mirrors Stopped)
      const tdGa = document.createElement("td");
      const gazeAlarmOpts = mode==="a" ? app.options.gaze_alarm_adaptive : app.options.gaze_alarm_baseline;
      gazeaSel = mkSel(gazeAlarmOpts, STOP_LABELS, t.Gaze_alarm, "s-gaze-a",
        val => { t.Gaze_alarm=val; scheduleSave(t); });
      tdGa.appendChild(gazeaSel); tr.appendChild(tdGa);

      // Note
      const tdNt = document.createElement("td");
      const noteInp = document.createElement("input");
      noteInp.type="text"; noteInp.className="inp-note";
      noteInp.value = t.Note||""; noteInp.placeholder="optional note…";
      noteInp.oninput = () => { t.Note=noteInp.value; scheduleSave(t); };
      tdNt.appendChild(noteInp); tr.appendChild(tdNt);

      // Status dot
      const tdD = document.createElement("td");
      const dot = document.createElement("span");
      dot.className = "dot "+(t.Actual_Outcome ? "dot-yes":"dot-no");
      dot.textContent = t.Actual_Outcome ? "✓" : "○";
      tdD.appendChild(dot); tr.appendChild(tdD);

      tbody.appendChild(tr);
    }

    tbl.appendChild(tbody);
    section.appendChild(tbl);
    el.appendChild(section);
  }
}

function refreshSummary() {
  // Re-render just the summary card (cheap; called after edits that affect it)
  const el = document.getElementById("content");
  const old = el.querySelector(".summary-card");
  if (old) old.remove();
  const tmp = document.createElement("div");
  renderPidSummary(tmp);
  el.prepend(tmp.firstChild);
}

function updateDot(tr, t) {
  const dot = tr.querySelector(".dot");
  if (!dot) return;
  dot.className = "dot "+(t.Actual_Outcome ? "dot-yes":"dot-no");
  dot.textContent = t.Actual_Outcome ? "✓":"○";
}

function refreshHdr(el, trials, mode) {
  const done = trials.filter(t=>t.Actual_Outcome).length;
  el.textContent = (mode==="a"?"Adaptive":"Baseline") +
    "  ·  P"+pid+"  ·  "+done+"/"+trials.length+" annotated";
}
function refreshHdrByMode(mode) {
  const trials = (app.trials[pid]||{})[mode]||[];
  const el = document.querySelector(".sec-hdr[data-mode='"+mode+"']");
  if (el) refreshHdr(el, trials, mode);
}

function refreshStats(el, trials) {
  const counts={};
  for (const t of trials) if(t.Actual_Outcome)
    counts[t.Actual_Outcome]=(counts[t.Actual_Outcome]||0)+1;
  el.innerHTML="";
  const entries=Object.entries(counts);
  if (!entries.length) { el.innerHTML="<span>No annotations yet</span>"; return; }
  for (const [k,v] of entries) {
    const s=document.createElement("span");
    s.innerHTML="<strong>"+v+"</strong> "+(OUT_LABELS[k]||k);
    el.appendChild(s);
  }
}
function refreshStatsByMode(mode) {
  const trials=(app.trials[pid]||{})[mode]||[];
  const el=document.querySelector(".stats-bar[data-mode='"+mode+"']");
  if (el) refreshStats(el, trials);
}

function scheduleSave(t) {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(()=>saveTrial(t), 350);
}

async function saveTrial(t) {
  try {
    const r = await fetch("/api/save", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        pid:t.PID, mode:t.Mode, run:t.Run,
        Actual_Outcome:   t.Actual_Outcome   ||"",
        Stopped_at_Alarm: t.Stopped_at_Alarm ||"",
        Gaze_general:     t.Gaze_general     ||"",
        Gaze_alarm:       t.Gaze_alarm       ||"",
        Alarm_reaction:   t.Alarm_reaction   ||"",
        Note:             t.Note             ||"",
      }),
    });
    if (!r.ok) {
      const errBody = await r.json().catch(() => ({}));
      throw new Error(errBody.error || ("HTTP " + r.status));
    }
    toast("Saved ✓");
  } catch(e) { toast("Save failed: " + e.message, true); }
}

async function saveParticipant(p) {
  try {
    const prof = app.participants[p] || {};
    const r = await fetch("/api/save_participant", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        pid: p,
        Walker_speed:  prof.Walker_speed  || "",
        Walking_style: prof.Walking_style || "",
      }),
    });
    if (!r.ok) {
      const errBody = await r.json().catch(() => ({}));
      throw new Error(errBody.error || ("HTTP " + r.status));
    }
    toast("Saved ✓");
  } catch(e) { toast("Save failed: " + e.message, true); }
}

function toast(msg, err=false) {
  const el=document.getElementById("toast");
  el.textContent=msg; el.style.background=err?"#dc2626":"#111827";
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer=setTimeout(()=>el.classList.remove("show"),1800);
}

init();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="V2P browser annotation tool")
    ap.add_argument("--port",       type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not state.trials:
        print(f"[WARN] No trial logs found under '{LOG_DIR}/'. "
              "Make sure you're in the right directory.")

    url = f"http://127.0.0.1:{args.port}"   # explicit IPv4 — avoids IPv6/localhost issues
    print(f"\n  V2P Annotation Tool")
    print(f"  URL:  {url}")
    print(f"  Trials loaded: {len(state.trials)}  |  "
          f"Annotated: {len(state.annotations)}")
    print(f"\n  Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    # ThreadingHTTPServer (not plain HTTPServer): the plain HTTPServer handles
    # one connection at a time. Modern browsers routinely open extra idle
    # connections (preconnect, keep-alive probes, multiple tabs) and if one of
    # those sits open without sending a request, a single-threaded server
    # blocks on it forever -- every other tab/request then hangs too, even
    # though the server process itself is alive and fine. This was confirmed
    # to be the cause of the "page loads forever in the browser but a plain
    # HTTP client works instantly" symptom.
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)   # bind all interfaces
    server.daemon_threads = True   # so Ctrl+C actually exits, even with open connections
    server.allow_reuse_address = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()