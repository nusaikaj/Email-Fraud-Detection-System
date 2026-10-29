"""PhishGuard AI V3.2 Flask interface."""
from __future__ import annotations
from email import policy
from email.parser import BytesParser
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template, request
from detector import Detector, MAX_CHARS

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024
_detector = None
DB_PATH = BASE / 'scan_history.db'

def db_connection():
    connection = sqlite3.connect(DB_PATH, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection

def init_db():
    with db_connection() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY AUTOINCREMENT, scanned_at TEXT NOT NULL, label TEXT NOT NULL, spam_probability REAL NOT NULL DEFAULT 0, signal_count INTEGER NOT NULL, link_count INTEGER NOT NULL)""")
        columns = {row[1] for row in db.execute('PRAGMA table_info(scans)')}
        if 'threat_probability' not in columns:
            db.execute('ALTER TABLE scans ADD COLUMN threat_probability REAL')
        if 'model_version' not in columns:
            db.execute('ALTER TABLE scans ADD COLUMN model_version TEXT')

def record_scan(result):
    with db_connection() as db:
        db.execute('INSERT INTO scans (scanned_at, label, spam_probability, threat_probability, model_version, signal_count, link_count) VALUES (?, ?, ?, ?, ?, ?, ?)', (datetime.now(timezone.utc).isoformat(timespec='seconds'), result['label'], 0.0, result['threat_probability'], result['model_version'], len(result['signals']), len(result['links_found'])))

init_db()

def get_detector():
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector

def extract_eml(data: bytes) -> str:
    """Read text/plain from an email; don't execute or render HTML attachments."""
    message = BytesParser(policy=policy.default).parsebytes(data)
    items = [message] if not message.is_multipart() else message.walk()
    contents = []
    for part in items:
        if part.get_content_type() == 'text/plain' and part.get_content_disposition() != 'attachment':
            try:
                contents.append(part.get_content())
            except (UnicodeError, LookupError, ValueError):
                continue
    return '\n'.join(contents)

@app.get('/')
def home():
    return render_template('index.html')

@app.get('/health')
def health():
    return jsonify(status='ok')

@app.get('/api/history')
def history():
    with db_connection() as db:
        rows = db.execute("SELECT id, scanned_at, label, CASE WHEN model_version IS NULL THEN spam_probability ELSE threat_probability END AS score, COALESCE(model_version, 'legacy-sms-model') AS model_version, signal_count, link_count FROM scans ORDER BY id DESC LIMIT 50").fetchall()
    return jsonify(scans=[dict(row) for row in rows])

@app.get('/api/stats')
def stats():
    with db_connection() as db:
        row = db.execute("SELECT COUNT(*) AS total, SUM(CASE WHEN label = 'Threat-like' THEN 1 ELSE 0 END) AS threats, SUM(CASE WHEN label = 'Safe-like' THEN 1 ELSE 0 END) AS safe, SUM(CASE WHEN signal_count > 0 THEN 1 ELSE 0 END) AS flagged FROM scans").fetchone()
    return jsonify(total=row['total'], threats=row['threats'] or 0, safe=row['safe'] or 0, flagged=row['flagged'] or 0)

@app.get('/api/analytics')
def analytics():
    """Aggregated local metadata only; excludes legacy model classifications from breakdown."""
    with db_connection() as db:
        rows = db.execute("""SELECT substr(scanned_at,1,10) AS day, label, model_version,
                    signal_count, link_count FROM scans ORDER BY id DESC""").fetchall()
    modern = [dict(r) for r in rows if r['model_version'] == 'v3.2-logistic-regression']
    days = {}
    for r in modern:
        day = r['day']
        if day not in days: days[day] = {'date': day, 'safe': 0, 'threat': 0}
        if r['label'] == 'Threat-like': days[day]['threat'] += 1
        elif r['label'] == 'Safe-like': days[day]['safe'] += 1
    return jsonify(
        total_scans=len(rows),
        current_model_scans=len(modern),
        legacy_scans=len(rows)-len(modern),
        safe=sum(r['label']=='Safe-like' for r in modern),
        threat=sum(r['label']=='Threat-like' for r in modern),
        flagged=sum(r['signal_count'] > 0 for r in modern),
        linked=sum(r['link_count'] > 0 for r in modern),
        daily=sorted(days.values(),key=lambda d:d['date'])[-14:]
    )

@app.get('/api/model-metrics')
def model_metrics():
    """Load evaluated report without exposing file paths or arbitrary user inputs."""
    import json
    report = BASE / 'reports' / 'grouped' / 'final_test_results.json'
    compare = BASE / 'reports' / 'grouped' / 'v32_model_comparison.json'
    def load(path):
        try:
            with path.open(encoding='utf-8') as f: return json.load(f)
        except (OSError, ValueError): return None
    return jsonify(holdout=load(report), validation=load(compare),
                   note='Scores are for safe vs mixed threat emails, not phishing-only; external validation pending.')

@app.post('/api/analyze')
def analyze():
    if request.is_json:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error='JSON object required.'), 400
        text = payload.get('email', '')
    else:
        upload = request.files.get('file')
        if upload:
            if not (upload.filename or '').lower().endswith('.eml'):
                return jsonify(error='Only .eml email files are supported.'), 400
            contents = upload.read(256 * 1024 + 1)
            if len(contents) > 256 * 1024:
                return jsonify(error='Email file is too large.'), 413
            text = extract_eml(contents)
            if not text.strip():
                return jsonify(error='This .eml file has no readable plain-text body.'), 400
        else:
            text = request.form.get('email', '')
    try:
        result = get_detector().analyze(text)
        record_scan(result)
        return jsonify(result)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

@app.errorhandler(413)
def too_large(_):
    return jsonify(error='File or request too large (max 256 KB).'), 413

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
