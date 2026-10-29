"""V3.2 grouped-split email-threat classifier with rule-based warning signals."""
from __future__ import annotations
import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit
import joblib

BASE = Path(__file__).resolve().parent
MAX_CHARS = 20000
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.I)
SIGNALS = [
    ('Credential request', re.compile(r'\b(password|passcode|otp|one.time code|verify your account|login details|credentials)\b', re.I)),
    ('Urgency or pressure', re.compile(r'\b(urgent|immediately|within 24 hours|account (?:will be )?(?:suspended|blocked)|act now|final warning)\b', re.I)),
    ('Payment or financial request', re.compile(r'\b(bank account|wire transfer|gift card|payment details|credit card|transfer funds)\b', re.I)),
]

class Detector:
    def __init__(self, model_path=None):
        # Load only locally trained, trusted joblib artifacts.
        path = Path(model_path) if model_path else BASE / 'models_v32_grouped/logistic_regression.joblib'
        if not path.is_file():
            raise FileNotFoundError(f'V3.2 model missing: {path}. Run grouped training first.')
        self.model = joblib.load(path)
        if list(self.model.classes_) != [0, 1]:
            raise ValueError('Unexpected model labels; expected 0=safe, 1=threat.')

    def analyze(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Please enter email content.')
        text = text.strip()
        if len(text) > MAX_CHARS:
            raise ValueError(f'Message exceeds {MAX_CHARS:,} characters.')
        probabilities = self.model.predict_proba([text])[0]
        threat_probability = round(float(probabilities[1]), 4)
        signals = [name for name, pattern in SIGNALS if pattern.search(text)]
        urls = []
        for found in URL_PATTERN.findall(text)[:20]:
            url = found.rstrip('.,;!?)')
            hostname = urlsplit(url).hostname
            if not hostname:
                continue
            urls.append(hostname[:200])
            if urlsplit(url).scheme.lower() == 'http':
                signals.append('Unencrypted HTTP link')
            try:
                ipaddress.ip_address(hostname)
                signals.append('Link uses IP address instead of domain')
            except ValueError:
                pass
            if hostname.lower().startswith('xn--') or '.xn--' in hostname.lower():
                signals.append('Internationalized domain requires verification')
        return {
            'label': 'Threat-like' if threat_probability >= 0.5 else 'Safe-like',
            'threat_probability': threat_probability,
            'model_version': 'v3.2-logistic-regression',
            'signals': list(dict.fromkeys(signals)),
            'links_found': urls,
            'disclaimer': 'Model score is a threat-class probability estimate, not a calibrated real-world phishing risk. Heuristic indicators do not prove an email is malicious.'
        }
