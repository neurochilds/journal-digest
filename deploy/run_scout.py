"""Run the existing tracker on Umbrel, using private credentials and persistent state."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import smtplib
import ssl
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

KEYS = ('OPENALEX_API_KEY', 'OPENAI_API_KEY', 'GMAIL_ADDRESS',
        'GMAIL_APP_PASSWORD', 'RECIPIENT_EMAIL')
STATE_FILES = ('seen_papers.json', 'first_observed.json', 'pending_papers.json', 'digest_log.csv')


class RedactedStream:
    def __init__(self, stream, values):
        self.stream, self.values, self.pending = stream, sorted(values, key=len, reverse=True), ''

    def write(self, text):
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            for value in self.values:
                line = line.replace(value, '[redacted]')
            self.stream.write(line+'\n')
        return len(text)

    def flush(self):
        # Keep incomplete lines buffered so a split credential cannot escape.
        self.stream.flush()

    def isatty(self):
        return False


def atomic(path, value):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name)
    try:
        with os.fdopen(descriptor, 'w') as output:
            json.dump(value, output)
            output.flush(); os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def guarded_delivery(marker, sender):
    def send(subject, html, text):
        if marker.exists():
            raise RuntimeError('Prior delivery requires reconciliation')
        proof = {'state': 'uncertain', 'started_at': datetime.now(timezone.utc).isoformat(),
                 'digest_sha256': hashlib.sha256((subject+'\n'+html+'\n'+text).encode()).hexdigest()}
        atomic(marker, proof)
        sender(subject, html, text)
        atomic(marker, proof | {'state': 'accepted'})
    return send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=Path('/var/lib/paper-scout'))
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--fetch-only', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    credential = args.credentials or Path(os.environ['CREDENTIALS_DIRECTORY'])/'credentials.json'
    values = json.loads(credential.read_text())
    required = set(KEYS)
    backend = os.environ.get('PAPER_SCOUT_AI_BACKEND', 'api')
    if backend == 'codex':
        required.remove('OPENAI_API_KEY')
    if not required <= set(values) <= set(KEYS) or any(not isinstance(v, str) or not v for v in values.values()):
        raise ValueError('Digest credentials are incomplete')
    sys.stdout = RedactedStream(sys.stdout, values.values())
    sys.stderr = RedactedStream(sys.stderr, values.values())
    os.environ.update({k: v for k, v in values.items() if backend != 'codex' or k != 'OPENAI_API_KEY'})
    if backend == 'codex':
        for key in ('OPENAI_API_KEY', 'CODEX_API_KEY'):
            os.environ.pop(key, None)
    os.environ['PAPER_SCOUT_STATE_DIR'] = str(args.state.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paper_tracker as tracker
    args.state.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (args.state/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.probe:
            from openalex_client import get_json
            response = get_json('https://api.openalex.org/works', params={'per-page': 1})
            if not isinstance(response.get('results'), list):
                raise ValueError('OpenAlex response invalid')
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=15,
                                  context=ssl.create_default_context()) as smtp:
                smtp.login(values['GMAIL_ADDRESS'], values['GMAIL_APP_PASSWORD'])
            print(json.dumps({'openalex_authenticated': True, 'smtp_authenticated': True,
                              'email_sent': False}))
            return 0
        marker = args.state/'delivery-guard.json'
        if marker.exists() and not (args.preview or args.fetch_only):
            print('Prior delivery requires reconciliation; no repeat email will be attempted.')
            return 3
        preview = args.preview or args.fetch_only
        before = {name: hashlib.sha256((args.state/name).read_bytes()).hexdigest()
                  for name in STATE_FILES if (args.state/name).exists()}
        if not preview:
            backups = args.state/'backups'; backups.mkdir(mode=0o700, exist_ok=True)
            snapshot = backups/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            snapshot.mkdir(mode=0o700)
            for name in STATE_FILES:
                if (args.state/name).exists(): shutil.copy2(args.state/name, snapshot/name)
            tracker.send_email = guarded_delivery(marker, tracker.send_email)
        result = tracker.main(historical=True, dry_run=preview, fetch_only=args.fetch_only,
                              days_override=1 if args.preview else None,
                              max_llm_candidates=1 if args.preview else None,
                              preview_file=args.state/'preview.html' if preview else None)
        if result == 0 and not preview:
            marker.unlink(missing_ok=True)
        if preview:
            after = {name: hashlib.sha256((args.state/name).read_bytes()).hexdigest()
                     for name in STATE_FILES if (args.state/name).exists()}
            if before != after: raise RuntimeError('Preview changed delivery history')
        atomic(args.state/'last-run.json', {'checked_at': datetime.now(timezone.utc).isoformat(),
               'exit_code': result, 'preview': preview, 'delivery_requires_reconciliation': marker.exists()})
        return result


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Credential/provider exception strings can contain sensitive values.
        print('Paper scout failed: '+type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
