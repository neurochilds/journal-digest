import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from deploy.encrypt_credentials import CONTEXT, KEYS, encrypt
from deploy.run_scout import guarded_delivery, RedactedStream
from deploy import run_scout
from codex_scorer import CodexScreeningError


class MigrationTests(unittest.TestCase):
    def test_native_scoring_without_private_profile_stops_before_tracker(self):
        import io, os, sys
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)/'state'
            credentials = Path(root)/'credentials.json'
            credentials.write_text(json.dumps({name: 'dummy-'+name for name in KEYS}))
            output = io.StringIO()
            with patch.dict(os.environ, {'PAPER_SCOUT_AI_BACKEND': 'codex',
                                         'PAPER_SCOUT_PROFILE_FILE': str(state/'missing.md')}), \
                 patch.object(sys, 'argv', ['scout', '--state', str(state), '--credentials', str(credentials)]), \
                 patch.object(sys, 'stdout', output), patch.object(sys, 'stderr', output), \
                 patch('paper_tracker.main') as tracker:
                with self.assertRaisesRegex(RuntimeError, 'Private research profile'):
                    run_scout.main()
                tracker.assert_not_called()

    def test_failed_run_replaces_stale_success_without_erasing_delivery_guard(self):
        import io, os, sys
        for uncertain in (False, True):
            with self.subTest(uncertain=uncertain), tempfile.TemporaryDirectory() as root:
                state = Path(root) / 'state'; state.mkdir()
                (state/'research-profile.md').write_text('Private fixture research context.')
                credentials = Path(root) / 'credentials.json'
                credentials.write_text(json.dumps({name: 'dummy-'+name for name in KEYS}))
                for name in run_scout.STATE_FILES: (state/name).write_text('original-history')
                (state/'last-run.json').write_text('{"exit_code": 0}')
                def fail(**kwargs):
                    if uncertain:
                        (state/'delivery-guard.json').write_text('{"state": "uncertain"}')
                        raise TimeoutError('private-test-secret')
                    raise CodexScreeningError('authentication')
                output = io.StringIO()
                with patch.dict(os.environ, {'PAPER_SCOUT_AI_BACKEND': 'codex'}), \
                     patch.object(sys, 'argv', ['scout', '--state', str(state), '--credentials', str(credentials)]), \
                     patch.object(sys, 'stdout', output), patch.object(sys, 'stderr', output), \
                     patch('paper_tracker.main', side_effect=fail):
                    with self.assertRaises((CodexScreeningError, TimeoutError)): run_scout.main()
                proof = json.loads((state/'last-run.json').read_text())
                self.assertEqual(proof['exit_code'], 1)
                self.assertEqual(proof['failure_code'], 'runtime_error' if uncertain else 'authentication')
                self.assertEqual(proof['delivery_requires_reconciliation'], uncertain)
                self.assertEqual((state/'delivery-guard.json').exists(), uncertain)
                self.assertNotIn('private-test-secret', output.getvalue())
                self.assertNotIn('private-test-secret', json.dumps(proof))
                for name in run_scout.STATE_FILES:
                    self.assertEqual((state/name).read_text(), 'original-history')

    def test_credentials_are_redacted_across_stream_writes(self):
        import io
        target = io.StringIO()
        stream = RedactedStream(target, ['dummy-secret'])
        stream.write('provider error: dummy-'); stream.flush()
        self.assertEqual(target.getvalue(), '')
        stream.write('secret\n')
        self.assertEqual(target.getvalue(), 'provider error: [redacted]\n')
    def test_only_umbrel_private_key_decrypts_and_tamper_fails(self):
        import base64
        private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        public = private.public_key().public_bytes(serialization.Encoding.PEM,
                                                  serialization.PublicFormat.SubjectPublicKeyInfo)
        values = {name: 'dummy-'+name for name in KEYS}
        package = encrypt(values, public)
        for value in values.values(): self.assertNotIn(value, json.dumps(package))
        key = private.decrypt(base64.b64decode(package['wrapped_key']),
                              padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                           algorithm=hashes.SHA256(), label=CONTEXT))
        nonce, body = base64.b64decode(package['nonce']), base64.b64decode(package['ciphertext'])
        self.assertEqual(json.loads(AESGCM(key).decrypt(nonce, body, CONTEXT)), values)
        with self.assertRaises(InvalidTag): AESGCM(key).decrypt(nonce, body[:-1]+bytes([body[-1]^1]), CONTEXT)
        other = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        with self.assertRaises(ValueError):
            other.decrypt(base64.b64decode(package['wrapped_key']),
                          padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=CONTEXT))
    def test_incomplete_credentials_and_small_keys_rejected(self):
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public = private.public_key().public_bytes(serialization.Encoding.PEM,
                                                  serialization.PublicFormat.SubjectPublicKeyInfo)
        with self.assertRaises(ValueError): encrypt({}, public)
        with self.assertRaises(ValueError): encrypt({name:'dummy' for name in KEYS}, public)
    def test_uncertain_send_cannot_be_replayed(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root)/'guard.json'
            def ambiguous(*args):
                self.assertEqual(json.loads(marker.read_text())['state'], 'uncertain')
                raise TimeoutError('unknown acceptance')
            sender = Mock(side_effect=ambiguous)
            guarded = guarded_delivery(marker, sender)
            with self.assertRaises(TimeoutError): guarded('s','h','t')
            with self.assertRaises(RuntimeError): guarded('s','h','t')
            self.assertEqual(sender.call_count, 1)
    def test_accepted_send_recorded_before_tracker_state_commit(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root)/'guard.json'
            sender = Mock()
            guarded_delivery(marker,sender)('s','h','t')
            self.assertEqual(json.loads(marker.read_text())['state'], 'accepted')
            with self.assertRaises(RuntimeError): guarded_delivery(marker,sender)('s','h','t')
            self.assertEqual(sender.call_count, 1)
    def test_external_state_directory_preserves_source_default(self):
        import subprocess, sys, os
        with tempfile.TemporaryDirectory() as root:
            code = 'import paper_tracker as t; print(str(t.SEEN_PAPERS_FILE)); print(str(t.DIGEST_LOG_FILE))'
            result = subprocess.run([sys.executable,'-c',code], check=True, capture_output=True,text=True,
                                    cwd=Path(__file__).resolve().parents[1],
                                    env=os.environ|{'PAPER_SCOUT_STATE_DIR':root})
            self.assertEqual(result.stdout.splitlines(), [str(Path(root)/'seen_papers.json'), str(Path(root)/'digest_log.csv')])


if __name__ == '__main__': unittest.main()
