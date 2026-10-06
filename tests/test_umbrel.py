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
from deploy.run_scout import guarded_delivery


class MigrationTests(unittest.TestCase):
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
