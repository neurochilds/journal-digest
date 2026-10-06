"""One-off migration: encrypt configured GitHub secrets for Umbrel's public key."""
import base64
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEYS = ('OPENALEX_API_KEY', 'OPENAI_API_KEY', 'GMAIL_ADDRESS',
        'GMAIL_APP_PASSWORD', 'RECIPIENT_EMAIL')
CONTEXT = b'neurochilds/journal-digest:umbrel-paper-scout:v1'


def encrypt(values, public_pem):
    if set(values) != set(KEYS) or any(not isinstance(v, str) or not v for v in values.values()):
        raise ValueError('All existing digest credentials must be configured')
    public = serialization.load_pem_public_key(public_pem)
    if not isinstance(public, rsa.RSAPublicKey) or public.key_size < 3072:
        raise ValueError('A dedicated RSA3072+ recipient key is required')
    key, nonce = AESGCM.generate_key(bit_length=256), os.urandom(12)
    body = AESGCM(key).encrypt(nonce, json.dumps(values, sort_keys=True).encode(), CONTEXT)
    wrapped = public.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                             algorithm=hashes.SHA256(), label=CONTEXT))
    return {'version': 1, 'public_key_sha256': hashlib.sha256(public_pem).hexdigest(),
            **{name: base64.b64encode(value).decode('ascii') for name, value in
               [('wrapped_key', wrapped), ('nonce', nonce), ('ciphertext', body)]}}


if __name__ == '__main__':
    os.umask(0o077)
    public = Path('deploy/umbrel-migration-public.pem').read_bytes()
    payload = encrypt({name: os.environ.get(name, '') for name in KEYS}, public)
    with Path('umbrel-credentials.enc.json').open('x') as output:
        json.dump(payload, output)
    print('Encrypted five existing credential fields for the committed Umbrel public key.')
