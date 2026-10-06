# Umbrel paper scout

The native systemd service runs the existing tracker on Umbrel's agent-worker
as a separate unprivileged `paperscout` user. It uses a root-owned immutable
source release, an isolated Python environment and private persistent state
outside source. There is no public web endpoint or GitHub Actions runner.
The service cannot read capture/HomeCast credentials and has a 40-minute
deadline, 512MiB memory cap and half-core CPU quota. Existing OpenClaw/apps
are not restarted.

Monday/Thursday09:45UTC remains the twice-weekly schedule. During cutover,
remove the GitHub cron after live worker checks pass, reimport the latest four
history/queue files byte-for-byte, ensure no hosted digest is running, then
mark the worker ready and activate its timer. Seed the persistent timer stamp
at cutover time to prevent replaying Monday's already-delivered hosted run.
Do not enable two production schedules. Manual GitHub runs remain available
for read-only previews; delivery moves to Umbrel.

An owner/main-only temporary workflow encrypts the five existing digest secrets
with a fresh AES256-GCM key wrapped to Umbrel's RSA3072 public key using OAEP
SHA256. Only the encrypted artifact passes through the Mac; the private key
and decrypted credentials remain root-owned0600 on Umbrel. No secret values
are logged. Remove the provisioning workflow/public key/artifact/private key
after the migration is complete. `vocapp.reminder@gmail.com` capture-inbox
setup remains separate: these credentials are for the existing digest only.

`run_scout.py --probe` validates authenticated OpenAlex reads and SMTP login
without sending email. `--preview` scores at most one candidate with a one-day
search and generates a preview without changing sent/queue history. The
production service retains original scoring rules/model/recipient/window.

All four state files are copied into a private snapshot before each real run.
No backup pruning is enabled. A file lock prevents concurrent worker runs.
The wrapper durably records uncertainty before SMTP and records acceptance
after return. The marker clears only when the tracker's history writes finish
successfully. If a process dies or delivery is ambiguous, future production
runs stop for reconciliation rather than automatically resending. Preserve
the marker and pending state; don't delete it without checking delivery.

Successful previews/authentication are not proof of the next scheduled digest
or independent inbox receipt. Record live checks, hashes, first scheduled run
and remaining limitations in the private Umbrel operations progress log.
