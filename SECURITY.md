# Security Policy

smf-bench talks to caller-supplied OpenAI-compatible endpoints. Treat
`--base-url` as untrusted: do not point it at metadata IPs or third-party
hosts with credentials in the environment unless you intend that.

No secrets belong in `models/` YAML or `results/`.

Report issues to `dev@smfworks.com`.
