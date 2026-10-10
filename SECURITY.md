# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.4.x   | Yes       |
| 1.3.x   | Yes       |
| 1.2.x   | Yes       |
| < 1.2   | No        |

## Reporting a Vulnerability

Report security issues privately through
[GitHub Security Advisories](https://github.com/gianlucamazza/langchain-rag-tutorial/security/advisories/new)
(the "Security" tab of this repository).

Please do **not** open a public issue or disclose the vulnerability publicly before a fix is
released.

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix, if any

## Response Time

There is **no committed SLA**. Responses are best-effort and depend on maintainer availability.

## Security Best Practices

### API Keys

- Keep keys in environment variables or a `.env` file; never commit `.env`.
- Rotate keys regularly and immediately after any exposure.
- Use a secrets manager if you take a template beyond a local demo (for example AWS Secrets Manager or Azure Key Vault).

### Docker

- The image runs as a non-root user and uses a multi-stage build.
- Scan images for vulnerabilities and keep the base image (`python:3.12-slim`) updated.

### If you deploy a template

These templates are teaching apps. If you take one further than a local demo:

- Enable HTTPS/TLS.
- Add rate limiting, authentication and authorization.
- Restrict CORS origins.
- Monitor for suspicious activity.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for starter notes.

## Known Issues

None at this time.

## Security Updates

Security updates are announced through GitHub Security Advisories, the
[changelog](docs/CHANGELOG.md) and git tags.

---

Last updated: 2026-10-10
