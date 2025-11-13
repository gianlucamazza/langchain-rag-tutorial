# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.2.x   | :white_check_mark: |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :x:                |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it by:

1. **Email**: <security@yourdomain.com> (replace with actual email)
2. **GitHub Security Advisories**: Use the "Security" tab in this repository

**Please do NOT**:

- Open a public issue
- Disclose the vulnerability publicly before it's fixed

## What to Include

When reporting a vulnerability, please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Time

- **Initial response**: Within 48 hours
- **Fix timeline**: Critical issues within 7 days, others within 30 days
- **Disclosure**: Coordinated disclosure after fix is released

## Security Best Practices

### API Keys

- ✅ Always use environment variables (`.env` file)
- ✅ Never commit `.env` to git
- ✅ Rotate keys regularly
- ✅ Use key management services in production (AWS Secrets Manager, etc.)

### Docker Security

- ✅ Run as non-root user (already configured)
- ✅ Use multi-stage builds (already configured)
- ✅ Scan images for vulnerabilities
- ✅ Keep base images updated

### Production Deployment

- ✅ Enable HTTPS/TLS
- ✅ Implement rate limiting
- ✅ Use CORS restrictions
- ✅ Enable authentication/authorization
- ✅ Monitor for suspicious activity

## Known Issues

None at this time.

## Security Updates

Security updates will be announced via:

- GitHub Security Advisories
- Release notes in CHANGELOG.md
- Git tags

---

Last updated: 2025-11-13
