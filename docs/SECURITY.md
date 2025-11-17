# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| 0.9.x   | :white_check_mark: |
| 0.8.x   | :x:                |
| < 0.8   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do Not Create a Public Issue

Please **do not** create a public GitHub issue for security vulnerabilities. This could put users at risk.

### 2. Email Us Privately

Send an email to: **arat252539@gmail.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)
- Your contact information

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity (see below)

### 4. Severity Levels

We classify vulnerabilities using the following severity levels:

#### Critical (Fix within 24-48 hours)
- Remote code execution
- Authentication bypass
- Data breach potential
- Privilege escalation

#### High (Fix within 1 week)
- Denial of service
- Information disclosure
- Significant security weakness

#### Medium (Fix within 2-4 weeks)
- Minor information disclosure
- Security misconfiguration
- Moderate impact vulnerabilities

#### Low (Fix in next release)
- Security improvements
- Best practice violations
- Low impact issues

## Security Best Practices

### For Users

1. **Keep Updated**: Always use the latest version
   ```bash
   pip install --upgrade resnet-bk
   ```

2. **Verify Checksums**: Verify package integrity
   ```bash
   pip hash resnet-bk
   ```

3. **Use Virtual Environments**: Isolate dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install resnet-bk
   ```

4. **Review Dependencies**: Check for vulnerable dependencies
   ```bash
   pip install safety
   safety check
   ```

5. **Secure Checkpoints**: Don't load untrusted checkpoints
   ```python
   # Only load checkpoints from trusted sources
   checkpoint = torch.load('checkpoint.pt', map_location='cpu')
   ```

6. **Validate Inputs**: Always validate user inputs
   ```python
   # Validate sequence length
   assert 0 < n_seq <= 1_000_000, "Invalid sequence length"
   ```

### For Developers

1. **Code Review**: All code changes require review
2. **Dependency Scanning**: Automated dependency scanning in CI
3. **Static Analysis**: Use bandit for security scanning
4. **Input Validation**: Validate all inputs
5. **Secure Defaults**: Use secure defaults in configuration
6. **Least Privilege**: Run with minimum required permissions
7. **Secrets Management**: Never commit secrets to repository

## Known Security Considerations

### 1. Checkpoint Loading

**Risk**: Loading untrusted checkpoints can execute arbitrary code

**Mitigation**:
```python
# Use weights_only=True (PyTorch 2.0+)
checkpoint = torch.load('checkpoint.pt', weights_only=True)

# Or verify checkpoints
from src.utils.checkpoint_manager import verify_checkpoint
if verify_checkpoint('checkpoint.pt'):
    checkpoint = torch.load('checkpoint.pt')
```

### 2. Pickle Deserialization

**Risk**: Pickle can execute arbitrary code during deserialization

**Mitigation**:
- Only load data from trusted sources
- Use safer formats (JSON, HDF5) when possible
- Implement checkpoint verification

### 3. Model Inputs

**Risk**: Malicious inputs could cause crashes or unexpected behavior

**Mitigation**:
```python
# Validate inputs
def validate_input(input_ids, max_length=1_000_000):
    assert input_ids.dim() == 2, "Input must be 2D"
    assert input_ids.size(1) <= max_length, "Sequence too long"
    assert (input_ids >= 0).all(), "Invalid token IDs"
    return input_ids
```

### 4. Resource Exhaustion

**Risk**: Large inputs could exhaust memory or compute

**Mitigation**:
```python
# Implement resource limits
from src.models.memory_optimization import MemoryLimiter

limiter = MemoryLimiter(max_memory_gb=16)
with limiter:
    output = model(input_ids)
```

### 5. Dependency Vulnerabilities

**Risk**: Vulnerable dependencies could introduce security issues

**Mitigation**:
- Regular dependency updates
- Automated vulnerability scanning
- Pin dependency versions

## Security Updates

Security updates are released as:
- **Patch releases** (X.Y.Z+1) for supported versions
- **Security advisories** on GitHub
- **Email notifications** to security mailing list

Subscribe to security updates:
- GitHub: Watch repository → Custom → Security alerts
- Email: arat252539@gmail.com

## Disclosure Policy

We follow **coordinated disclosure**:

1. **Private Disclosure**: Report sent to arat252539@gmail.com
2. **Acknowledgment**: We acknowledge receipt within 48 hours
3. **Investigation**: We investigate and develop a fix
4. **Coordination**: We coordinate disclosure timeline with reporter
5. **Public Disclosure**: We publish security advisory after fix is released
6. **Credit**: We credit the reporter (unless they prefer anonymity)

### Typical Timeline

- **Day 0**: Vulnerability reported
- **Day 1-2**: Acknowledgment and initial assessment
- **Day 3-7**: Investigation and fix development
- **Day 7-14**: Testing and validation
- **Day 14-21**: Coordinated disclosure and release
- **Day 21+**: Public disclosure

## Bug Bounty Program

We currently do not have a formal bug bounty program, but we:
- Acknowledge security researchers in our SECURITY.md
- Provide credit in release notes
- May offer rewards for critical vulnerabilities (case-by-case basis)

## Security Hall of Fame

We thank the following researchers for responsibly disclosing security issues:

<!-- Add researchers here -->
- *No vulnerabilities reported yet*

## Contact

- **Security Issues**: arat252539@gmail.com
- **Security Mailing List**: arat252539@gmail.com
- **PGP Key**: https://resnet-bk.org/security.asc

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [PyTorch Security](https://pytorch.org/docs/stable/notes/security.html)
- [Python Security](https://python.org/dev/security/)

## Compliance

ResNet-BK follows:
- [OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CIS Controls](https://www.cisecurity.org/controls/)

---

Last updated: 2025-01-15
