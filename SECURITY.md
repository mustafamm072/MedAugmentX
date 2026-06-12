# Security Policy

MedAugmentX is a local Python library for medical-image augmentation. It does
not require network access at runtime and does not transmit image data.

## Supported versions

Security fixes are expected on the latest minor release line until `1.0`.
Older pre-`1.0` releases may receive fixes when practical, but users should
upgrade to the latest release for security and compatibility updates.

## Reporting a vulnerability

Please do not open a public issue for a suspected security vulnerability.
Report it privately to the project maintainers through the repository security
advisory flow or by contacting the maintainers listed on the package/repository.

Include:

- affected MedAugmentX version,
- affected optional extras or framework integrations,
- a minimal reproduction,
- impact assessment,
- whether patient data or private artifacts are involved.

We aim to acknowledge reports promptly, investigate reproducible issues, and
publish fixes or mitigations with clear release notes.

## Data privacy expectations

MedAugmentX may preserve metadata supplied by callers, including DICOM-derived
fields. Users are responsible for de-identifying protected health information,
controlling access to local datasets, and reviewing metadata before sharing
examples, bug reports, serialized configs, or generated artifacts.

Avoid posting patient identifiers, full DICOM headers, private scanner
protocols, signed URLs, access tokens, or customer data in public issues.

## Dependency posture

The core package depends only on NumPy and SciPy. Optional extras add their own
dependency trees:

- `io`: `pydicom`, `nibabel`
- `yaml`: `pyyaml`
- `torch`: `torch`
- `monai`: `monai`
- `torchio`: `torchio`

Commercial users should pin dependency versions, run vulnerability scanning in
their own environment, and review optional dependency licenses before
production use.

## Clinical safety boundary

MedAugmentX is not a diagnostic device or clinical decision-support system.
Security, privacy, clinical validation, and regulatory obligations for products
built with MedAugmentX remain the responsibility of the adopting organization.
