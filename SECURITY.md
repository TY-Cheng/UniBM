# Security policy

## Supported code

UniBM is under active development. Security fixes target the latest `main` branch;
older commits do not have a separate backport commitment. Include the affected commit
or package version in a report.

## Reporting a vulnerability

Email [Tuoyuan Cheng](mailto:tuoyuan.cheng@nus.edu.sg) privately with the subject
`UniBM security report`. Do not open a public issue or pull request for an undisclosed
vulnerability.

Please include:

- The affected commit/version and relevant dependency versions.
- A description of the vulnerability, its impact, and the conditions needed to trigger it.
- Minimal reproduction steps or a proof of concept using synthetic or redacted data.

Do not include credentials, personal records, or data you are not authorized to share.
Please coordinate public disclosure with the maintainer while a report is assessed and
a fix is considered. Response and remediation times depend on maintainer availability;
there is no guaranteed service timeline.

Ordinary numerical bugs, statistical-method questions, and documentation corrections
can use [public issues](https://github.com/TY-Cheng/UniBM/issues), provided they do not
expose a vulnerability or sensitive information.
