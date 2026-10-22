# Security policy

## Supported versions

Only the latest commit on `main` receives fixes. There are no maintained release branches.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's advisory form:
https://github.com/hellpuffyt/model-bench/security/advisories/new

Do not open a public issue for a suspected vulnerability. Include the version or commit,
a minimal reproduction, and the impact you observed. You should receive an acknowledgement
within a few days; fixes are released as soon as they are verified.

## Scope

model-bench sends prompts to an inference endpoint you name (default: a local Ollama on 127.0.0.1) and reads an optional API key only from the environment variable you name with `--api-key-env`; the key is never written to recordings or reports. Relevant reports include: a key or prompt content leaking into output files, requests sent anywhere other than the configured endpoint, or a crafted recording that executes code.
