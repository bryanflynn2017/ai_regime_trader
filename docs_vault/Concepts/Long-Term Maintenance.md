# Long-Term Maintenance

Long-term maintenance keeps the app durable as dependencies, data providers, and workflows evolve.

## Current Status
Permanent project practice.

## Plan
- Pin dependencies where appropriate.
- Track Python version.
- Prefer Python 3.11+ or newer stable supported versions.
- Keep tests passing before upgrades.
- Use small dependency upgrades.
- Maintain changelog and upgrade notes in Obsidian.
- Avoid vendor lock-in.
- Keep [[Data Providers|provider]] and [[Broker Adapters|broker]] interfaces modular.
- Keep secrets in `.env` only.
- Never commit `.env`.

## Role
Protects [[Version 1.0]] reliability and keeps the [[Long-Term AI Direction]] possible.

## Related Notes
- [[Market Data Providers]]
- [[Broker Adapters]]
- [[Safety Rules]]
- [[Experiment Tracking]]
