# Discord Alerts

Discord alerts are an optional notification integration using webhooks.

## Current Status
Implemented for [[Setup Alerts]] and [[Trade Monitor]] alerts.

## Role
Send [[Setup Alerts]] and [[Trade Monitor]] alerts outside the terminal while keeping the workflow [[Human-in-the-Loop Workflow|human-in-the-loop]].

## Rules
- Webhook-based.
- Disabled unless `DISCORD_WEBHOOK_URL` is configured in local `.env`.
- Can be disabled per command with `--no-discord-alerts`.
- No webhook secrets in logs, dashboard, or terminal output.
- No order execution.
- Repeated/suppressed alerts are not sent.

## Supported Alerts
- New ranked setup alerts from `setup-scan-loop`.
- Important trade monitor alerts from `trade-monitor --loop`.

## Related Notes
- [[Safety Rules]]
- [[Setup Alerts]]
- [[Trade Monitor]]
- [[Version 1.0]]
