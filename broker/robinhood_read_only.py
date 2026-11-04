"""NON-CONNECTED Robinhood foundation. No MCP session or HTTP client exists.

Not a BrokerAdapter: that interface authorizes paper mutations. Official MCP
schema/authentication/least-privilege mapping requires a separate reviewed task.
"""
from broker.read_only import ReadCode, ReadOnlyError


class RobinhoodReadOnly:
    @property
    def is_connected(self):
        return False

    @property
    def capabilities(self):
        return frozenset()

    def snapshot(self):
        raise ReadOnlyError(ReadCode.DISCONNECTED)

    def order_status(self, order_id):
        raise ReadOnlyError(ReadCode.DISCONNECTED)

    def require_capability(self, capability):
        raise ReadOnlyError(ReadCode.UNSUPPORTED_CAPABILITY)

    def log_safe_info(self):
        return {"provider": "robinhood", "connected": False,
                "contract_version": "read-only-v1", "capabilities": []}
