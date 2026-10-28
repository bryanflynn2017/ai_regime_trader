"""
Broker layer — Alpaca client, order executor, position tracker.

Phase 7: AlpacaClientWrapper, OrderExecutor, PositionTracker
Phase 11: BrokerAdapter ABC, AlpacaBrokerAdapter, SimulatedBrokerAdapter

Usage:
    from broker import BrokerAdapter, AlpacaBrokerAdapter, SimulatedBrokerAdapter
"""
from broker.base import BrokerAdapter, BrokerError
from broker.alpaca_adapter import AlpacaBrokerAdapter
from broker.simulated_adapter import SimulatedBrokerAdapter

__all__ = [
    "BrokerAdapter",
    "BrokerError",
    "AlpacaBrokerAdapter",
    "SimulatedBrokerAdapter",
]
