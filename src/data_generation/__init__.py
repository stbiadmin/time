"""
Data Generation Module
======================

Synthetic data generators for demonstration purposes:
- rma_generator: RMA shipping/freight data with temporal patterns
- network_events_generator: Network log events with cluster patterns
"""

from .rma_generator import generate_rma_data
from .network_events_generator import generate_network_events

__all__ = ["generate_rma_data", "generate_network_events"]
