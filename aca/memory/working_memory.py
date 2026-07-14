"""
ACA Working Memory
==================

Maintains the transient operational context for the current cognitive
cycle. Stores active goals, recent observations, in-flight hypotheses,
and pending tasks. Entries are automatically evicted when the capacity
limit is reached (FIFO eviction).

Working Memory is the scratchpad of the cognitive loop — fast reads,
fast writes, bounded size.

Design Decisions:
    - Namespace-partitioned storage (goals, observations, hypotheses, etc.).
    - Capacity-bounded with oldest-first eviction per namespace.
    - Thread-safe via reentrant lock.
    - No persistence — contents are lost on restart.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from aca.config import MemoryConfig
from aca.logging_config import get_logger

logger = get_logger("memory.working")


class WorkingMemory:
    """
    Bounded, namespace-partitioned transient memory store.

    Args:
        config: Memory configuration (controls total capacity).

    Example::

        wm = WorkingMemory(MemoryConfig(working_memory_capacity=200))
        wm.store("goals", "g1", {"metric": "soil_moisture", "value": 0.4})
        entry = wm.retrieve("goals", "g1")
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._capacity = config.working_memory_capacity
        self._namespaces: Dict[str, OrderedDict[str, Any]] = {}
        self._lock = threading.RLock()
        logger.info("WorkingMemory initialised (capacity=%d)", self._capacity)

    # ── Write ─────────────────────────────────────────────────────────

    def store(self, namespace: str, key: str, value: Any) -> None:
        """
        Store a value under a namespace/key pair.

        If total entries across all namespaces exceed capacity, the
        oldest entry (across all namespaces) is evicted.

        Args:
            namespace: Logical partition (e.g. ``goals``, ``observations``).
            key: Unique identifier within the namespace.
            value: The data to store.
        """
        with self._lock:
            ns = self._namespaces.setdefault(namespace, OrderedDict())
            ns[key] = value
            ns.move_to_end(key)
            self._enforce_capacity()

    # ── Read ──────────────────────────────────────────────────────────

    def retrieve(self, namespace: str, key: str) -> Optional[Any]:
        """
        Retrieve a single entry.

        Args:
            namespace: Target namespace.
            key: Entry key.

        Returns:
            The stored value, or ``None`` if not found.
        """
        with self._lock:
            ns = self._namespaces.get(namespace, {})
            return ns.get(key)

    def list_namespace(self, namespace: str) -> List[str]:
        """Return all keys in a namespace."""
        with self._lock:
            return list(self._namespaces.get(namespace, {}).keys())

    def get_all(self, namespace: str) -> Dict[str, Any]:
        """Return a shallow copy of all entries in a namespace."""
        with self._lock:
            return dict(self._namespaces.get(namespace, {}))

    # ── Delete ────────────────────────────────────────────────────────

    def remove(self, namespace: str, key: str) -> bool:
        """
        Remove an entry.

        Returns:
            ``True`` if the entry existed and was removed.
        """
        with self._lock:
            ns = self._namespaces.get(namespace)
            if ns and key in ns:
                del ns[key]
                return True
            return False

    def clear_namespace(self, namespace: str) -> None:
        """Remove all entries from a namespace."""
        with self._lock:
            self._namespaces.pop(namespace, None)

    def clear_all(self) -> None:
        """Wipe all working memory contents."""
        with self._lock:
            self._namespaces.clear()

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def total_entries(self) -> int:
        """Count of all entries across all namespaces."""
        with self._lock:
            return sum(len(ns) for ns in self._namespaces.values())

    @property
    def namespaces(self) -> List[str]:
        """List of active namespaces."""
        with self._lock:
            return list(self._namespaces.keys())

    # ── Internal ──────────────────────────────────────────────────────

    def _enforce_capacity(self) -> None:
        """Evict oldest entries until total is within capacity."""
        while self.total_entries > self._capacity:
            # Find the namespace with the oldest entry
            oldest_ns = None
            oldest_key = None
            for ns_name, ns_dict in self._namespaces.items():
                if ns_dict:
                    first_key = next(iter(ns_dict))
                    if oldest_ns is None:
                        oldest_ns = ns_name
                        oldest_key = first_key
                    # OrderedDict preserves insertion order; first = oldest
                    break
            if oldest_ns and oldest_key:
                del self._namespaces[oldest_ns][oldest_key]
                logger.debug(
                    "Evicted %s/%s (capacity enforcement)", oldest_ns, oldest_key
                )
            else:
                break
