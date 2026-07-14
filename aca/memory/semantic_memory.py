"""
ACA Semantic Memory
===================

Holds internal agronomic knowledge: crop growth thresholds, disease
symptom maps, optimal temperature bands per phenological stage, and
general action policies.

Semantic Memory is the architecture's long-term factual knowledge base.
It is distinct from the Knowledge Layer (which stores *external*
references like research papers, government policies, and RAG indices).

Design Decisions:
    - Read-heavy, write-rare (optionally readonly via config).
    - Organised as a flat key-value store within named domains.
    - Supports bulk loading from JSON for initialisation.
    - Thread-safe.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from aca.config import MemoryConfig
from aca.logging_config import get_logger

logger = get_logger("memory.semantic")


class SemanticMemory:
    """
    Domain-partitioned store for internal agronomic facts and policies.

    Domains act as logical groupings (e.g. ``crop_thresholds``,
    ``disease_symptoms``, ``treatment_protocols``).

    Args:
        config: Memory configuration.

    Example::

        sm = SemanticMemory(MemoryConfig())
        sm.store("crop_thresholds", "rice_water_min", 0.35)
        val = sm.retrieve("crop_thresholds", "rice_water_min")
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._readonly = config.semantic_readonly
        self._domains: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._frozen = False
        logger.info(
            "SemanticMemory initialised (readonly=%s)", self._readonly
        )

    # ── Bulk Loading ──────────────────────────────────────────────────

    def load_from_dict(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Bulk-load domain data from a nested dictionary.

        Args:
            data: ``{domain: {key: value, ...}, ...}``

        Raises:
            RuntimeError: If memory has been frozen.
        """
        with self._lock:
            if self._frozen:
                raise RuntimeError("SemanticMemory is frozen (readonly)")
            for domain, entries in data.items():
                self._domains.setdefault(domain, {}).update(entries)
            logger.info("Loaded %d domains from dict", len(data))

    def load_from_file(self, path: str) -> None:
        """
        Bulk-load domain data from a JSON file.

        The JSON structure must be ``{domain: {key: value}}``.

        Args:
            path: Path to a JSON file.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.load_from_dict(data)

    def freeze(self) -> None:
        """
        Freeze memory, preventing further writes.

        Once frozen, ``store()`` and ``remove()`` will raise
        ``RuntimeError``.
        """
        with self._lock:
            self._frozen = True
            logger.info("SemanticMemory frozen")

    # ── Write ─────────────────────────────────────────────────────────

    def store(self, domain: str, key: str, value: Any) -> None:
        """
        Store a fact under a domain/key pair.

        Args:
            domain: Knowledge domain.
            key: Fact identifier.
            value: The fact value.

        Raises:
            RuntimeError: If memory is frozen or readonly after init.
        """
        with self._lock:
            if self._frozen:
                raise RuntimeError("SemanticMemory is frozen")
            self._domains.setdefault(domain, {})[key] = value

    # ── Read ──────────────────────────────────────────────────────────

    def retrieve(self, domain: str, key: str) -> Optional[Any]:
        """Retrieve a single fact by domain and key."""
        with self._lock:
            return self._domains.get(domain, {}).get(key)

    def list_domain(self, domain: str) -> List[str]:
        """List all keys in a domain."""
        with self._lock:
            return list(self._domains.get(domain, {}).keys())

    def get_domain(self, domain: str) -> Dict[str, Any]:
        """Return a shallow copy of all entries in a domain."""
        with self._lock:
            return dict(self._domains.get(domain, {}))

    @property
    def domains(self) -> List[str]:
        """List of registered domains."""
        with self._lock:
            return list(self._domains.keys())

    # ── Delete ────────────────────────────────────────────────────────

    def remove(self, domain: str, key: str) -> bool:
        """
        Remove a fact.

        Raises:
            RuntimeError: If memory is frozen.
        """
        with self._lock:
            if self._frozen:
                raise RuntimeError("SemanticMemory is frozen")
            d = self._domains.get(domain)
            if d and key in d:
                del d[key]
                return True
            return False
