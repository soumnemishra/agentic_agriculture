"""
ACA Episodic Memory
===================

Stores chronological execution episodes — each episode captures the
full lifecycle of an intervention: the initial state, planned actions,
executed actions, resulting state, and yield impact.

Episodic Memory enables retrospective learning: the Learning layer
queries past episodes to identify patterns of success and failure,
adjust crop models, and calibrate belief priors.

Design Decisions:
    - Episodes are immutable once committed (append-only store).
    - Queryable by time range, zone, outcome assessment, and tags.
    - Configurable retention window (default 365 days).
    - Thread-safe.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aca.config import MemoryConfig
from aca.logging_config import get_logger

logger = get_logger("memory.episodic")


@dataclass(frozen=True)
class Episode:
    """
    An immutable record of a single intervention lifecycle.

    Attributes:
        episode_id: Unique identifier.
        timestamp: ISO-8601 creation time.
        zone: Farm zone where the episode occurred.
        initial_state: Snapshot of relevant state before intervention.
        planned_actions: What the planner intended.
        executed_actions: What was actually dispatched.
        resulting_state: State snapshot after execution.
        outcome_assessment: Qualitative verdict (e.g. ``SUCCESS``, ``PARTIAL``).
        yield_impact: Estimated impact on yield (positive = beneficial).
        tags: Searchable metadata tags.
    """

    episode_id: str
    timestamp: str
    zone: str
    initial_state: Dict[str, Any]
    planned_actions: List[Dict[str, Any]]
    executed_actions: List[Dict[str, Any]]
    resulting_state: Dict[str, Any]
    outcome_assessment: str = ""
    yield_impact: float = 0.0
    tags: tuple = ()  # frozen-compatible (immutable)


class EpisodicMemory:
    """
    Append-only chronological store of intervention episodes.

    Args:
        config: Memory configuration (retention window).

    Example::

        em = EpisodicMemory(MemoryConfig())
        em.commit(episode)
        recent = em.query(zone="field_1_zone_a", limit=10)
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._retention_days = config.episodic_retention_days
        self._episodes: List[Episode] = []
        self._index_by_zone: Dict[str, List[int]] = {}
        self._index_by_id: Dict[str, int] = {}
        self._lock = threading.RLock()
        logger.info(
            "EpisodicMemory initialised (retention=%d days)",
            self._retention_days,
        )

    # ── Write ─────────────────────────────────────────────────────────

    def commit(self, episode: Episode) -> None:
        """
        Commit an episode to long-term episodic storage.

        Args:
            episode: A fully populated ``Episode`` dataclass.

        Raises:
            ValueError: If an episode with the same ID already exists.
        """
        with self._lock:
            if episode.episode_id in self._index_by_id:
                raise ValueError(
                    f"Episode {episode.episode_id} already committed"
                )
            idx = len(self._episodes)
            self._episodes.append(episode)
            self._index_by_id[episode.episode_id] = idx
            self._index_by_zone.setdefault(episode.zone, []).append(idx)
            logger.debug("Committed episode %s", episode.episode_id)

    # ── Read ──────────────────────────────────────────────────────────

    def get(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID."""
        with self._lock:
            idx = self._index_by_id.get(episode_id)
            return self._episodes[idx] if idx is not None else None

    def query(
        self,
        zone: Optional[str] = None,
        assessment: Optional[str] = None,
        tag: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50,
    ) -> List[Episode]:
        """
        Query episodes with optional filters.

        Args:
            zone: Filter by farm zone.
            assessment: Filter by outcome assessment string.
            tag: Filter by tag membership.
            since: ISO-8601 timestamp; only episodes after this time.
            limit: Maximum results to return.

        Returns:
            List of matching episodes, most recent first.
        """
        with self._lock:
            if zone and zone in self._index_by_zone:
                candidates = [
                    self._episodes[i] for i in self._index_by_zone[zone]
                ]
            else:
                candidates = list(self._episodes)

        # Apply filters
        results: List[Episode] = []
        for ep in reversed(candidates):
            if assessment and ep.outcome_assessment != assessment:
                continue
            if tag and tag not in ep.tags:
                continue
            if since and ep.timestamp < since:
                continue
            results.append(ep)
            if len(results) >= limit:
                break
        return results

    def count(self) -> int:
        """Total number of committed episodes."""
        with self._lock:
            return len(self._episodes)

    def zones(self) -> List[str]:
        """List all zones with at least one episode."""
        with self._lock:
            return list(self._index_by_zone.keys())
