"""
ACA MessageBus
==============

Implements the central publish-subscribe message broker for the
Agricultural Cognitive Architecture. All inter-component communication
flows through this bus.

Design Decisions:
    - Topic-based pub/sub using ``MessageType`` as topic keys.
    - Priority queue per topic for ordered consumption.
    - Synchronous dispatch (suitable for single-process simulation;
      can be swapped for async transport in production).
    - Full message history for tracing and replay.
    - Dependency-injected configuration via ``MessageBusConfig``.
"""

from __future__ import annotations

import heapq
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from aca.config import MessageBusConfig
from aca.logging_config import get_logger
from aca.orchestration.schemas import ACAMessage, MessageType

logger = get_logger("orchestration.message_bus")

# Subscriber callback signature: receives an ACAMessage, returns nothing.
Subscriber = Callable[[ACAMessage], None]


class MessageBus:
    """
    Publish-subscribe message broker for ACA components.

    Components subscribe to ``MessageType`` topics and receive all
    messages published to those topics, ordered by descending priority.

    Args:
        config: Bus configuration (queue limits, tracing flags).

    Example::

        bus = MessageBus(MessageBusConfig())
        bus.subscribe(MessageType.OBSERVATION, my_handler)
        bus.publish(obs_message)
    """

    def __init__(self, config: MessageBusConfig) -> None:
        self._config = config
        self._subscribers: Dict[MessageType, List[Subscriber]] = defaultdict(list)
        self._wildcard_subscribers: List[Subscriber] = []
        self._history: List[ACAMessage] = []
        self._lock = threading.Lock()
        self._pending: Dict[MessageType, List[Tuple[int, int, ACAMessage]]] = (
            defaultdict(list)
        )
        self._counter = 0  # tie-breaker for heapq stability
        logger.info(
            "MessageBus initialised (max_queue=%d, tracing=%s)",
            config.max_queue_size,
            config.enable_tracing,
        )

    # ── Subscription ──────────────────────────────────────────────────

    def subscribe(
        self,
        message_type: MessageType,
        callback: Subscriber,
    ) -> None:
        """
        Register a callback for a specific message type.

        Args:
            message_type: The topic to subscribe to.
            callback: A callable invoked with each matching ``ACAMessage``.
        """
        with self._lock:
            self._subscribers[message_type].append(callback)
        logger.debug("Subscriber added for %s", message_type.value)

    def subscribe_all(self, callback: Subscriber) -> None:
        """
        Register a callback that receives *all* message types.

        Useful for logging, tracing, or audit components.

        Args:
            callback: A callable invoked with every published ``ACAMessage``.
        """
        with self._lock:
            self._wildcard_subscribers.append(callback)
        logger.debug("Wildcard subscriber added")

    def unsubscribe(
        self,
        message_type: MessageType,
        callback: Subscriber,
    ) -> None:
        """
        Remove a previously registered callback.

        Args:
            message_type: The topic to unsubscribe from.
            callback: The exact callable reference to remove.
        """
        with self._lock:
            try:
                self._subscribers[message_type].remove(callback)
            except ValueError:
                pass

    # ── Publishing ────────────────────────────────────────────────────

    def publish(self, message: ACAMessage) -> int:
        """
        Publish a message to all subscribers of its type.

        Messages are validated before dispatch. If ``enable_tracing`` is
        active, the message is appended to the history log.

        Args:
            message: A fully constructed ``ACAMessage``.

        Returns:
            Number of subscribers that received the message.

        Raises:
            TypeError: If the message payload does not match its type.
        """
        message.validate()

        with self._lock:
            if self._config.enable_tracing:
                if len(self._history) < self._config.max_queue_size:
                    self._history.append(message)

            targets = list(self._subscribers.get(message.message_type, []))
            wildcards = list(self._wildcard_subscribers)

        dispatched = 0
        for cb in targets + wildcards:
            try:
                cb(message)
                dispatched += 1
            except Exception:
                logger.exception(
                    "Subscriber %s raised exception on %s",
                    cb,
                    message.uuid,
                )

        logger.debug(
            "Published %s [%s] → %d subscribers",
            message.message_type.value,
            message.uuid[:8],
            dispatched,
        )
        return dispatched

    # ── Queued (priority-ordered) dispatch ────────────────────────────

    def enqueue(self, message: ACAMessage) -> None:
        """
        Add a message to the priority queue for deferred processing.

        Messages are ordered by *descending* priority (5 = critical
        dispatched first). Use ``drain()`` to flush the queue.

        Args:
            message: A validated ``ACAMessage``.
        """
        message.validate()
        with self._lock:
            self._counter += 1
            heapq.heappush(
                self._pending[message.message_type],
                (-message.priority, self._counter, message),
            )

    def drain(self, message_type: Optional[MessageType] = None) -> List[ACAMessage]:
        """
        Flush queued messages, returning them in priority order.

        Args:
            message_type: If provided, drain only this topic.
                          If ``None``, drain all topics.

        Returns:
            List of ``ACAMessage`` instances in priority-descending order.
        """
        result: List[ACAMessage] = []
        with self._lock:
            types = (
                [message_type] if message_type else list(self._pending.keys())
            )
            for mt in types:
                heap = self._pending.get(mt, [])
                while heap:
                    _, _, msg = heapq.heappop(heap)
                    result.append(msg)
                if mt in self._pending:
                    self._pending[mt] = []
        return result

    # ── Introspection ─────────────────────────────────────────────────

    def get_history(
        self,
        message_type: Optional[MessageType] = None,
        limit: int = 100,
    ) -> List[ACAMessage]:
        """
        Retrieve published message history for tracing.

        Args:
            message_type: Optional filter by type.
            limit: Maximum number of messages to return.

        Returns:
            List of historical ``ACAMessage`` instances (most recent last).
        """
        with self._lock:
            if message_type:
                filtered = [
                    m for m in self._history if m.message_type == message_type
                ]
            else:
                filtered = list(self._history)
        return filtered[-limit:]

    @property
    def subscriber_count(self) -> Dict[str, int]:
        """Return a mapping of topic → number of subscribers."""
        with self._lock:
            return {
                mt.value: len(subs)
                for mt, subs in self._subscribers.items()
            }

    def clear_history(self) -> None:
        """Purge the message history."""
        with self._lock:
            self._history.clear()
