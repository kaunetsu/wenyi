"""Route registered operations through reusable adapters and one usage ledger."""

from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import contextmanager
from threading import Lock
from uuid import uuid4

from .base import LLMClient, Messages
from .configuration import LLMConfig
from .limits import RequestCancelled, RequestLimits, RequestStopped
from .operations import require_operation
from .registry import provider_spec
from .retrying import _provider_error_metadata, is_retryable_provider_error
from .routing import ResolvedRoute, model_route, resolve_routes
from .transport import ProviderAdapter, RequestContext
from .usage import UsageSample


class RoutedLLMClient(LLMClient):
    """Freeze one routing plan per invocation; adapters never own cumulative usage."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config = LLMConfig.model_validate(config.model_dump())
        self.routes = resolve_routes(self.config)
        self.limits = RequestLimits(self.config)
        self._adapters: dict[str, ProviderAdapter] = {}
        self._adapter_lock = Lock()

    def adapter(self, connection: str) -> ProviderAdapter:
        with self._adapter_lock:
            if connection not in self._adapters:
                cfg = self.config.providers[connection]
                self._adapters[connection] = provider_spec(cfg.kind).adapter_type()(cfg)
            return self._adapters[connection]

    def validate_credentials(self, operations: Iterable[str] | None = None) -> None:
        connections: set[str] = set()
        for operation in self.routes if operations is None else operations:
            require_operation(operation)
            route = self.routes[operation]
            connections.add(route.provider)
            connections.update(self.config.models[profile].provider for profile in route.fallbacks)
            self._validate_token_reservation(route)
            for profile in route.fallbacks:
                self._validate_token_reservation(
                    model_route(self.config, operation, profile, origin="fallback")
                )
        for connection in sorted(connections):
            self.adapter(connection).validate_credentials()

    def cancel(self) -> None:
        self.limits.cancel()

    def complete(
        self,
        messages: Messages,
        *,
        operation: str,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        require_operation(operation)
        primary = self.routes[operation]
        return self._complete(messages, primary, json_mode=json_mode, max_tokens=max_tokens)

    def _complete(
        self,
        messages: Messages,
        primary: ResolvedRoute,
        *,
        json_mode: bool,
        max_tokens: int | None = None,
    ) -> str:
        operation = primary.operation
        if max_tokens is not None:
            primary = model_route(
                self.config,
                operation,
                primary.profile,
                origin=primary.origin,
                tier=primary.tier,
                fallbacks=primary.fallbacks,
                output_hint=max_tokens,
            )
        routes = [
            primary,
            *(
                model_route(
                    self.config,
                    operation,
                    profile,
                    origin="explicit fallback",
                    output_hint=max_tokens,
                )
                for profile in primary.fallbacks
            ),
        ]
        call_id = uuid4().hex
        input_bytes: int | None = None
        attempt_number = 0
        current_route = primary
        current_route_index = 0
        fallback_count = 0

        def route_metadata(route: ResolvedRoute, route_index: int) -> dict[str, object]:
            return {
                "call_id": call_id,
                "operation": operation,
                "stage": operation,
                "tier": route.tier or "direct",
                "profile": route.profile,
                "connection": route.provider,
                "provider": route.provider_kind,
                "model": route.model,
                "inference_fingerprint": route.fingerprint,
                "input_bytes": input_bytes,
                "message_count": len(messages),
                "json_mode": json_mode,
                "max_output_tokens": route.max_output_tokens,
                "route_index": route_index,
            }

        def emit_terminal(outcome: str, error: BaseException | None = None) -> None:
            error_fields = _provider_error_metadata(error) if error is not None else None
            self._emit_event(
                "llm_call_finished",
                **route_metadata(current_route, current_route_index),
                attempt=attempt_number,
                logical_elapsed_ms=round(max(0.0, self.limits.clock() - logical_started) * 1000, 3),
                outcome=outcome,
                total_attempts=attempt_number,
                final_status_code=(error_fields or {}).get("status_code"),
                final_reason=(
                    outcome
                    if outcome in {"cancelled", "stopped"}
                    else (error_fields or {}).get("reason")
                ),
                used_fallback=fallback_count > 0,
                fallback_count=fallback_count,
                **(
                    {
                        "request_id": error_fields["request_id"],
                        "error_type": error_fields["error_type"],
                    }
                    if error_fields is not None
                    else {}
                ),
            )

        logical_started = self.limits.clock()
        try:
            input_bytes = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
            for position, route in enumerate(routes):
                self.limits.check()
                current_route = route
                current_route_index = position
                fallback_count = position
                metadata = route_metadata(route, position)

                def emit(event: str, **payload) -> None:
                    self._emit_event(event, **{**metadata, "attempt": attempt_number, **payload})

                emit("llm_request_scheduled")
                active_reservation = None

                def record(sample: UsageSample | None) -> None:
                    if sample is not None and active_reservation is not None:
                        active_reservation.actual_tokens = sample.total_tokens
                    self.usage.record(
                        route.tier or "direct",
                        sample,
                        operation,
                        provider=route.provider_identity,
                        model=route.model_identity,
                        labels={
                            route.provider_identity: (
                                f"{route.provider_kind} {route.endpoint or ''}".strip()
                            ),
                            route.model_identity: f"{route.provider_kind} / {route.model}",
                        },
                    )

                @contextmanager
                def attempt_scope():
                    nonlocal active_reservation, attempt_number
                    self._validate_token_reservation(route)
                    estimate = input_bytes + 256 + (route.max_output_tokens or 0)
                    with self.limits.attempt(route.provider, estimate, emit) as reservation:
                        active_reservation = reservation
                        attempt_number += 1
                        emit("llm_request_started", estimated_tokens=estimate)
                        try:
                            yield
                        finally:
                            active_reservation = None

                context = RequestContext(
                    operation=operation,
                    tier=route.tier or "direct",
                    max_tokens=route.max_output_tokens,
                    emit=emit,
                    record_usage=record,
                    attempt_scope=attempt_scope,
                    sleep=self.limits.wait_for_retry,
                    clock=self.limits.clock,
                    route_index=position,
                )
                try:
                    result = self.adapter(route.provider).generate(
                        [dict(message) for message in messages],
                        route.request_model(),
                        json_mode=json_mode,
                        context=context,
                    )
                except Exception as error:
                    emit("llm_request_failed", error_type=type(error).__name__)
                    if position == len(routes) - 1 or not is_retryable_provider_error(error):
                        raise
                    emit("llm_model_failover", next_profile=routes[position + 1].profile)
                else:
                    emit("llm_request_completed")
                    emit_terminal("success")
                    return result
            raise RuntimeError("No model route was selected")
        except RequestCancelled as error:
            emit_terminal("cancelled", error)
            raise
        except RequestStopped as error:
            emit_terminal("stopped", error)
            raise
        except BaseException as error:
            emit_terminal("cancelled" if self.limits.cancelled.is_set() else "failed", error)
            raise

    def _validate_token_reservation(self, route: ResolvedRoute) -> None:
        connection = self.config.providers[route.provider]
        quota = self.config.quotas.get(connection.quota_group or "")
        if (
            self.config.budget.max_tokens or (quota and quota.tokens_per_minute)
        ) and route.max_output_tokens is None:
            raise ValueError(
                f"{route.operation}: token limits require an explicit max_output_tokens"
            )
