"""OpenTelemetry setup and small helpers for application level spans."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:  # SDK can still collect spans when an exporter is not installed.
        OTLPSpanExporter = None
    TELEMETRY_AVAILABLE = True
except ImportError:
    trace = None
    Status = StatusCode = None
    TracerProvider = Resource = BatchSpanProcessor = ConsoleSpanExporter = None
    OTLPSpanExporter = None
    TELEMETRY_AVAILABLE = False


def _configure_provider() -> None:
    if not TELEMETRY_AVAILABLE:
        return
    current = trace.get_tracer_provider()
    # Respect a provider configured by the hosting application or auto-instrumentation.
    if current.__class__.__name__ != "ProxyTracerProvider":
        return
    provider = TracerProvider(resource=Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "global-inflation-dashboard"),
    }))
    traces_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    traces_exporter = os.getenv("OTEL_TRACES_EXPORTER", "otlp" if traces_endpoint else "none").lower()
    if traces_exporter == "otlp" and OTLPSpanExporter is not None:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    elif traces_exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


_configure_provider()
_tracer = trace.get_tracer("global-inflation-dashboard.macro-agent") if TELEMETRY_AVAILABLE else None


@contextmanager
def start_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span and record failures without changing application behavior."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None and isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
