import os
import json
from datetime import datetime
from backend.core.config import get_settings
from backend.core.config import Settings
import asyncpg

_settings = get_settings()

# Async PostgreSQL connection pool for audit logging
_pool: asyncpg.pool.Pool = None

async def get_audit_pool() -> asyncpg.pool.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=_settings.database_url, min_size=1, max_size=5)
    return _pool

class AuditMiddleware:
    """FastAPI middleware that logs each request immutably to the audit table.
    The log includes:
      - user_id (from JWT payload "sub" if present)
      - role
      - ip address
      - endpoint (path)
      - method
      - request params (JSON string)
      - execution_time_ms
      - request_signature (optional, placeholder)
    It runs within the same DB transaction as the request handler when possible.
    """
    def __init__(self, app):
        self.app = app
        self.table = _settings.audit_table_name

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = datetime.utcnow()
        request = {
            "method": scope.get("method"),
            "path": scope.get("path"),
            "client": scope.get("client")[0] if scope.get("client") else None,
        }
        # Capture request body for logging
        body = b""
        async def receive_wrapper():
            nonlocal body
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
            return message
        # Process request
        await self.app(scope, receive_wrapper, send)
        end = datetime.utcnow()
        exec_ms = (end - start).total_seconds() * 1000
        # Extract JWT payload if available via headers
        token = None
        for header, value in scope.get("headers", []):
            if header.decode().lower() == "authorization":
                token = value.decode().split(" ")[-1]
                break
        user_id = None
        role = None
        if token:
            try:
                from backend.core.security import decode_token
                payload = decode_token(token)
                user_id = payload.get("sub") or payload.get("username")
                role = payload.get("role")
            except Exception:
                pass
        # Prepare log record
        log_record = {
            "user_id": user_id,
            "role": role,
            "ip": request["client"],
            "endpoint": request["path"],
            "method": request["method"],
            "params": body.decode(errors="ignore"),
            "execution_time_ms": exec_ms,
            "request_signature": None,
        }
        # Insert into DB
        pool = await get_audit_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {self.table} (user_id, role, ip, endpoint, method, params, execution_time_ms, request_signature) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                log_record["user_id"],
                log_record["role"],
                log_record["ip"],
                log_record["endpoint"],
                log_record["method"],
                json.dumps(log_record["params"]),
                log_record["execution_time_ms"],
                log_record["request_signature"],
            )

# Export middleware for import
__all__ = ["AuditMiddleware"]
