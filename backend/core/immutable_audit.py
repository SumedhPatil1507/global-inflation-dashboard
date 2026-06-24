import asyncpg
import json
from datetime import datetime
from backend.core.config import get_settings

_settings = get_settings()

async def log_audit(event_type: str, user_id: str | None, role: str | None, ip: str, request_signature: str | None, query_time_ms: float, rows: int | None, params: dict) -> None:
    """Insert an immutable audit record into the `security_audit_ledger` table.
    This function should be called within the same asyncpg transaction as the API handler when possible.
    """
    pool: asyncpg.pool.Pool = await asyncpg.create_pool(dsn=_settings.database_url, min_size=1, max_size=5)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO security_audit_ledger (
                event_type,
                user_id,
                role,
                ip_address,
                request_signature,
                query_time_ms,
                rows_affected,
                params,
                event_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            event_type,
            user_id,
            role,
            ip,
            request_signature,
            query_time_ms,
            rows,
            json.dumps(params),
            datetime.utcnow()
        )
