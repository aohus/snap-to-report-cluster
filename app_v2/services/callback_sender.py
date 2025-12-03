import logging
import httpx
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class CallbackSender:
    async def send_result(self, url: str, payload: Dict[str, Any], task_id: str):
        logger.info(f"📤 [Task {task_id}] Sending callback to {url}")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, timeout=10.0)
                response.raise_for_status()
                logger.info(f"✅ [Task {task_id}] Callback sent successfully. Status: {response.status_code}")
            except httpx.HTTPStatusError as e:
                logger.error(f"❌ [Task {task_id}] Callback failed: HTTP {e.response.status_code} - {e.response.text}")
            except Exception as e:
                logger.error(f"❌ [Task {task_id}] Callback connection error: {e}")

    async def send_error(self, url: str, error_message: str, task_id: str, request_id: Optional[str] = None):
        payload = {
            "task_id": task_id,
            "request_id": request_id,
            "status": "failed",
            "error": error_message
        }
        await self.send_result(url, payload, task_id)
