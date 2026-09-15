"""
Fixture: Async functions and async class methods.
"""


async def fetch_data(url: str) -> dict:
    """Fetch remote payload asynchronously."""
    return {"url": url}


class AsyncService:
    """Async service handler."""

    async def execute_task(self, task_id: str) -> bool:
        """Execute task asynchronously."""
        return True
