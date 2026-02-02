from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
import sys
from typing import Any, Dict

import aiohttp

SERPER_API_KEY = ""  # Set via environment variable SERPER_API_KEY
SERPER_BASE_URL = "https://google.serper.dev/search"


def get_api_key() -> str:
    return os.getenv("SERPER_API_KEY", SERPER_API_KEY)


def get_base_url() -> str:
    return os.getenv("SERPER_BASE_URL", SERPER_BASE_URL)


async def serper_search(
    api_key: str,
    query: str,
    base_url: str,
    max_results: int = 8,
) -> Dict[str, Any]:
    # 🔁 对齐 patch 后 SerperWrapper.get_headers 的逻辑
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 🔁 对齐 patch 后 SerperWrapper.get_payloads 的逻辑
    payload = {
        "q": query,
        "num": max_results,
        # 如果你在 SerperWrapper 里有 self.payload 里默认字段（比如 type: "search"）
        # 可以在这里手动加上：
        # "type": "search",
    }

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        # patch 后 SerperWrapper 用的是 data=JSON_STRING，这里两种都可以：
        # 1) data=json.dumps(payload)
        # 2) json=payload
        # 为了和 SerperWrapper 尽量一致，这里用 data=...
        async with session.post(
            base_url,
            data=json.dumps(payload),
            headers=headers,
        ) as resp:
            data = await resp.json(content_type=None)
            resp.raise_for_status()
            return data


async def main():
    if len(sys.argv) < 2:
        print("Usage: python run_search.py \"your query\"")
        sys.exit(1)

    query = sys.argv[1]
    api_key = get_api_key()
    base_url = get_base_url()

    if not api_key:
        print("Missing SERPER_API_KEY. Set env SERPER_API_KEY or update SERPER_API_KEY constant.")
        sys.exit(1)

    try:
        data = await serper_search(api_key, query, base_url=base_url)
    except Exception as e:
        print(f"Search failed: {e}")
        sys.exit(1)

    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

    