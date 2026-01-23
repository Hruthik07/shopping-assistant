"""Debug routes (disabled in production by default)."""

from fastapi import APIRouter, HTTPException, Query
from src.agent.shopping_agent import get_shopping_agent
from src.utils.config import settings

import hashlib

router = APIRouter(prefix="/api/debug", tags=["debug"])


@router.get("/system-prompt")
async def get_system_prompt(
    persona: str = Query("friendly"),
    tone: str = Query("warm"),
):
    """Return the exact merged system prompt string being sent to the LLM.

    Disabled in production unless DEBUG_PROMPTS=true is set.
    """
    if getattr(settings, "production_mode", False) and (
        str(getattr(settings, "debug_prompts", "false")).lower() != "true"
    ):
        raise HTTPException(status_code=404, detail="Not found")

    agent = get_shopping_agent()
    prompt = agent._get_system_prompt(persona=persona, tone=tone)
    raw = agent._get_system_prompt_raw()

    return {
        "source_file": "src/agent/prompts/shopping_assistant.txt",
        "persona": persona,
        "tone": tone,
        "raw_prompt_sha256": hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest(),
        "merged_prompt_sha256": hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest(),
        "merged_prompt": prompt,
    }
