"""
Compatibility patches applied at import time.

Import this module BEFORE importing ragas anywhere in the codebase.

Issue
-----
ragas 0.4.x imports ``langchain_community.chat_models.vertexai.ChatVertexAI``
at module load time (ragas/llms/base.py:12).  That module was removed in
langchain-community 1.x (moved to ``langchain-google-vertexai``).

We inject a lightweight stub so ragas can import successfully without
installing the Google Cloud SDK or downgrading LangChain.
"""

import sys
import types
import logging

logger = logging.getLogger(__name__)

_PATCHED_MODULES: list[str] = []


def _patch_vertexai_stub() -> None:
    target = "langchain_community.chat_models.vertexai"
    if target in sys.modules:
        return

    stub = types.ModuleType(target)

    class ChatVertexAI:
        """Stub class — satisfies ragas import; never instantiated in this project."""
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[target] = stub
    _PATCHED_MODULES.append(target)
    logger.debug("Compatibility stub injected for: %s", target)


# ── Apply all patches immediately on import ───────────────────────────────────
_patch_vertexai_stub()
