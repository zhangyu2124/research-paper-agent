"""Research paper agent with local paper search tools."""

import logging
import os

from langchain.agents import create_agent

from src.agent.config import (
    default_model,
    model_fallback_middleware,
    model_retry_middleware,
    tool_retry_middleware,
)
from src.prompts.paper_agent_prompt import paper_agent_prompt as _local_prompt
from src.tools.paper_tools import (
    get_paper_chunk_context,
    get_paper_detail,
    list_papers,
    search_paper_evidence,
    search_papers,
)

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.info("Research paper agent module loaded")

docs_agent_prompt = _local_prompt
prompt_commit = None
prompt_source = "local:src/prompts/paper_agent_prompt.py"
logger.info("Using local research paper agent prompt")

docs_agent_tools = [
    search_papers,
    search_paper_evidence,
    get_paper_detail,
    get_paper_chunk_context,
    list_papers,
]

docs_agent_middleware = [
    middleware
    for middleware in [
        tool_retry_middleware,
        model_retry_middleware,
        model_fallback_middleware,
    ]
    if middleware is not None
]

docs_agent = create_agent(
    model=default_model,
    tools=docs_agent_tools,
    system_prompt=docs_agent_prompt,
    middleware=docs_agent_middleware,
)

_prompt_metadata: dict[str, str] = {
    "prompt_source": prompt_source,
}
if prompt_commit:
    _prompt_metadata["prompt_commit"] = prompt_commit
if _revision_id := os.environ.get("LANGCHAIN_REVISION_ID"):
    _prompt_metadata["LANGSMITH_AGENT_VERSION"] = _revision_id

docs_agent = docs_agent.with_config(metadata=_prompt_metadata)
docs_agent.tools = docs_agent_tools
docs_agent.middleware = docs_agent_middleware
