# main.py
import contextlib
from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncGenerator
import time
from uuid import uuid4
import json

import os

from agents.mcp import (
    MCPServerStreamableHttp,
)  # Adjust import based on your MCP library
from agents import (
    Runner,
    InputGuardrailTripwireTriggered,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents_manager import AgentsManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create a lifespan to manage session manager
@contextlib.asynccontextmanager
async def lifespan(server: FastAPI):
    logger.info("Starting up - connecting to MCP server")

    # Create and store the MCP server connection in app state
    app.state.mcp_server = MCPServerStreamableHttp(
        name="StreamableHttp Container App Server",
        params={
            "url": os.environ.get("MCP_SERVER_URL"),
            "headers": {"x-api-key": os.environ.get("MCP_API_KEY")},
        },
    )

    # Enter the async context manager
    await app.state.mcp_server.__aenter__()

    try:
        logger.info("MCP server connection established")
        # Create and store the agents manager in app state
        app.state.agents_manager = AgentsManager().initialize(app.state.mcp_server)
        logger.info("Agents manager initialized successfully")
        yield
    finally:
        logger.info("Shutting down - closing MCP server connection")
        await app.state.mcp_server.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)

# CORS (only needed if you're calling from a browser front-end)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Get the MCP Server and Agent
# =========================


# Dependency to get MCP server
async def get_mcp_server(request: Request):
    """Dependency to provide the MCP server connection"""
    if (
        not hasattr(request.app.state, "mcp_server")
        or request.app.state.mcp_server is None
    ):
        raise RuntimeError("MCP server is not initialized")
    return request.app.state.mcp_server


# Dependency to get the agents manager
async def get_agents_manager(request: Request):
    """Dependency to provide the agents manager"""
    if (
        not hasattr(request.app.state, "agents_manager")
        or request.app.state.agents_manager is None
    ):
        raise RuntimeError("Agents manager is not initialized")
    return request.app.state.agents_manager


# Dependency to get a specific agent or the default agent
async def get_agent(request: Request, agent_id: Optional[str] = None):
    """Dependency to provide an agent by ID or the default agent"""
    agents_manager = await get_agents_manager(request)
    return agents_manager.get_agent(agent_id)


# =========================
# Models (same as before)
# =========================


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    agent_id: Optional[str] = None  # Added to support specifying which agent to use


class MessageResponse(BaseModel):
    content: str
    agent: str


class AgentEvent(BaseModel):
    id: str
    type: str
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


class GuardrailCheck(BaseModel):
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float


class ChatResponse(BaseModel):
    conversation_id: str
    current_agent: str
    messages: List[MessageResponse]
    events: List[AgentEvent]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[GuardrailCheck] = []


# =========================
# In-memory store for conversation state
# =========================


class ConversationStore:
    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        pass

    def save(self, conversation_id: str, state: Dict[str, Any]):
        pass


class InMemoryConversationStore(ConversationStore):
    _conversations: Dict[str, Dict[str, Any]] = {}

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def save(self, conversation_id: str, state: Dict[str, Any]):
        self._conversations[conversation_id] = state


# TODO: when deploying this app in scale, switch to your own production-ready implementation
conversation_store = InMemoryConversationStore()

# =========================
# Helpers
# =========================


def _get_guardrail_name(g) -> str:
    """Extract a friendly guardrail name."""
    name_attr = getattr(g, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    guard_fn = getattr(g, "guardrail_function", None)
    if guard_fn is not None and hasattr(guard_fn, "__name__"):
        return guard_fn.__name__.replace("_", " ").title()
    fn_name = getattr(g, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name.replace("_", " ").title()
    return str(g)


async def generate_chat_stream(
    request: ChatRequest, agents_manager: AgentsManager
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE-formatted JSON data for streaming chat responses."""
    
    logger.info(
        f"Streaming request conversation_id: {request.conversation_id} - Message: {request.message}"
    )

    # Get the specified agent or the default agent
    current_agent = agents_manager.get_agent(request.agent_id)
    agent_id = request.agent_id or agents_manager.default_agent_id

    # Use one of the sports tools
    message = request.message
    logger.info(f"\n\nRunning streamed: {message}")

    guardrail_checks: List[GuardrailCheck] = []

    try:
        result = Runner.run_streamed(
            starting_agent=current_agent, input=message, max_turns=30
        )
    except InputGuardrailTripwireTriggered as e:
        failed = e.guardrail_result.guardrail
        gr_output = e.guardrail_result.output.output_info
        gr_reasoning = getattr(gr_output, "reasoning", "")
        gr_input = request.message
        gr_timestamp = time.time() * 1000
        for g in current_agent.input_guardrails:
            guardrail_checks.append(
                GuardrailCheck(
                    id=uuid4().hex,
                    name=_get_guardrail_name(g),
                    input=gr_input,
                    reasoning=(gr_reasoning if g == failed else ""),
                    passed=(g != failed),
                    timestamp=gr_timestamp,
                )
            )
        refusal = "Sorry, I can only answer questions related to sports news."
        
        # Send guardrail failure as a stream event
        error_response = {
            "type": "error",
            "data": {
                "conversation_id": request.conversation_id,
                "current_agent": current_agent.name,
                "message": refusal,
                "guardrails": [gc.model_dump() for gc in guardrail_checks],
                "timestamp": time.time()
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        return

    messages: List[MessageResponse] = []
    events: List[AgentEvent] = []

    # Send initial response
    initial_response = {
        "type": "start",
        "data": {
            "conversation_id": request.conversation_id or str(uuid4()),
            "current_agent": agent_id,
            "timestamp": time.time()
        }
    }
    yield f"data: {json.dumps(initial_response)}\n\n"

    async for event in result.stream_events():
        if event.type == "run_item_stream_event":
            print(f"Got event of type {event.item.__class__.__name__}")

            if isinstance(event.item, ToolCallItem):
                tool_name = getattr(event.item.raw_item, "name", None)
                raw_args = getattr(event.item.raw_item, "arguments", None)
                tool_args: Any = raw_args
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args)
                    except Exception:
                        pass
                
                agent_event = AgentEvent(
                    id=uuid4().hex,
                    type="tool_call",
                    agent=event.item.agent.name,
                    content=tool_name or "",
                    metadata={"tool_args": tool_args},
                    timestamp=time.time()
                )
                events.append(agent_event)
                
                # Stream the tool call event
                stream_event = {
                    "type": "tool_call",
                    "data": agent_event.model_dump()
                }
                yield f"data: {json.dumps(stream_event)}\n\n"
                
                # If the tool is display_seat_map, send a special message so the UI can render the seat selector.
                if tool_name == "display_seat_map":
                    message_response = MessageResponse(
                        content="DISPLAY_SEAT_MAP",
                        agent=event.item.agent.name,
                    )
                    messages.append(message_response)
                    
                    # Stream the special message
                    message_event = {
                        "type": "message",
                        "data": message_response.model_dump()
                    }
                    yield f"data: {json.dumps(message_event)}\n\n"
                    
            elif isinstance(event.item, ToolCallOutputItem):
                agent_event = AgentEvent(
                    id=uuid4().hex,
                    type="tool_output",
                    agent=event.item.agent.name,
                    content=str(event.item.output),
                    metadata={"tool_result": event.item.output},
                    timestamp=time.time()
                )
                events.append(agent_event)
                
                # Stream the tool output event
                stream_event = {
                    "type": "tool_output",
                    "data": agent_event.model_dump()
                }
                yield f"data: {json.dumps(stream_event)}\n\n"

    logger.info(result.final_output)

    # Send final response with all collected data
    final_response = {
        "type": "complete",
        "data": {
            "conversation_id": request.conversation_id or str(uuid4()),
            "current_agent": agent_id,
            "messages": [msg.model_dump() for msg in messages],
            "events": [event.model_dump() for event in events],
            "context": {},
            "agents": [],
            "guardrails": [],
            "final_output": result.final_output,
            "timestamp": time.time()
        }
    }
    yield f"data: {json.dumps(final_response)}\n\n"


@app.post("/chat")
async def chat_stream_endpoint(
    request: ChatRequest, agents_manager=Depends(get_agents_manager)
):
    """Streaming chat endpoint that returns Server-Sent Events (SSE)."""
    return StreamingResponse(
        generate_chat_stream(request, agents_manager),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
