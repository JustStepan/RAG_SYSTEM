
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    SystemMessage,
)

from langgraph.graph.message import add_messages
from logger import logger
from tools import tools
from prompts import system_prompt
from models import local_model


local_model = local_model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState) -> bool:
    """Check if the last message contains tool_calls"""

    result = state["messages"][-1]
    logger.info(f'Применена логическая нода для сообщения: {state["messages"][-1].content[:70]}')
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


tools_dict = {
    our_tool.name: our_tool for our_tool in tools
}  # Creating a dictionary of our tools


def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = local_model.invoke(messages)
    return {"messages": [message]}


def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        logger.info(
            f"Вызванный тул: {t['name']} с запросом: {t['args'].get('query', 'No query provided')}"
        )

        if not t["name"] in tools_dict:  # Checks if a valid tool is present
            logger.error(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            logger.info(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(
            ToolMessage(
                tool_call_id=t["id"], name=t["name"], content=str(result)
            )
        )

    logger.info("Выполнение тула завершено! Двигаемся обратно к модели!")
    return {"messages": results}
