import re

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command
from typing_extensions import Annotated

# 用于匹配空白字符的正则表达式
WHITESPACE_RE = re.compile(r"\s+")
# 元数据键名，用于标识交接工具的目标智能体
METADATA_KEY_HANDOFF_DESTINATION = "__handoff_destination"


def _normalize_agent_name(agent_name: str) -> str:
    """标准化智能体名称，使其适合用作工具名称的一部分。
    
    将智能体名称中的空白字符替换为下划线，并转换为小写。
    """
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()


def create_handoff_tool(*, agent_name: str, description: str | None = None) -> BaseTool:
    """创建一个可以将控制权交接给指定智能体的工具。

    参数:
        agent_name: 要交接控制权的智能体名称，即多智能体图中智能体节点的名称。
            智能体名称应该简单、清晰且唯一，最好使用snake_case格式，
            虽然只受限于LangGraph节点接受的名称以及LLM提供商接受的工具名称
            (工具名称将类似于: `transfer_to_<智能体名称>`)。
        description: 交接工具的可选描述。
    """
    # 生成工具名称，格式为 transfer_to_<标准化后的智能体名称>
    name = f"transfer_to_{_normalize_agent_name(agent_name)}"
    # 如果没有提供描述，使用默认描述
    if description is None:
        description = f"向智能体 '{agent_name}' 请求帮助"

    # 使用@tool装饰器创建工具
    @tool(name, description=description)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],  # 注入的状态字典
        tool_call_id: Annotated[str, InjectedToolCallId],  # 注入的工具调用ID
    ):
        # 创建工具消息，表示成功交接
        tool_message = ToolMessage(
            content=f"已成功交接至智能体 {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )
        # 返回命令，指示转移到指定智能体
        return Command(
            goto=agent_name,  # 目标智能体名称
            graph=Command.PARENT,  # 引用父图
            update={
                "messages": state["messages"] + [tool_message],  # 更新消息列表
                "active_agent": agent_name  # 设置当前活跃智能体
            },
        )

    # 在工具元数据中记录目标智能体名称
    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return handoff_to_agent


def get_handoff_destinations(agent: CompiledStateGraph, tool_node_name: str = "tools") -> list[str]:
    """从智能体的交接工具中获取目标智能体名称列表。
    
    参数:
        agent: 已编译的状态图智能体
        tool_node_name: 工具节点的名称，默认为"tools"
    """
    # 获取图中所有节点
    nodes = agent.get_graph().nodes
    # 如果图中没有工具节点，返回空列表
    if tool_node_name not in nodes:
        return []

    # 获取工具节点数据
    tool_node = nodes[tool_node_name].data
    # 如果不是ToolNode类型，返回空列表
    if not isinstance(tool_node, ToolNode):
        return []

    # 获取所有工具
    tools = tool_node.tools_by_name.values()
    # 返回所有有交接目标元数据的工具的目标智能体名称列表
    return [
        tool.metadata[METADATA_KEY_HANDOFF_DESTINATION]
        for tool in tools
        if tool.metadata is not None and METADATA_KEY_HANDOFF_DESTINATION in tool.metadata
    ]