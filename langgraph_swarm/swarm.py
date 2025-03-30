from typing import Type, TypeVar

from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from langgraph_swarm.handoff import get_handoff_destinations


class SwarmState(MessagesState):
    """多智能体集群（Swarm）的状态模式。

    继承自 MessagesState，并添加 active_agent 字段来跟踪当前活跃智能体。
    """

    active_agent: str  # 当前活跃的智能体名称


# 定义状态模式的类型变量
StateSchema = TypeVar("StateSchema", bound=SwarmState)
StateSchemaType = Type[StateSchema]


def add_active_agent_router(
    builder: StateGraph,
    *,
    route_to: list[str],
    default_active_agent: str,
) -> StateGraph:
    """向状态图中添加活跃智能体路由逻辑。

    参数:
        builder: 要添加路由器的状态图构建器（StateGraph）
        route_to: 可路由到的智能体（节点）名称列表
        default_active_agent: 默认路由的智能体名称（当没有活跃智能体时使用）

    返回:
        添加了路由器的 StateGraph

    异常:
        ValueError: 如果 builder 的状态模式缺少 'active_agent' 字段
                  或默认智能体不在路由列表中
    """
    # 检查状态模式是否包含 active_agent 字段
    channels = builder.schemas[builder.schema]
    if "active_agent" not in channels:
        raise ValueError("构建器的 state_schema 中缺少必需的 'active_agent' 字段")

    # 验证默认智能体是否在路由列表中
    if default_active_agent not in route_to:
        raise ValueError(
            f"默认活跃智能体 '{default_active_agent}' 不在路由列表 {route_to} 中"
        )

    # 定义路由函数：返回当前活跃智能体或默认智能体
    def route_to_active_agent(state: dict):
        return state.get("active_agent", default_active_agent)

    # 添加条件边：从 START 节点根据路由函数跳转到对应智能体
    builder.add_conditional_edges(START, route_to_active_agent, path_map=route_to)
    return builder


def create_swarm(
    agents: list[CompiledStateGraph],
    *,
    default_active_agent: str,
    state_schema: StateSchemaType = SwarmState,
) -> StateGraph:
    """创建多智能体集群。

    参数:
        agents: 要加入集群的智能体列表
        default_active_agent: 默认路由的智能体名称（当没有活跃智能体时使用）
        state_schema: 用于多智能体图的状态模式（默认为 SwarmState）

    返回:
        构建完成的多智能体集群 StateGraph

    异常:
        ValueError: 如果状态模式缺少 'active_agent' 字段
    """
    # 验证状态模式是否包含 active_agent 字段
    if "active_agent" not in state_schema.__annotations__:
        raise ValueError("state_schema 中缺少必需的 'active_agent' 字段")

    # 初始化状态图构建器
    builder = StateGraph(state_schema)
    
    # 添加活跃智能体路由器
    add_active_agent_router(
        builder,
        route_to=[agent.name for agent in agents],  # 所有智能体名称作为路由目标
        default_active_agent=default_active_agent,
    )
    
    # 将每个智能体添加为图中的节点
    for agent in agents:
        builder.add_node(
            agent.name,  # 节点名称 = 智能体名称
            agent,  # 编译后的智能体图
            destinations=tuple(get_handoff_destinations(agent)),  # 可交接的目标智能体
        )

    return builder
