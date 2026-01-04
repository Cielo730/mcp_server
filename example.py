import asyncio
import sys
import os
import json
from typing import Annotated, Literal, TypedDict, List, Any, Optional, Type
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import create_model, Field

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# MCP 工具适配器 
def _get_python_type(json_type: str) -> Type:
    type_map = {
        "string": str, "integer": int, "number": float, 
        "boolean": bool, "array": list, "object": dict
    }
    return type_map.get(json_type, str)

async def get_mcp_tools(session: ClientSession) -> List[StructuredTool]:
    """获取 MCP 工具并转换为 LangChain 工具 (带 Debug 功能)"""
    mcp_tools_list = await session.list_tools()
    langchain_tools = []

    for mcp_tool in mcp_tools_list.tools:
        # 动态构建参数模型
        schema = mcp_tool.inputSchema
        fields = {}
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})
        
        for prop_name, prop_def in properties.items():
            py_type = _get_python_type(prop_def.get("type", "string"))
            desc = prop_def.get("description", "")
            if prop_name in required_fields:
                fields[prop_name] = (py_type, Field(description=desc))
            else:
                fields[prop_name] = (Optional[py_type], Field(default=None, description=desc))
        
        # 创建 Pydantic 模型
        InputModel = create_model(f"{mcp_tool.name}_Input", **fields)

        current_tool_name = mcp_tool.name

        # 定义执行逻辑
        async def _dynamic_tool_func(*args, _name=current_tool_name, **kwargs):   #每次循环只执行:前的流程，例如会更新_name=current_tool_name，但不执行tool_name = _name
            tool_name = _name
            # [Debug] 打印请求参数
            print(f"\n[执行工具] {tool_name} | 参数: {kwargs}")
            
            try:
                # 清理 None 参数
                clean_args = {k: v for k, v in kwargs.items() if v is not None}
                
                # 调用 MCP Server
                result = await session.call_tool(tool_name, arguments=clean_args)
                
                # 提取文本结果
                if not result.content:
                    output = "Success (No content returned)"
                else:
                    output = "\n".join([c.text for c in result.content if hasattr(c, 'text')])
                
                # 打印成功结果
                print(f"[工具返回] {output[:100]}..." if len(output) > 100 else f"[工具返回] {output}")
                return output

            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                print(f" [工具报错] {error_msg}")
                # 返回错误信息给 LLM，让它知道出错了，而不是无限重试
                return error_msg

        # 包装为 LangChain 工具
        lt = StructuredTool.from_function(
            coroutine=_dynamic_tool_func,  # 每次执行_dynamic_tool_func的_name都是当前循环下的mcp_tool
            name=mcp_tool.name,
            description=mcp_tool.description or "No description",
            args_schema=InputModel
        )
        langchain_tools.append(lt)
    
    return langchain_tools

# 主流程
async def run_agent_with_mcp():
    # 使用 sys.executable 确保使用当前环境的 Python
    server_params = StdioServerParameters(
        command=sys.executable, 
        # args=["-m", "mcp_server_time"], # -m表示使用本地安装的标准库或第三方库，这里是mcp_server_time
        args = ["D:/Info/WorkSpace/Project/YXYL/FUN/langgraph/mcp/mcp_test_serve.py"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print("Connecting to MCP Server...")
            await session.initialize()
            
            tools = await get_mcp_tools(session)
            print(f" Loaded tools: {[t.name for t in tools]}")

            llm = ChatOpenAI(
                model="qwen-plus", 
                temperature=0,
                api_key="",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            llm_with_tools = llm.bind_tools(tools)

            def chatbot(state: AgentState):
                return {"messages": [llm_with_tools.invoke(state["messages"])]}

            graph_builder = StateGraph(AgentState)
            graph_builder.add_node("chatbot", chatbot)
            graph_builder.add_node("tools", ToolNode(tools=tools))

            graph_builder.add_edge(START, "chatbot")
            
            def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
                messages = state["messages"]
                last_message = messages[-1]
                if last_message.tool_calls:
                    return "tools"
                return "__end__"                                                      

            graph_builder.add_conditional_edges("chatbot", should_continue)
            graph_builder.add_edge("tools", "chatbot")

            graph = graph_builder.compile()

            user_input = "合肥未来三天的天气怎么样？"
            print(f"\n User: {user_input}")
            
            async for event in graph.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"recursion_limit": 30} 
            ):
                for key, value in event.items():
                    if key == "chatbot":
                        msg = value['messages'][0]
                        if msg.content:
                            print(f" Agent: {msg.content}")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(run_agent_with_mcp())
    except Exception as e:
        print(f"\n Program crashed: {e}")