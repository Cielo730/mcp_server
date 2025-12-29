from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import asyncio
import sys
import os
from dotenv import load_dotenv

base_path = Path(__file__).parent
env_path = base_path / '.env'
load_dotenv(dotenv_path=env_path)

async def main():
    # 初始化 MCP 客户端
    client = MultiServerMCPClient(
        {
            "tools_server": {
                "transport": "stdio",
                "command": sys.executable, 
                "args": [os.getenv('mcp_args')],
            },
        }
    )

    print("🔌 正在连接 MCP 服务器...")
    try:
        # 获取工具 自动启动子进程并保持连接
        tools = await client.get_tools()
        print(f"成功加载工具: {[t.name for t in tools]}")

        llm = ChatOpenAI(
            model=os.getenv('qwen_model_name'),
            temperature=0,
            api_key=os.getenv('qwen_api_key'), 
            base_url=os.getenv('qwen_base_url')
        )

        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", "遇到任何问题必须先调用工具来获取结果，如果没有合适的工具则自己解决。"),
        #     ("user", "{input}"),
        #     ("placeholder", "{agent_scratchpad}"),
        # ])

        # 创建 Agent，checkpointer=None 表示不持久化记忆，仅本次运行有效
        agent = create_agent(llm, tools)

        print("Agent 正在思考...")
        
        # response = await agent.ainvoke(
        #     {"messages": [("user", "南京未来三天的天气怎么样")]}
        # )

        # final_response = response["messages"][-1].content
        # print(f"\n最终结果: {final_response}")
        async for event in agent.astream(
            {"messages": [("user", "2026春晚分会场")]},
            stream_mode="values"
        ):
            message = event["messages"][-1]
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"正在调用工具: {tool_call['name']}")
                    # print(f" [参数]: {tool_call['args']}")
            # elif message.type == "tool":
            #     content_preview = message.content[:100] + "..." if len(message.content) > 100 else message.content
            #     print(f"[工具返回结果]: {content_preview}")
            # elif message.type == "ai" and not message.tool_calls:
            #     pass
        if message.type == "ai":
             print(f"结果如下:\n{message.content}")

    except Exception as e:
        print(f"运行出错: {e}")
    

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())