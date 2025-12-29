import os
import sys
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
import asyncio
import nest_asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.tools import BaseTool
from pydantic import PrivateAttr, BaseModel
from typing import Any

base_path = Path(__file__).parent
env_path = base_path / '.env'
load_dotenv(dotenv_path=env_path)

nest_asyncio.apply()

class MCPToolWrapper(BaseTool):
    name: str
    description: str
    _langchain_tool: Any = PrivateAttr() 

    def __init__(self, langchain_tool, **kwargs):
        schema = None
        
        if hasattr(langchain_tool, "get_input_schema"):
            try:
                schema = langchain_tool.get_input_schema()
            except Exception:
                pass
        
        if schema is None:
            raw_schema = getattr(langchain_tool, "args_schema", None)
            if isinstance(raw_schema, type) and issubclass(raw_schema, BaseModel):
                schema = raw_schema
        
        super().__init__(
            name=langchain_tool.name,
            description=langchain_tool.description,
            args_schema=schema, 
            **kwargs
        )
        self._langchain_tool = langchain_tool

    def _run(self, *args, **kwargs):
        print(f"\n[DEBUG] Agent正在调用工具: {self.name}")
        
        try:
            # 参数处理逻辑
            tool_input = {}
            if kwargs:
                tool_input = kwargs
            elif args:
                if isinstance(args[0], dict):
                    tool_input = args[0]
                else:
                    tool_input = str(args[0])

            # print(f"[DEBUG] 最终传给MCP的参数: {tool_input}")

            # 获取当前循环并执行
            loop = asyncio.get_running_loop()
            
            # 如果 args_schema 没设置好，LLM 可能会传错误的参数
            coro = self._langchain_tool.arun(tool_input)
            
            result = loop.run_until_complete(coro)
            
            # 截断过长的输出，防止控制台刷屏
            preview = str(result)[:100].replace('\n', ' ')
            print(f"[DEBUG] 工具返回: {preview}...")
            return result
            
        except Exception as e:
            error_msg = f"工具执行出错: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg

async def main():
    loop = asyncio.get_running_loop()
    nest_asyncio.apply(loop)
    print(f"[System] 已Patch当前事件循环")

    mcp_client = MultiServerMCPClient(
        {
            "tools_server":{
                "transport": "stdio",
                "command": sys.executable,
                "args": [os.getenv('mcp_args')],
            }
        }
    )
    print("正在连接MCP服务器...")
    
    try:
        raw_mcp_tools = await mcp_client.get_tools()
        
        crew_tools = []
        for tool in raw_mcp_tools:
            try:
                wrapped_tool = MCPToolWrapper(langchain_tool=tool)
                crew_tools.append(wrapped_tool)
            except Exception as e:
                print(f"[WARN] 跳过工具 {tool.name}: {e}")
        
        print(f"成功加载工具：{[t.name for t in crew_tools]}")

        # 配置 LLM
        llm = LLM(
            model=os.getenv('qwen_model_name'),
            api_key=os.getenv('qwen_api_key'),
            api_base=os.getenv('qwen_base_url'),
            temperature=0.7
        )

        jounalist = Agent(
            role="资深记者",
            goal="搜集并整理关于 {topic} 的最新事实和数据",
            backstory="你是一名善于利用数字化工具进行深度调查的记者。请务必使用工具。",
            llm=llm,
            tools=crew_tools,
            verbose=True,
            allow_delegation=False
        )

        writer = Agent(
            role="作家",
            goal="负责将收集到的信息整理合并，撰写出高质量的文章",
            backstory="你擅长将枯燥的数据转化为生动的故事。",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

        explore_task = Task(
            description="请使用 web_search 工具搜索关于 '{topic}' 的最新详细信息。",
            agent=jounalist,
            expected_output="一份包含详细事实的新闻素材。", 
        )

        write_task = Task(
            description="根据上一步提供的素材，生成一份完整的新闻报道。标题为:《{title}》",
            agent=writer,
            context=[explore_task], 
            expected_output="一篇格式完整的新闻报道Markdown文本。"
        )

        crew = Crew(
            agents=[jounalist, writer],
            tasks=[explore_task, write_task],
            process=Process.sequential,
            verbose=True
        )

        print("Crew开始工作...")
        result = crew.kickoff(inputs={
            "topic": "洪承畴康熙生父传闻",
            "title": "“野史”突然爆火！康熙生父为洪承畴？"
        })
        
        print("\n\n=== 最终任务结果 ===")
        print(result.raw)
        
        with open("kawasaki_report.md", 'w', encoding='utf-8') as f:
            f.write(result.raw)

    except Exception as e:
        print(f"全局错误: {e}")
        import traceback
        traceback.print_exc()
            
    finally:
        # await mcp_client.close()
        print("程序结束")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())