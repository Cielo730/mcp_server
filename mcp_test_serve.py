from mcp.server.fastmcp import FastMCP
import httpx
from tavily import TavilyClient
import os

base_path = Path(__file__).parent
env_path = base_path / '.env'

mcp = FastMCP("Tools")

AMAP_KEY =  os.getenv('AMAP_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@mcp.tool()
def add(a: int, b: int) -> int:
    """两个数相加"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """两个数相乘"""
    return a * b

@mcp.tool()
async def make_name(sex: str) -> str:
    """为男孩或女孩取名字."""
    return "取名为大帅"

@mcp.tool()
async def get_city_adcode(address: str) -> str:
    """
    通过城市名称或地址获取高德行政区划代码 (adcode)。
    在查询天气之前，通常需要先调用此工具获取 adcode。
    
    Args:
        address: 城市名称或具体地址
    """
    url = os.getenv('geocode_url')
    params = {
        "key": AMAP_KEY,
        "address": address
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            data = response.json()
            
            if data["status"] == "1" and data["geocodes"]:
                result = data["geocodes"][0]
                return f"城市: {result['formatted_address']}, Adcode: {result['adcode']}"
            else:
                return f"未找到地址 '{address}' 的编码信息。错误信息: {data.get('info', '未知错误')}"
        except Exception as e:
            return f"请求失败: {str(e)}"

@mcp.tool()
async def get_weather(adcode: str, extensions: str = "base") -> str:
    """
    根据行政区划代码 (adcode) 查询天气。
    
    Args:
        adcode: 城市的 adcode (由 get_city_adcode 工具获取)，例如 "110000"
        extensions: 'base' 返回实况天气 (默认), 'all' 返回预报天气
    """
    url = os.getenv('weather_url')
    params = {
        "key": AMAP_KEY,
        "city": adcode,
        "extensions": extensions
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            data = response.json()
            
            if data["status"] == "1":
                if extensions == "base" and data["lives"]:
                    live = data["lives"][0]
                    return (f"【实况天气】\n"
                            f"省份: {live['province']}\n"
                            f"城市: {live['city']}\n"
                            f"天气: {live['weather']}\n"
                            f"温度: {live['temperature']}℃\n"
                            f"风向: {live['winddirection']}风, 风力: {live['windpower']}级\n"
                            f"湿度: {live['humidity']}%\n"
                            f"发布时间: {live['reporttime']}")
                elif extensions == "all" and data["forecasts"]:
                    forecast = data["forecasts"][0]
                    result_text = f"【天气预报】城市: {forecast['city']}\n"
                    for cast in forecast["casts"]:
                        result_text += (f"- {cast['date']} (周{cast['week']}): "
                                        f"白天{cast['dayweather']} {cast['daytemp']}℃, "
                                        f"夜间{cast['nightweather']} {cast['nighttemp']}℃\n")
                    return result_text
                else:
                    return "未查询到天气数据。"
            else:
                return f"天气查询失败: {data.get('info', '未知错误')}"
        except Exception as e:
            return f"请求失败: {str(e)}"

@mcp.tool()
async def web_search(query: str) -> str:
    """
    使用 Tavily 进行实时网络搜索。
    当用户询问当前事件、新闻等即时消息时请使用此工具。
    """
    response = tavily_client.search(query=query, search_depth="basic")
    return str(response.get("results", []))

if __name__ == "__main__":
    mcp.run(transport="stdio")
