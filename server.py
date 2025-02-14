import asyncio
import websockets
import json
from smolagents import CodeAgent, Tool, LiteLLMModel
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import os
import logging
import requests
from bs4 import BeautifulSoup
from functools import partial
from dotenv import load_dotenv

# 加載環境變量
load_dotenv()

# 設置日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# HTTP 服務器
def run_http_server():
    # 切換到腳本所在目錄
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
    print(f"HTTP 服務器已啟動於 http://localhost:8000")
    print(f"當前工作目錄: {os.getcwd()}")
    httpd.serve_forever()

class GoogleSearchTool(Tool):
    name = "google_search"
    description = "使用 Google 搜索引擎查找信息，支持高級搜索語法"
    inputs = {
        "query": {
            "type": "string",
            "description": "要搜索的查詢內容"
        }
    }
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.cx = os.getenv('GOOGLE_CX')
        self.agent = None  # 將在 set_agent 中設置

    def set_agent(self, agent):
        self.agent = agent

    def forward(self, query: str) -> str:
        try:
            if not self.agent:
                logger.error("Agent not set for GoogleSearchTool")
                return "搜索工具未正確初始化"

            base_url = "https://www.googleapis.com/customsearch/v1"
            all_results = []
            
            # 直接執行搜索，不使用 CodeAgent 優化
            params = {
                'key': self.api_key,
                'cx': self.cx,
                'q': query,
                'num': 10,
                'start': 1,
                'safe': 'off',
                'fields': 'items(title,link,snippet)',
                'hl': 'zh-TW',
                'gl': 'tw'
            }
            
            response = requests.get(base_url, params=params)
            results = response.json()
            
            if 'items' in results:
                all_results.extend(results['items'])

            if not all_results:
                return "未找到相關結果"

            # 格式化結果
            formatted_results = "\n\n".join([
                f"標題：{item['title']}\n"
                f"連結：{item['link']}\n"
                f"摘要：{item.get('snippet', '無摘要')}"
                for item in all_results
            ])
            
            return formatted_results

        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            return f"搜索過程中發生錯誤：{str(e)}"

# WebSocket 服務器
class ResearchServer:
    def __init__(self):
        try:
            self.model = LiteLLMModel(
                "o3-mini",
                api_key=os.getenv('OPENAI_API_KEY')
            )
            search_tool = GoogleSearchTool()
            self.agent = CodeAgent(tools=[search_tool], model=self.model)
            
            # 設置 tool 的 agent
            search_tool.set_agent(self.agent)
            
            logger.debug(f"Created search tool: {search_tool}")
            logger.debug(f"Agent tools: {self.agent.tools}")
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            raise

    async def send_update(self, websocket, type, content):
        try:
            await websocket.send(json.dumps({
                "type": type,
                "content": content
            }))
            logger.debug(f"Sent update: {type}")
        except Exception as e:
            logger.error(f"Error sending update: {str(e)}")
            raise

    async def handle_research(self, websocket, path):
        try:
            async for message in websocket:
                logger.info("Received new message")
                query = json.loads(message)["query"]
                logger.info(f"Processing query: {query}")
                
                # 檢查工具是否存在
                if 'google_search' not in self.agent.tools:
                    logger.error("Google search tool not available")
                    await self.send_update(websocket, "error", "系統錯誤：搜索工具未初始化")
                    return
                
                await self.send_update(websocket, "status", "開始研究流程")
                
                try:
                    # 1. 分析查詢意圖
                    logger.info("Analyzing query intent")
                    await self.send_update(websocket, "step", "分析查詢意圖...")
                    intent_prompt = f"""
請分析以下查詢的主要意圖和關鍵要素，並以純文本格式返回：

查詢：{query}

請按以下格式輸出：
主要主題：
關鍵詞：
時間範圍：
地理位置：
特定領域：
查詢目的：
"""
                    intent_analysis = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        partial(self.agent.run, intent_prompt)
                    )
                    await self.send_update(websocket, "intent_analysis", intent_analysis)

                    # 2. 執行網路搜索
                    logger.info("Starting web search")
                    await self.send_update(websocket, "step", "執行網路搜索...")
                    search_tool = self.agent.tools['google_search']
                    search_results = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        partial(search_tool.forward, query)
                    )
                    await self.send_update(websocket, "search_results", search_results)
                    
                    # 3. 初步資料分類
                    logger.info("Classifying results")
                    await self.send_update(websocket, "step", "對搜索結果進行分類...")
                    classification_prompt = f"""
請將以下搜索結果分類並以純文本格式輸出：

{search_results}

請按以下格式分類：

品牌活動：
- [活動名稱] - [時間] - [簡短描述]

行銷活動：
- [活動名稱] - [時間] - [簡短描述]

企業合作：
- [合作項目] - [時間] - [簡短描述]

社會責任：
- [專案名稱] - [時間] - [簡短描述]

其他活動：
- [活動名稱] - [時間] - [簡短描述]
"""
                    classification = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        partial(self.agent.run, classification_prompt)
                    )
                    await self.send_update(websocket, "classification", classification)

                    # 4. 深入分析
                    logger.info("Starting detailed analysis")
                    await self.send_update(websocket, "step", "進行深入分析...")
                    analysis_prompt = f"""
請對 PUMA 在台灣的公關活動進行深入分析，並以純文本格式輸出：

時間軸分析：
[按時間順序列出重要活動]

主要策略方向：
1. [策略1]
2. [策略2]
...

目標受眾：
1. [受眾群體1]
2. [受眾群體2]
...

活動特點：
1. [特點1]
2. [特點2]
...

效果評估：
1. [成效1]
2. [成效2]
...
"""
                    analysis = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        partial(self.agent.run, analysis_prompt)
                    )
                    
                    # 5. 生成最終見解
                    logger.info("Generating final insights")
                    await self.send_update(websocket, "step", "生成研究見解...")
                    insights_prompt = f"""
基於以上分析，請提供具體的見解和建議，以純文本格式輸出：

關鍵發現：
1. [發現1]
2. [發現2]
3. [發現3]

發展趨勢：
1. [趨勢1]
2. [趨勢2]

建議方向：
1. [建議1]
2. [建議2]

需注意風險：
1. [風險1]
2. [風險2]

未來展望：
[簡要說明未來可能的發展方向]
"""
                    insights = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        partial(self.agent.run, insights_prompt)
                    )
                    
                    # 發送最終結果
                    await self.send_update(websocket, "final_result", {
                        "intent_analysis": intent_analysis,
                        "classification": classification,
                        "analysis": analysis,
                        "insights": insights
                    })
                    
                    logger.info("Research completed successfully")
                    
                except Exception as e:
                    logger.error(f"Error during research: {str(e)}", exc_info=True)
                    await self.send_update(websocket, "error", f"研究過程發生錯誤：{str(e)}")
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            try:
                await self.send_update(websocket, "error", f"系統錯誤：{str(e)}")
            except:
                pass

async def main():
    try:
        # 啟動 HTTP 服務器
        http_thread = threading.Thread(target=run_http_server)
        http_thread.daemon = True
        http_thread.start()
        logger.info("HTTP server started")

        # 啟動 WebSocket 服務器
        server = ResearchServer()
        async with websockets.serve(
            server.handle_research, 
            "localhost", 
            8765,
            ping_interval=None  # 禁用自動 ping
        ):
            logger.info("WebSocket server started")
            print("WebSocket 服務器已啟動於 ws://localhost:8765")
            await asyncio.Future()
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}") 