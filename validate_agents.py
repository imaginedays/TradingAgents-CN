import os
import sys
from datetime import datetime

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.agents.utils.agent_states import AgentState
from langchain_openai import ChatOpenAI
from tradingagents.agents.utils.agent_utils import Toolkit
from tradingagents.utils.logging_init import init_logging, get_logger

# 初始化日志系统
init_logging()
logger = get_logger("validation")

def run_validation():
    logger.info("🚀 开始验证 TradingAgentsGraph 的新 Agent 集成...")

    # 模拟 LLM 实例
    # 实际使用时，这里会是配置好的真实 LLM
    # 为了验证图结构和Agent调用，我们可以使用一个模拟的LLM或者一个非常简单的LLM
    # 这里使用一个占位符，因为我们主要验证图的构建和状态更新，而不是LLM的实际输出
    # 如果需要更真实的测试，可以配置一个真正的OpenAI或DashScope LLM
    try:
        # 尝试使用环境变量中的API Key初始化LLM
        quick_thinking_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, openai_api_base=os.getenv("OPENAI_API_BASE"), openai_api_key=os.getenv("OPENAI_API_KEY"))
        deep_thinking_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7, openai_api_base=os.getenv("OPENAI_API_BASE"), openai_api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("✅ 成功初始化 ChatOpenAI LLM 实例")
    except Exception as e:
        logger.warning(f"⚠️ 初始化 ChatOpenAI 失败: {e}。将使用模拟LLM进行验证。")
        # 如果无法初始化，则使用一个简单的模拟LLM
        class MockLLM:
            def invoke(self, prompt):
                logger.info(f"[MockLLM] Invoked with prompt: {prompt[:100]}...")
                return "Mocked LLM response."
            def bind_tools(self, tools):
                logger.info(f"[MockLLM] Bound tools: {[t.name for t in tools]}")
                return self
        quick_thinking_llm = MockLLM()
        deep_thinking_llm = MockLLM()

    # 模拟 Toolkit 实例
    # 实际使用时，Toolkit 会包含各种真实工具
    # 这里我们只模拟必要的工具，以避免实际的网络请求
    class MockStockUtils:
        @staticmethod
        def get_market_info(ticker):
            return {
                "is_china": True, "is_hk": False, "is_us": False,
                "market_name": "中国A股", "currency_name": "人民币", "currency_symbol": "¥"
            }

    class MockToolkit(Toolkit):
        def __init__(self):
            super().__init__()
            # 覆盖实际工具，使用模拟工具
            self.get_stock_news_unified = self._mock_tool("get_stock_news_unified", "模拟新闻数据")
            self.get_stock_market_data_unified = self._mock_tool("get_stock_market_data_unified", "模拟市场数据")
            self.get_stock_fundamentals_unified = self._mock_tool("get_stock_fundamentals_unified", "模拟基本面数据")
            self.get_social_media_sentiment_unified = self._mock_tool("get_social_media_sentiment_unified", "模拟社交媒体情绪")

        def _mock_tool(self, name, result_content):
            def tool_func(ticker: str, date: str = None, start_date: str = None, end_date: str = None):
                logger.info(f"[MockTool] Calling {name} for {ticker} with date {date} / {start_date}-{end_date}")
                return f"这是 {ticker} 的{result_content}。"
            tool_func.__name__ = name # LangChain需要工具函数有__name__属性
            return tool_func

    toolkit = MockToolkit()

    # 模拟配置
    config = {
        "llm_provider": "mock", # 模拟LLM提供商
        "quick_think_llm": "mock-quick", # 模拟模型名称
        "deep_think_llm": "mock-deep", # 模拟模型名称
        "selected_analysts": ["macro", "sector", "market", "social", "news", "fundamentals"],
        "debug": True,
        "project_dir": "/home/ubuntu/TradingAgents-CN",
        "default_base_url": "http://dummy-base-url.com" # 模拟基础URL
    }

    # 初始化 TradingAgentsGraph
    try:
        trading_graph = TradingAgentsGraph(
            selected_analysts=config["selected_analysts"],
            debug=config["debug"],
            config=config
        )
        app = trading_graph.get_graph()
        logger.info("✅ TradingAgentsGraph 成功初始化并编译。")
    except Exception as e:
        logger.error(f"❌ TradingAgentsGraph 初始化或编译失败: {e}")
        return False

    # 模拟初始状态
    initial_state = AgentState(
        company_of_interest="000001.SZ",
        trade_date=datetime.now().strftime("%Y-%m-%d"),
        messages=[],
        market_report="",
        sentiment_report="",
        news_report="",
        fundamentals_report="",
        macro_report="", # 新增字段
        sector_report="", # 新增字段
        macro_tool_call_count=0,
        sector_tool_call_count=0,
        market_tool_call_count=0,
        social_tool_call_count=0,
        news_tool_call_count=0,
        fundamentals_tool_call_count=0,
        investment_debate_state={
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "count": 0,
            "judge_decision": ""
        },
        risk_analysis_state={
            "history": "",
            "risky_history": "",
            "neutral_history": "",
            "safe_history": "",
            "current_response": "",
            "count": 0,
            "judge_decision": ""
        },
        investment_plan="",
        risk_assessment=""
    )

    # 运行图
    logger.info("🏃‍♂️ 模拟运行 TradingAgentsGraph...")
    try:
        # 模拟运行几个步骤，确保宏观和行业分析师被调用并更新状态
        # 由于是模拟LLM和工具，我们不能期望完整的报告，但可以检查状态字段是否被触及
        for _ in range(5): # 运行5步，足以触发分析师调用
            initial_state = app.invoke(initial_state)
            logger.info(f"🔄 Graph 运行一步，当前状态: {initial_state}")
            if initial_state.get("macro_report") and initial_state.get("sector_report"):
                logger.info("✅ 宏观和行业报告已在状态中更新。")
                break

        # 检查最终状态
        if initial_state.get("macro_report") and len(initial_state["macro_report"]) > 0:
            macro_report_preview = initial_state["macro_report"][:100]
            logger.info(f"✅ 宏观分析报告已生成: {macro_report_preview}...")
        else:
            logger.error("❌ 宏观分析报告未生成或为空。")
            return False

        if initial_state.get("sector_report") and len(initial_state["sector_report"]) > 0:
            sector_report_preview = initial_state["sector_report"][:100]
            logger.info(f"✅ 行业分析报告已生成: {sector_report_preview}...")
        else:
            logger.error("❌ 行业分析报告未生成或为空。")
            return False

        logger.info("🎉 验证成功：新 Agent 已成功集成并更新状态！")
        return True

    except Exception as e:
        logger.error(f"❌ 模拟运行 TradingAgentsGraph 失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if run_validation():
        logger.info("所有验证通过！项目可执行。")
    else:
        logger.error("验证失败！请检查日志以获取详细信息。")
        sys.exit(1)
