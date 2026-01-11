from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from datetime import datetime

# 导入统一日志系统和分析模块日志装饰器
from tradingagents.utils.logging_init import get_logger
from tradingagents.utils.tool_logging import log_analyst_module
# 导入股票工具类
from tradingagents.utils.stock_utils import StockUtils
# 导入Google工具调用处理器
from tradingagents.agents.utils.google_tool_handler import GoogleToolCallHandler

logger = get_logger("analysts.macro")

def create_macro_analyst(llm, toolkit):
    @log_analyst_module("macro")
    def macro_analyst_node(state):
        start_time = datetime.now()

        # 🔧 工具调用计数器 - 防止无限循环
        tool_call_count = state.get("macro_tool_call_count", 0)
        max_tool_calls = 3  # 最大工具调用次数
        logger.info(f"🔧 [宏观分析师] 当前工具调用次数: {tool_call_count}/{max_tool_calls}")

        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        logger.info(f"[宏观分析师] 开始分析 {ticker} 的宏观背景，交易日期: {current_date}")
        
        # 获取市场信息
        market_info = StockUtils.get_market_info(ticker)
        
        # 宏观分析师主要使用搜索工具或通用的宏观数据接口
        # 这里我们复用 toolkit 中的搜索工具或新闻工具来获取宏观信息
        tools = [toolkit.get_stock_news_unified] # 暂时复用新闻工具，实际可扩展专门的宏观工具
        
        system_message = (
            """您是一位专业的宏观经济分析师，负责分析全球及区域宏观经济环境对特定行业和公司的影响。

您的主要职责包括：
1. 分析当前的货币政策（利率、通胀、流动性）对市场的影响。
2. 评估财政政策、行业补贴或监管变动对目标公司的潜在冲击。
3. 识别地缘政治风险或重大宏观事件（如大选、贸易冲突）。
4. 分析大宗商品价格、汇率波动对公司成本或收入的影响。
5. 提供宏观层面的风险预警和投资基调建议。

分析要点：
- 宏观环境是利好、中性还是利空？
- 当前所处的经济周期阶段。
- 政策导向对该股票所属板块的支持力度。
- 外部宏观冲击的概率及潜在损失。

📝 输出格式要求：
## 🌍 宏观经济环境分析
[分析当前的宏观大背景]

## 🏛️ 政策与监管动态
[分析相关政策对公司的影响]

## 💰 货币与市场流动性
[分析利率、汇率等因素]

## ⚠️ 宏观风险提示
[列出主要的宏观不确定性]

## 💡 宏观投资基调
[给出明确的宏观面建议：积极/谨慎/避险]

请使用中文撰写报告。"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一位专业的宏观经济分析师。"
                    "\n1. 您的第一个动作必须是调用工具获取最新的宏观或行业相关信息。"
                    "\n2. 只有在获取数据后，才能开始分析。"
                    "\n3. 您的回答必须基于真实数据和逻辑推理。"
                    "\n您可以访问以下工具：{tool_names}。"
                    "\n{system_message}"
                    "\n当前日期是{current_date}。目标公司是{ticker}。"
                    "\n请用中文撰写所有分析内容。",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name if hasattr(tool, 'name') else tool.__name__ for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)
        
        chain = prompt | llm.bind_tools(tools)
        
        # 检查是否已经有工具结果
        last_message = state["messages"][-1] if state["messages"] else None
        
        # 如果最后一条消息是工具消息，或者已经有报告，则生成最终报告
        if (hasattr(last_message, 'content') and "ToolMessage" in str(type(last_message))) or tool_call_count >= max_tool_calls:
            logger.info(f"[宏观分析师] 正在生成最终报告...")
            result = chain.invoke({"messages": state["messages"]})
            
            # 如果 LLM 还是想调用工具但已经达到上限，强制它生成内容
            if hasattr(result, 'tool_calls') and result.tool_calls and tool_call_count >= max_tool_calls:
                 logger.warning(f"[宏观分析师] 达到工具调用上限，强制生成文本内容")
                 # 这里可以再次调用 LLM 并不带工具，或者简单处理
            
            return {
                "messages": [result],
                "macro_report": result.content,
                "macro_tool_call_count": tool_call_count
            }
        
        # 否则，调用 LLM（它可能会选择调用工具）
        result = chain.invoke({"messages": state["messages"]})
        
        new_tool_call_count = tool_call_count
        if hasattr(result, 'tool_calls') and result.tool_calls:
            new_tool_call_count += 1
            
        return {
            "messages": [result],
            "macro_tool_call_count": new_tool_call_count
        }

    return macro_analyst_node
