from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from datetime import datetime

# 导入统一日志系统和分析模块日志装饰器
from tradingagents.utils.logging_init import get_logger
from tradingagents.utils.tool_logging import log_analyst_module
# 导入股票工具类
from tradingagents.utils.stock_utils import StockUtils

logger = get_logger("analysts.sector")

def create_sector_analyst(llm, toolkit):
    @log_analyst_module("sector")
    def sector_analyst_node(state):
        start_time = datetime.now()

        # 🔧 工具调用计数器
        tool_call_count = state.get("sector_tool_call_count", 0)
        max_tool_calls = 3

        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        logger.info(f"[行业分析师] 开始分析 {ticker} 所属行业及竞争格局")
        
        # 行业分析师使用市场数据工具来对比同行业表现
        tools = [toolkit.get_stock_market_data_unified]
        
        system_message = (
            """您是一位专业的行业分析师，负责分析目标公司在其所属行业中的竞争地位及行业整体趋势。

您的主要职责包括：
1. 识别目标公司所属的核心行业和细分赛道。
2. 对比目标公司与行业内主要竞争对手的表现（股价走势、估值水平）。
3. 分析行业周期性、市场饱和度及增长潜力。
4. 评估行业内的技术变革或商业模式创新。
5. 判断目标公司是否具备竞争优势（护城河）。

分析要点：
- 行业整体估值是偏高还是偏低？
- 目标公司是行业领跑者、追随者还是落后者？
- 行业近期的资金流向和关注度。
- 关键竞争对手的动态对目标公司的影响。

📝 输出格式要求：
## 🏗️ 行业概况与趋势
[描述行业现状及未来走向]

## ⚔️ 竞争格局分析
[对比主要竞争对手，分析公司地位]

## 📊 行业估值对比
[对比行业平均 PE/PB 等指标]

## 💎 竞争优势评估
[分析公司的核心竞争力]

## 🏁 行业面结论
[给出行业层面的投资评价：超配/中性/低配]

请使用中文撰写报告。"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一位专业的行业分析师。"
                    "\n1. 您的第一个动作必须是调用工具获取行业及竞争对手的数据。"
                    "\n2. 只有在获取数据后，才能开始分析。"
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
        
        if (hasattr(last_message, 'content') and "ToolMessage" in str(type(last_message))) or tool_call_count >= max_tool_calls:
            logger.info(f"[行业分析师] 正在生成最终报告...")
            result = chain.invoke({"messages": state["messages"]})
            return {
                "messages": [result],
                "sector_report": result.content,
                "sector_tool_call_count": tool_call_count
            }
        
        result = chain.invoke({"messages": state["messages"]})
        
        new_tool_call_count = tool_call_count
        if hasattr(result, 'tool_calls') and result.tool_calls:
            new_tool_call_count += 1
            
        return {
            "messages": [result],
            "sector_tool_call_count": new_tool_call_count
        }

    return sector_analyst_node
