"""
Entry point for running the analyzer as a module:
    python -m sentiment_analysis
"""

import asyncio
import sys

from sentiment_analysis.orchestration.analyzer import analyze_document


SAMPLE_DOC = """
Q3 Earnings Report — FY2024

Introduction: The quarter began under significant headwinds. Revenue declined by 8%
year-over-year amid challenging macroeconomic conditions, and we have been rightsizing
our workforce to better align costs with current demand levels.

Product Performance: Product A received outstanding reviews from enterprise clients,
with NPS scores reaching an all-time high of 72. Our engineering team has delivered
a record number of features, and customer adoption is accelerating.

Mid-Year Challenges: Oh great — just as we were gaining momentum, supply-chain
disruptions rattled our hardware division. Product A's margins compressed sharply,
and frankly the results were, shall we say, less than inspiring.

Strategic Realignment: We are rationalizing our go-to-market approach and creating
synergies across the enterprise and SMB units. By leveraging our ecosystem and
moving the needle on mission-critical initiatives, we aim to unlock significant value.

Outlook: Going forward, we expect robust tailwinds from our AI product line.
In the next two quarters, we project revenue growth of 15–20%. The ball is in our
court to capitalize on these opportunities, and we are confident in our ability
to deliver exceptional results. Product A, despite earlier challenges, is now
positioned as a market leader with an outstanding roadmap ahead.
"""


async def _main():
    result = await analyze_document(
        SAMPLE_DOC,
        filename="sample_earnings.txt",
        use_cache=False,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
