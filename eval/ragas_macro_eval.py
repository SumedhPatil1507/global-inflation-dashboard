"""Run RAGAS Faithfulness, Context Precision, and Answer Relevancy on 50 macro cases.

Usage (from repository root)::

    pip install -r eval/requirements.txt
    set OPENAI_API_KEY=...                 # PowerShell: $env:OPENAI_API_KEY="..."
    python eval/ragas_macro_eval.py

The script exercises the production macro retriever and memorandum synthesizer,
then writes per-case scores, errors, aggregate means, and run metadata to JSON.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


TOPIC_CASES = [
    {
        "country": "IND", "topic": "India food inflation",
        "reference": "India's CPI has been volatile because of recurring vegetable and cereal price spikes; the RBI highlights food inflation risks to expectations and its 4 percent target.",
        "questions": [
            "Why have vegetable and cereal prices made India's headline CPI volatile?",
            "How do food price shocks affect India's inflation outlook and expectations?",
            "What does the RBI identify as the main source of recent CPI volatility in India?",
            "Explain the role of tomato, onion, potato, and cereal prices in Indian inflation.",
            "What inflation risk does persistent food inflation pose for the RBI's 4 percent target?",
        ],
    },
    {
        "country": "IND", "topic": "RBI policy stance",
        "reference": "The RBI MPC kept the repo rate unchanged at 6.50 percent, with a focus on withdrawal of accommodation; geopolitical and climate-related shocks remained upside risks to headline inflation.",
        "questions": [
            "What repo rate did the RBI MPC maintain, and what was its policy stance?",
            "Summarize the RBI's recent rate decision and the risks it cited for inflation.",
            "Did the RBI change its repo rate, and how did it characterize monetary policy?",
            "What did the RBI MPC decide about the repo rate amid food and geopolitical risks?",
            "Which inflation risks were cited alongside the RBI's policy rate pause?",
        ],
    },
    {
        "country": "IND", "topic": "Indian rate transmission",
        "reference": "During the RBI tightening cycle, weighted average lending rates rose 193 basis points on fresh rupee loans and 112 basis points on outstanding loans; credit growth remained robust at 15.5 percent year over year.",
        "questions": [
            "How much did Indian bank lending rates rise during monetary transmission?",
            "Compare the RBI bulletin's lending-rate changes for fresh and outstanding loans.",
            "What happened to WALR and credit growth during India's tightening cycle?",
            "How did policy tightening pass through to rupee loan rates?",
            "What lending rate and credit growth figures did the RBI report?",
        ],
    },
    {
        "country": "USA", "topic": "Global disinflation outlook",
        "reference": "The World Bank projected global headline inflation to decline from 5.8 percent in 2023 to 3.7 percent in 2024 and 3.4 percent in 2025, while noting persistent services and shelter pressures.",
        "questions": [
            "What global inflation path did the World Bank project for 2023 through 2025?",
            "How quickly was global headline inflation expected to fall in the World Bank outlook?",
            "What were the World Bank's global CPI projections and remaining sticky-price risks?",
            "Summarize the projected disinflation path and its drivers in the World Bank outlook.",
            "What did the World Bank say about headline and core inflation through 2025?",
        ],
    },
    {
        "country": "USA", "topic": "Oil supply and energy costs",
        "reference": "Brent averaged 85 dollars per barrel in Q1 2024 as OPEC+ cuts and Middle East risks affected supply; maritime rerouting added shipping premiums, while agricultural and fertilizer prices fell 9 percent year over year.",
        "questions": [
            "What factors influenced Brent oil prices in early 2024?",
            "How did OPEC+ production cuts and Middle East risks affect energy markets?",
            "What oil price and shipping premium figures did the World Bank report?",
            "How did maritime rerouting affect crude oil transport costs?",
            "What happened to agricultural and fertilizer prices as energy prices spiked?",
        ],
    },
    {
        "country": "IND", "topic": "Food weight across economies",
        "reference": "World Bank research found food represents 35 to 50 percent of CPI baskets in developing economies versus 10 to 15 percent in advanced economies; climate shocks can therefore have larger headline effects in developing regions.",
        "questions": [
            "How does food's CPI basket weight differ between developing and advanced economies?",
            "Why can climate shocks have an outsized inflation impact in South Asia?",
            "What food-weight ranges did World Bank research report for CPI baskets?",
            "How do agricultural supply shocks affect emerging market inflation differently?",
            "Explain why food inflation matters more for CPI in developing economies.",
        ],
    },
    {
        "country": "USA", "topic": "Federal Reserve policy stance",
        "reference": "The FOMC maintained the federal funds target range at 5.25 to 5.50 percent and said rate cuts require greater confidence that inflation is moving sustainably toward the 2 percent target.",
        "questions": [
            "What federal funds target range did the FOMC maintain?",
            "What conditions did the Federal Reserve say were needed before rate cuts?",
            "How did the FOMC characterize progress toward its 2 percent inflation goal?",
            "Summarize the Fed's rate stance and its stated threshold for easing.",
            "What did the Federal Reserve say about inflation confidence and future rate cuts?",
        ],
    },
    {
        "country": "USA", "topic": "US shelter inflation",
        "reference": "US shelter CPI rose 5.5 percent over 12 months and accounted for over two-thirds of the increase in core CPI; energy prices rebounded 2.6 percent over the quarter.",
        "questions": [
            "How much did shelter inflation contribute to US core CPI growth?",
            "What did the BLS report about shelter and energy price changes?",
            "Why was shelter an important driver of US core inflation?",
            "Give the reported year-over-year shelter CPI increase and its contribution to core CPI.",
            "What were the recent shelter, energy, and grocery inflation readings in the US?",
        ],
    },
    {
        "country": "USA", "topic": "Freight and supply chains",
        "reference": "Freight rates on Asia-to-Europe and US East Coast lanes rose over 140 percent after vessels were diverted around the Cape of Good Hope; broad inflation pass-through remained muted.",
        "questions": [
            "How much did container freight rates jump after Red Sea diversions?",
            "What happened to Asia-Europe shipping rates during the logistics disruption?",
            "Did the freight shock cause broad inflationary pass-through?",
            "How did Cape of Good Hope rerouting affect freight rates and supplier times?",
            "What does the macro dispatch report say about freight inflation risks?",
        ],
    },
    {
        "country": "IND", "topic": "Commodity cost buffers",
        "reference": "Despite energy price spikes, agricultural and fertilizer prices fell 9 percent year over year, buffering food-importing economies from severe import-cost inflation.",
        "questions": [
            "How did fertilizer prices affect food importers during the oil price spike?",
            "What commodity price changes helped buffer food-importing economies?",
            "Did agricultural input costs rise or fall year over year in the World Bank report?",
            "How did lower fertilizer and agricultural prices influence import inflation?",
            "What offsetting commodity trends did the Commodity Markets Outlook describe?",
        ],
    },
]


def build_scenarios() -> list[dict[str, str]]:
    scenarios = []
    for case in TOPIC_CASES:
        for index, question in enumerate(case["questions"], start=1):
            scenarios.append({
                "scenario_id": f"{case['topic'].lower().replace(' ', '-')}-{index:02d}",
                "topic": case["topic"], "country": case["country"],
                "question": question, "reference": case["reference"],
            })
    if len(scenarios) != 50:
        raise ValueError(f"Expected exactly 50 evaluation cases, found {len(scenarios)}")
    return scenarios


def run_macro_case(scenario: dict[str, str]) -> tuple[str, list[str], dict[str, Any]]:
    """Exercise the production retriever and synthesizer without external data calls."""
    from backend.services.macro_rag import retrieve_macro_reports
    from backend.services.macro_agent import synthesis_node

    documents = retrieve_macro_reports(
        scenario["question"], country=scenario["country"], top_k=3,
    )
    state = {
        "query": scenario["question"], "country": scenario["country"],
        "time_series_data": {
            "latest_inflation": 3.2, "previous_inflation": 3.1,
            "yoy_inflation_delta": 0.1, "policy_rate": 5.25,
            "real_interest_rate": 2.05, "gdp_growth": 2.4,
            "data_source": "RAGAS macro evaluation fixture",
        },
        "rag_documents": documents, "backtest_result": None,
        "execution_trace": [],
    }
    synthesis = synthesis_node(state)
    context = [f"{doc['title']} ({doc['citation']}): {doc['content']}" for doc in documents]
    return synthesis["synthesis"], context, synthesis.get("validation", {})


async def evaluate_scenarios(scenarios: list[dict[str, str]]) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY for the RAGAS judge model.")

    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

    model = os.getenv("RAGAS_LLM_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
    client = AsyncOpenAI()
    judge_llm = llm_factory(model, client=client)
    judge_embeddings = embedding_factory("openai", model=embedding_model, client=client)
    metrics = {
        "faithfulness": Faithfulness(llm=judge_llm),
        "context_precision": ContextPrecision(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
    }
    semaphore = asyncio.Semaphore(int(os.getenv("RAGAS_MAX_CONCURRENCY", "3")))

    async def score_case(scenario: dict[str, str]) -> dict[str, Any]:
        try:
            answer, contexts, validation = await asyncio.to_thread(run_macro_case, scenario)
        except Exception as exc:
            return {**scenario, "answer": "", "retrieved_contexts": [], "validation": {},
                    "scores": {name: None for name in metrics}, "error": f"pipeline: {exc}"}

        scores: dict[str, float | None] = {}
        errors = []
        calls = {
            "faithfulness": {"user_input": scenario["question"], "response": answer,
                             "retrieved_contexts": contexts or ["No source documents retrieved."]},
            "context_precision": {"user_input": scenario["question"], "reference": scenario["reference"],
                                  "retrieved_contexts": contexts or ["No source documents retrieved."]},
            "answer_relevancy": {"user_input": scenario["question"], "response": answer},
        }
        async with semaphore:
            for name, metric in metrics.items():
                try:
                    result = await metric.ascore(**calls[name])
                    scores[name] = float(result.value)
                except Exception as exc:
                    scores[name] = None
                    errors.append(f"{name}: {exc}")
        row = {**scenario, "answer": answer, "retrieved_contexts": contexts,
               "validation": validation, "scores": scores}
        if errors:
            row["error"] = "; ".join(errors)
        return row

    results = await asyncio.gather(*(score_case(scenario) for scenario in scenarios))
    means = {}
    for name in metrics:
        values = [row["scores"][name] for row in results if row["scores"].get(name) is not None]
        means[name] = sum(values) / len(values) if values else None
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "RAGAS macro research quality audit",
        "scenario_count": len(results),
        "models": {"llm": model, "embeddings": embedding_model},
        "metric_means": means,
        "successful_metric_counts": {
            name: sum(row["scores"].get(name) is not None for row in results) for name in metrics
        },
        "scenarios": results,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON report path (default: eval/reports/macro_ragas_audit_<UTC timestamp>.json)")
    args = parser.parse_args()
    report = await evaluate_scenarios(build_scenarios())
    output = args.output or Path("eval/reports") / (
        "macro_ragas_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    temp_output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp_output.replace(output)
    print(json.dumps({"report": str(output), "scenario_count": report["scenario_count"],
                      "metric_means": report["metric_means"]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
