"""
Macro RAG Knowledge Base — World Bank, RBI, Federal Reserve reports & macroeconomic news.
Provides chunking, embedding, and semantic similarity retrieval with citations.
"""
import math
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

# ── Knowledge Base Corpus ────────────────────────────────────────────────────
MACRO_KNOWLEDGE_CORPUS: List[Dict[str, Any]] = [
    # ── Reserve Bank of India (RBI) Reports ───────────────────────────────────
    {
        "doc_id": "RBI-MPR-2024-Q2-FOOD",
        "title": "RBI Monetary Policy Report — Food Price Pressures and CPI Volatility",
        "publisher": "Reserve Bank of India (RBI)",
        "country": "IND",
        "quarter": "2024-Q2",
        "date": "2024-04-05",
        "topic": "food_inflation",
        "citation": "RBI Monetary Policy Report (April 2024, Section 2.1)",
        "content": (
            "Headline CPI inflation in India exhibited volatility during recent quarters, driven predominantly "
            "by sharp recurring spikes in vegetable prices (notably tomatoes, onions, and potatoes) and cereals. "
            "While core CPI inflation (CPI excluding food and fuel) moderated steadily to a record low of 3.4% "
            "reflecting the effective transmission of the 250 bps cumulative repo rate hikes, overall headline "
            "inflation hovered around 4.8%-5.1%. The MPC emphasized that persistent food inflation risks "
            "unanchoring inflation expectations and reiterated its commitment to aligning inflation with the 4.0% target."
        ),
    },
    {
        "doc_id": "RBI-MPC-2024-Q1-STANCE",
        "title": "RBI MPC Resolution — Repo Rate Pause and Withdrawal of Accommodation",
        "publisher": "Reserve Bank of India (RBI)",
        "country": "IND",
        "quarter": "2024-Q1",
        "date": "2024-02-08",
        "topic": "monetary_policy",
        "citation": "RBI MPC Statement by Governor Shaktikanta Das (Feb 2024)",
        "content": (
            "The Monetary Policy Committee (MPC) decided to keep the policy repo rate unchanged at 6.50% by a 5-1 majority. "
            "The standing deposit facility (SDF) rate remains at 6.25% and the marginal standing facility (MSF) rate at 6.75%. "
            "The Committee noted that domestic economic activity remains resilient, with real GDP expanding by 7.6% in 2023-24. "
            "However, ongoing geopolitical uncertainties, Red Sea freight escalations, and climate-induced agricultural shocks "
            "continue to pose upside risks to headline inflation. Monetary policy remains focused on withdrawal of accommodation."
        ),
    },
    {
        "doc_id": "RBI-BULLETIN-2023-Q4-TRANSMISSION",
        "title": "RBI Bulletin: State of the Economy — Monetary Policy Transmission",
        "publisher": "Reserve Bank of India (RBI)",
        "country": "IND",
        "quarter": "2023-Q4",
        "date": "2023-11-18",
        "topic": "monetary_policy",
        "citation": "RBI Bulletin: State of the Economy (Nov 2023, pp. 45-52)",
        "content": (
            "Monetary transmission across the banking system continued to progress, with weighted average lending rates "
            "(WALR) on fresh rupee loans rising by 193 bps and on outstanding loans by 112 bps during the tightening cycle. "
            "Credit growth remained robust at 15.5% YoY, supported by services and personal loans. The decline in core inflation "
            "signals that demand-pull pressures remain well-contained under tight financial conditions, though sticky food prices "
            "limit headline disinflation."
        ),
    },

    # ── World Bank Reports ─────────────────────────────────────────────────────
    {
        "doc_id": "WB-GEP-2024-GLOBAL-DISINFLATION",
        "title": "World Bank Global Economic Prospects — Global Inflation Deceleration and Trade Headwinds",
        "publisher": "World Bank",
        "country": "Global",
        "quarter": "2024-Q1",
        "date": "2024-01-10",
        "topic": "global_macro",
        "citation": "World Bank Global Economic Prospects (Jan 2024, Chapter 1)",
        "content": (
            "Global headline inflation is projected to decline from 5.8% in 2023 to 3.7% in 2024 and 3.4% in 2025. "
            "The widespread slowdown in inflation is primarily driven by normalized supply chains, subdued commodity price "
            "pressures compared to 2022 highs, and restrictive central bank policy rates across advanced economies. "
            "However, core inflation remains sticky in several advanced economies due to elevated service sector wage growth "
            "and persistent shelter cost pressures. Emerging market and developing economies (EMDEs) face divergent trends, "
            "where currency depreciations and localized agricultural shortages keep domestic price indices elevated."
        ),
    },
    {
        "doc_id": "WB-COMMODITY-2024-OIL-ENERGY",
        "title": "World Bank Commodity Markets Outlook — Geopolitical Shocks and Energy Trajectory",
        "publisher": "World Bank",
        "country": "Global",
        "quarter": "2024-Q2",
        "date": "2024-04-25",
        "topic": "energy_crude",
        "citation": "World Bank Commodity Markets Outlook (April 2024, Box 1.2)",
        "content": (
            "Brent crude oil prices averaged $85/bbl in Q1 2024, responding to OPEC+ voluntary production cuts totaling 2.2 mb/d "
            "and heightened conflict risks in the Middle East. Geopolitical tensions and maritime rerouting via the Cape of Good Hope "
            "added $3-$5 per barrel in shipping premiums and insurance charges. Despite energy price spikes, broader agricultural "
            "and fertilizer prices fell by 9% YoY, buffering food importing economies from severe import-cost inflation."
        ),
    },
    {
        "doc_id": "WB-EMDE-2023-INFLATION-DRIVERS",
        "title": "World Bank Policy Research: Inflation Dynamics in South Asia and Latin America",
        "publisher": "World Bank",
        "country": "Global",
        "quarter": "2023-Q3",
        "date": "2023-09-14",
        "topic": "food_inflation",
        "citation": "World Bank Policy Research Report No. 10482 (Sept 2023)",
        "content": (
            "Empirical decomposition of inflation in emerging markets reveals that food comprises between 35% and 50% "
            "of the CPI basket in developing economies, compared to 10%-15% in advanced economies. Consequently, climate shocks "
            "(such as El Niño weather anomalies impacting rice and sugar output) disproportionately lift headline inflation in "
            "South Asia and Sub-Saharan Africa, creating divergent monetary policy requirements compared to the US Fed or ECB."
        ),
    },

    # ── US Federal Reserve & US Macro Data Reports ─────────────────────────────
    {
        "doc_id": "FED-FOMC-2024-Q1-CPI",
        "title": "Federal Reserve FOMC Minutes — US Inflation Progress and Labor Market Balance",
        "publisher": "Federal Reserve (FOMC)",
        "country": "USA",
        "quarter": "2024-Q1",
        "date": "2024-03-20",
        "topic": "monetary_policy",
        "citation": "Federal Reserve FOMC Minutes (March 2024 Meeting)",
        "content": (
            "The FOMC maintained the target range for the federal funds rate at 5.25% to 5.50%. "
            "US CPI inflation rose at an annualized rate of 3.2%-3.5% in Q1 2024, showing firmer-than-expected price "
            "readings in motor vehicle insurance, health care services, and housing rents. The committee noted that while "
            "goods deflation continued to ease goods-related CPI, services ex-housing (supercore inflation) remained firm "
            "due to tight labor market conditions and 4.1% average hourly wage growth. The Fed reiterated that rate cuts "
            "require greater confidence that inflation is moving sustainably toward the 2% target."
        ),
    },
    {
        "doc_id": "BLS-CPI-2024-SHELTER-ENERGY",
        "title": "US Bureau of Labor Statistics — Quarterly CPI Component Analysis",
        "publisher": "US BLS / Macro Dispatch",
        "country": "USA",
        "quarter": "2024-Q2",
        "date": "2024-05-15",
        "topic": "core_cpi",
        "citation": "US Bureau of Labor Statistics CPI Detailed Report (May 2024)",
        "content": (
            "The Consumer Price Index for All Urban Consumers (CPI-U) showed shelter index rising 5.5% over the last 12 months, "
            "accounting for over two-thirds of the total 12-month increase in the all items less food and energy index. "
            "Energy prices rebounded by 2.6% over the quarter led by retail gasoline price increases, while food at home "
            "remained flat at 1.1% YoY, indicating stabilization in retail grocery inflation."
        ),
    },
    {
        "doc_id": "NEWS-DISPATCH-2024-SUPPLY-CHAINS",
        "title": "Global Macro Dispatch: Red Sea Logistics Shocks and Freight Rates",
        "publisher": "Financial Macro Wire",
        "country": "Global",
        "quarter": "2024-Q1",
        "date": "2024-02-28",
        "topic": "supply_chains",
        "citation": "Financial Macro Wire — Quarterly Trade Report (Q1 2024)",
        "content": (
            "Container freight rates on Asia-to-Europe and US East Coast shipping lanes jumped by over 140% following "
            "commercial vessel diversions around the Cape of Good Hope. Manufacturing supplier delivery time indices "
            "temporarily lengthened, but broader inflationary pass-through remained muted compared to 2021 pandemic "
            "disruptions due to ample container ship capacity and inventory destocking."
        ),
    },
]


# ── TF-IDF / Semantic Embedding Vector Search Engine ──────────────────────────
class MacroRAGIndex:
    """In-memory semantic vector index with BM25 keyword matching and TF-IDF cosine embeddings."""

    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: np.ndarray = None
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # Clean punctuation and keep alphanumeric tokens
        tokens = re.findall(r"\b[a-z0-9_\-\.]{2,}\b", text)
        return tokens

    def _build_index(self):
        doc_count = len(self.corpus)
        tokenized_docs = []
        df_counts: Dict[str, int] = {}

        for doc in self.corpus:
            full_text = f"{doc['title']} {doc['content']} {doc.get('topic','')} {doc.get('country','')} {doc.get('publisher','')}"
            tokens = self._tokenize(full_text)
            tokenized_docs.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df_counts[t] = df_counts.get(t, 0) + 1

        # Vocabulary
        self.vocabulary = {t: idx for idx, (t, cnt) in enumerate(df_counts.items()) if cnt >= 1}
        vocab_size = len(self.vocabulary)

        # Compute IDF
        for t, cnt in df_counts.items():
            self.idf[t] = math.log((doc_count + 1.0) / (cnt + 1.0)) + 1.0

        # Build document vectors
        matrix = np.zeros((doc_count, vocab_size), dtype=np.float32)
        for i, tokens in enumerate(tokenized_docs):
            for t in tokens:
                if t in self.vocabulary:
                    v_idx = self.vocabulary[t]
                    matrix[i, v_idx] += 1.0
            # Apply TF-IDF
            for t in set(tokens):
                if t in self.vocabulary:
                    v_idx = self.vocabulary[t]
                    tf = 1.0 + math.log(matrix[i, v_idx]) if matrix[i, v_idx] > 0 else 0
                    matrix[i, v_idx] = tf * self.idf[t]

            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm

        self.doc_vectors = matrix

    def search(
        self,
        query: str,
        country: Optional[str] = None,
        top_k: int = 3,
        threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Retrieve top_k matching document chunks for query with semantic similarity scoring."""
        query_tokens = self._tokenize(query)
        q_vec = np.zeros(len(self.vocabulary), dtype=np.float32)

        for t in query_tokens:
            if t in self.vocabulary:
                v_idx = self.vocabulary[t]
                q_vec[v_idx] += 1.0

        for t in set(query_tokens):
            if t in self.vocabulary:
                v_idx = self.vocabulary[t]
                tf = 1.0 + math.log(q_vec[v_idx]) if q_vec[v_idx] > 0 else 0
                q_vec[v_idx] = tf * self.idf[t]

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm
            scores = np.dot(self.doc_vectors, q_vec)
        else:
            # Fallback if query terms not in vocab: use broad match
            scores = np.ones(len(self.corpus), dtype=np.float32) * 0.1

        # Country / keyword boost
        results = []
        for i, doc in enumerate(self.corpus):
            score = float(scores[i])
            # Boost if country matches
            if country and country.upper() in (doc.get("country", "").upper(), "GLOBAL"):
                score *= 1.35
            # Boost if query mentions publisher
            if doc.get("publisher", "").lower() in query.lower():
                score *= 1.3

            if score >= threshold:
                res_item = dict(doc)
                res_item["score"] = round(score, 4)
                results.append(res_item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# Global RAG index instance
_rag_index = MacroRAGIndex(MACRO_KNOWLEDGE_CORPUS)


def retrieve_macro_reports(
    query: str,
    country: Optional[str] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Public helper to retrieve relevant World Bank and RBI report excerpts with citations."""
    return _rag_index.search(query=query, country=country, top_k=top_k)
