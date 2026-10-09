# Small development pilot

The retrieval engine was evaluated before its standalone packaging with [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench), using source commit `538026089d1a8a8eff05121d0db89b388f360eba` and dataset revision `7ea066982b140a19337e17e60d45d4076e042faf` (`ai-hyz/MemoryAgentBench`).

This was 32 questions across six public contexts, eight questions per category, selected with seed 42. It was a development set reused to diagnose retrieval failures, not an independent held-out evaluation or a full leaderboard run. The answering model was GLM-5.3-Flash through a gateway, temperature 0, max_tokens 4096. Index ingestion used 4096 GPT-token source chunks, then 448 BGE-token windows with 64-token overlap. Six retrieved anchors were supplied; the fixed engine added neighboring context. The 32 no-memory answers were reused exactly rather than generated again.

| Category | Before repair | After repair | Scoring |
| --- | ---: | ---: | --- |
| Accurate Retrieval | 5/8 | 8/8 | Official substring match |
| Conflict Resolution | 6/8 | 7/8 | Official substring match |
| Test-Time Learning | 0/8 | 0/8 | Official strict format/match |
| Long-Range Understanding | 0/8 | 0/8 | Official strict format/match |
| Total | 11/32 (34.4%) | 15/32 (46.9%) | Official strict |

After stripping label prefixes or extracting the answer field from JSON, the diagnostic total was 15/32 → 23/32 (46.9% → 71.9%); TTL 3/8 → 6/8 and LRU 1/8 → 2/8. These diagnostic numbers are **not** official leaderboard scores. Three final LRU outputs were empty after the generation budget was consumed. End-to-end scores include generation and formatting behavior, rather than measuring retrieval accuracy alone.

The repair addressed a top lexical match buried by fusion, a relevant scene ranked outside the top six, missing neighboring evidence, and an initial 48-term limit that discarded late query words. One former empty answer became correct without increasing the output budget. A single repeated sample does not isolate each rule's causal contribution or generation variability.

This repository contains the algorithm, synthetic regression tests and this aggregate summary. It does not publish real agent history, model-provider credentials, full dataset excerpts or the gateway adapter. Subsequent standalone packaging changes are validated through unit and real-model integration checks; the 32-question model evaluation was not rerun on the packaged CLI.
