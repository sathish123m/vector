# Oral explanation — 10 minutes

**Total:** 10:00. **Pace:** about 105–110 words per minute, with room for pauses.
**Setup:** open `submission/oral_script.md` on one screen, timer visible.
**Rule:** if you fall behind, cut “Why” and shorten “Correctness”. Never cut “The bug that mattered”.

---

## 0:00–0:45 — Title

> Good morning. This is ChurnVec, a vector index I built from scratch in Python and NumPy.
> The goal of this milestone was not to beat a library. It was to build the static core, prove
> it is correct, and decide whether Python can carry the deletion and concurrency work that
> comes next. At efSearch 64 on SIFT100K it reaches **0.9846 Recall@10**, within **0.052
> percentage points** of hnswlib, with a fully connected graph and **zero repair links**.

*Cue: pause on the three numbers, then move on.*

## 0:45–1:25 — Why

> Vector search is infrastructure now: retrieval, recommendations, semantic search. HNSW is a
> great static index. The hard part starts after the build: a vector is deleted, a document is
> re-embedded, a customer leaves. Removing a node from a proximity graph can disconnect it,
> and concurrent readers need a stable view while compaction runs. The larger project is about
> those problems. Today I am presenting the foundation that makes them tractable.

*Cue: do not over-explain the 28-week plan. One sentence is enough.*

## 1:25–2:03 — Scope

> I want to be precise about scope. What I built exists and is tested. What I did not build is
> deliberately absent. There is no delete path, no concurrent snapshot, no tiering,
> and no SIFT1M run. I will not use the static result to claim any of those. This discipline
> is part of the work: if I cannot point to an artifact, I do not claim it.

*Cue: this section buys credibility. Say it confidently, not apologetically.*

## 2:03–2:57 — HNSW in one minute

> HNSW is a hierarchy of proximity graphs. Every vector is in layer 0; a random subset is
> promoted to layer 1, a smaller subset to layer 2, and so on. To insert, you descend greedily
> through the upper layers, then run a wider search at the layers where the node lives, pick
> diverse neighbours, and link both ways. To search, you descend to layer 0 and run a
> best-first search there. **M** controls degree; **efConstruction** controls build quality;
> **efSearch** controls how much of the graph you explore at query time. Recall and latency
> move together.

*Cue: walk through build first, then search. No need to read the formulas.*

## 2:57–3:36 — Architecture

> The implementation has three layers. The data layer loads TEXMEX files and computes exact
> ground truth for the snapshot. The index core is dependency-free apart from NumPy:
> GraphStore owns vectors, IDs, adjacency, and in-degrees; HNSW owns the algorithm. The
> harness owns measurement and evidence. That separation let me test the graph on synthetic
> data, swap ground truth for a prefix, and keep the benchmark from leaking into the index.

*Cue: describe the three layers top to bottom; do not read the box labels.*

## 3:36–4:18 — Correctness

> Recall alone is not correctness. The validator checks degree bounds, self-loops, duplicate
> neighbours, the external ID map, in-degree bookkeeping, and directed reachability from the
> entry point. The property tests generate random graphs and assert that every insert leaves
> a valid graph and that a full-width ef returns the exact k distances. The duplicate-vector
> case was found here, by Hypothesis, not by SIFT. That is the point of testing the structure,
> not just the score.

*Cue: the 100,000/100,000 and 0 repair numbers are the bridge to the result.*

## 4:18–5:00 — Method

> Most benchmark mistakes are ground-truth mistakes. The shipped SIFT ground truth is for the
> exact one-million-vector base. SIFT100K is a prefix, so I recomputed exact top-100 for that
> snapshot and cached it. The cache is 3.8 megabytes of IDs, not a 40-gigabyte distance matrix.
> Both engines use M=16, efConstruction=100, the same ef sweep, one thread, and a 32-query
> warmup. Recall is computed over all 10,000 queries, not a sample.

*Cue: if asked, the exact-GT command is in Appendix A.*

## 5:00–5:51 — Result

> Here is the central result. At ef=64 the two recall values are **0.9846** and **0.9851**: a
> gap of **0.052 percentage points**, well inside the two-point gate. The curves overlap across
> the whole sweep, so this is not a lucky point. The latency curves do not overlap. ChurnVec is
> about **6.3 times slower per query** and **11.8 times slower to build**. That is the honest
> reading: the algorithm is right, the Python implementation is not fast. The graph is fully
> reachable and needs zero repairs.

*Cue: this is the section to slow down on. Pause after "not fast."*

## 5:51–7:03 — The bug that mattered

> This is the finding I would want a reviewer to remember. My first 100K run passed the recall
> gate at 0.9881. It also produced **99,673 repair links**. A repair chain through almost the
> whole graph is not a pass. I cut the build to 15,000 vectors and found **7,950 unreachable**,
> split into two dense components with exactly one cross-edge. The optional keep_pruned fill
> was retaining candidates that were farther away than an already-selected neighbour, filling
> capacity that should have held cross-cluster bridges. Disabling it gave a connected graph,
> zero repairs, half the build time, and a 0.35-point recall change that is still far inside
> the gate. The lesson: **recall can hide a broken graph; repair count has to be visible.**

*Cue: tell it as a story: symptom, suspicion, diagnostic, fix, caveat. This is the strongest 80 seconds.*

## 7:03–7:52 — Duplicates

> The second edge case came from a property test with all-identical vectors. With zero
> distances, tie-breaking and fixed-capacity pruning can leave a node with outgoing links but
> no incoming link. Before repair, only 36 of 100 were reachable. I added incremental in-degree
> tracking, made link insertion report whether the new node survived pruning, protected an
> existing node's last incoming edge, and kept a bulk-build repair that runs only when
> reachability is actually broken. SIFT100K still needs zero repairs. I describe this as a
> fallback, not a guarantee.

*Cue: keep the three-layer fix short; the report has the detail.*

## 7:52–8:47 — Decision

> The gate had four criteria: recall, parity, build projection, and latency. All four pass. The
> plan said to start a native core if the 1M build projection exceeded three hours or p50
> exceeded ten milliseconds. Neither fired. So the decision is to **defer the rewrite**. That
> is not a claim that Python is fast. hnswlib is still six times faster per query. But a
> rewrite now would not improve the static result, and it would delay the deletion and
> concurrency work where the real questions are. I will revisit after a measured 1M build and
> churn profiling.

*Cue: say "defer", not "reject". The option stays open.*

## 8:47–9:31 — Limits and next

> The limits are important. There is no 1M run, no delete path, no concurrency, no tiering, and
> no multi-thread scaling. The 1M figure is a linear projection. The next step is a measured
> 1M build, then immutable segments with tombstones and stable-ID deletes. Ground truth gets
> recomputed at checkpoints. Concurrency uses an atomic snapshot registry, compared against
> lock baselines. Tiering starts hot/cold, then the hot/warm/cold ablation. The native-core
> question comes back after those profiles.

*Cue: do not drift into the full 28-week plan. Three next steps are enough.*

## 9:31–10:00 — Conclusion

> Three takeaways. First, the static core exists and is tested end to end. Second, it reaches
> recall parity with hnswlib at the gate point, and the graph is genuinely connected. Third,
> the most useful result was finding a construction bug that the recall number hid. I am happy
> to take questions.

*Cue: stop talking. Let the audience ask.*

---

# Q&A preparation

Answer in 2–4 sentences. Lead with the conclusion, then the evidence.

**1. Why build HNSW from scratch instead of using hnswlib or FAISS?**
> Because the project is about deletion, concurrency, and tiering, not just search. I need to
> own the graph, the ID map, and the persistence format to add those safely. The from-scratch
> version is also a reference oracle for a later native port. hnswlib is the speed baseline,
> not the product.

**2. Is 0.9846 actually competitive, or did you tune to one point?**
> The whole sweep overlaps: at ef=16 it is 0.8637 vs 0.8633, at 32 it is 0.9467 vs 0.9468, at
> 64 it is 0.9846 vs 0.9851, and at 256 it is 0.9987 vs 0.9990. The gate point is not an
> outlier. Recall is stable across the curve.

**3. How do you know the ground truth is right?**
> The shipped SIFT GT is only valid for the exact 1M base, so I did not use it for SIFT100K. I
> recomputed exact top-100 over the 100K prefix with chunked brute force and cached the IDs.
> The integration test compares exact distances against the shipped SIFT10K GT, which is valid
> for that set. The brute-force baseline is itself tested against a naive implementation.

**4. Why is Python so much slower if the algorithm is the same?**
> Graph traversal, heap operations, and per-neighbour decisions run in interpreted code and
> cross into NumPy for small batches. hnswlib is C++ with tight loops and no Python boundary.
> The 6.3× query gap is the price of that. It is acceptable for the static gate, not for a
> production service.

**5. Why not start the native core now?**
> Neither trigger fired: the projected 1M build is 0.67 hours, not three hours, and p50 is 0.978
> ms, not ten milliseconds. A rewrite would consume the next milestone without improving the
> static result. The right time is after a measured 1M build and churn profiling, when I know
> whether the bottleneck is the language or the algorithm.

**6. What exactly was wrong with keep_pruned?**
> The optional fill retained candidates that were farther from the source than an
> already-selected neighbour. At scale those low-quality links occupied capacity that would
> otherwise hold cross-cluster bridges, so layer 0 split into two dense components with almost
> no cross-edges. Disabling the fill restored connectivity and cut build time roughly in half.
> The option is legitimate in the paper; it interacted badly with this data and implementation.

**7. Did the repair path fake the final result?**
> No. The final SIFT100K run reports zero repair links: the graph is connected before repair.
> The repair path is only used for the synthetic duplicate case. The first 100K run did use
> 99,673 repair links, and I rejected that run as a pass even though its recall was higher.

**8. What about deletion? HNSW can't delete, right?**
> HNSW does not support cheap in-place deletion: removing a node can disconnect the graph. But
> that is not a fundamental impossibility. The plan is immutable segments plus tombstones and
> an atomic snapshot registry, with compaction when the dead fraction crosses a threshold. I
> have not implemented or measured that yet, so I am not claiming it.

**9. Is the 1M projection credible?**
> It is a linear projection from 100K, and I label it as such. Insertion cost grows with graph
> depth, so a real 1M build may be slower. That is exactly why the first next step is a measured
> SIFT1M build, not another projection.

**10. What is novel here?**
> Not HNSW itself. The contribution of this milestone is an inspectable from-scratch core with
> explicit invariants, a snapshot-correct benchmark harness, and two documented failure modes:
> pruning-induced fragmentation and duplicate-vector stranding. The novel work, if it succeeds,
> will be the deletion and tiering design in later milestones.

**11. Why only 10,000 queries for SIFT100K?**
> SIFT100K uses the full 10,000-query SIFT1M query set. SIFT10K is the one with only 100 queries;
> that is a property of the official small dataset, and E0 is documented as a smoke test, not a
> performance claim.

**12. What would you do differently with another week?**
> Measure SIFT1M, then build the churn generator and the first segment/tombstone prototype. I
> would also add a multi-threaded read benchmark, because the current implementation is
> single-threaded and that is the biggest unknown for the concurrency milestone.

**13. Does the repair path guarantee connectivity?**
> No. It guarantees reachability for the bulk build it runs on, by threading a chain through
> unreachable nodes. It does not guarantee recall quality on adversarial duplicate data, and it
> is not part of normal insertion. I explicitly forbid the "guaranteed connectivity" claim.

**14. What is the biggest threat to the result?**
> The 1M projection and the single-machine protocol. The recall result is solid because the GT
> is recomputed and both engines are matched, but scaling and multi-threaded behaviour are
> unmeasured. The duplicate repair is also structural, not quality-characterised.

---

# If you have only 5 minutes

Keep the opening, “HNSW in one minute”, “Architecture”, “Method”, “Result”, “The bug that
mattered”, “Decision”, and “Conclusion”. Cut “Why” to one sentence, skip the right-hand side of
“Scope”, and compress “Correctness”, “Duplicates”, and “Limits and next” to one sentence each.
Protect the 80 seconds on “The bug that mattered”.

# Delivery notes

- Do not read the report. The report carries the evidence; you carry the argument.
- Say "recall parity, not speed parity" before anyone asks.
- When you say 0.9846, immediately say "against 0.9851 for hnswlib".
- When you say "zero repair links", add "on the final SIFT100K build".
- If a number is challenged, point to `reports/phase1/e1_sift100k/result.json`.
- If a claim is challenged, use the allowed wording in `docs/claims.md`.
