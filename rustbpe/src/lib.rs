use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::HashMap as StdHashMap;
use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;
use pyo3::types::PyIterator;
use ahash::AHashMap;
use compact_str::CompactString;
use rayon::prelude::*;

// Default GPT-4 style regex pattern for splitting text
const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

type Pair = (u32, u32);

/// A Byte Pair Encoding tokenizer that matches the GPT-4 style implementation
#[pyclass]
pub struct Tokenizer {
    /// Maps pairs of token IDs to their merged token ID
    pub merges: StdHashMap<Pair, u32>,
    /// The regex pattern used for text splitting
    pub pattern: String,
    /// Compiled regex for efficiency
    compiled_pattern: Regex,
}

// ------------------------ internal helpers ------------------------

#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    #[inline]
    fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline]
    fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair> + 'a {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    /// Merge all non-overlapping occurrences of pair -> new_id.
    /// Returns a small Vec of local pair-count deltas for THIS word only:
    /// -1 for removed pairs, +1 for newly created pairs.
    ///
    /// NOTE: this version deliberately avoids a HashMap in the hot loop.
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }
        let mut out: Vec<u32> = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);
        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                let left = out.last().copied();
                let right = if i + 2 < n { Some(self.ids[i + 2]) } else { None };
                // remove old pairs
                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }
                // write merged token
                out.push(new_id);
                i += 2; // skip 'a' and 'b'
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }
        self.ids = out;
        deltas
    }
}

// Fix C: lean MergeJob without embedded pos set
#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
}

impl PartialEq for MergeJob {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by count; tie-break: deterministically prefer smaller pair (ascending order)
        // other.pair.cmp(&self.pair) reverses comparison for max-heap so smaller pair wins
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            other.pair.cmp(&self.pair)
        }
    }
}

// Fix A+B: return i64 counts; Fix B: return Vec<usize> positions (sorted+deduped)
// C6: #[inline] removed — large parallel function, inlining is not beneficial here
fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i64>, AHashMap<Pair, Vec<usize>>) {
    let (pair_counts, pair_positions) = words
        .par_iter()
        .enumerate()
        .map(|(i, w)| {
            let mut local_pc: AHashMap<Pair, i64> = AHashMap::new();
            let mut local_wtu: AHashMap<Pair, Vec<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for pair in w.pairs() {
                    *local_pc.entry(pair).or_default() += counts[i] as i64;
                    // Each word contributes at most one position entry (its own index)
                    let positions = local_wtu.entry(pair).or_default();
                    if positions.is_empty() {
                        positions.push(i);
                    }
                }
            }
            (local_pc, local_wtu)
        })
        .reduce(
            || (AHashMap::new(), AHashMap::new()),
            |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                for (k, v) in pc {
                    *acc_pc.entry(k).or_default() += v;
                }
                for (k, mut v) in wtu {
                    acc_wtu.entry(k).or_default().append(&mut v);
                }
                (acc_pc, acc_wtu)
            },
        );

    // Fix B: sort and dedup positions for determinism
    let mut pair_positions_sorted = pair_positions;
    for positions in pair_positions_sorted.values_mut() {
        positions.sort_unstable();
        positions.dedup();
    }

    (pair_counts, pair_positions_sorted)
}

// ------------------------ END helpers ------------------------

impl Tokenizer {
    /// Core incremental BPE training given unique words and their counts.
    /// words: one entry per unique chunk (Vec<u32> of token-ids/bytes).
    /// counts: same length as words, count per chunk.
    fn train_core_incremental(&mut self, words: Vec<Word>, counts: Vec<i32>, vocab_size: u32) {
        let mut words = words;
        assert!(vocab_size >= 256, "vocab_size must be at least 256");
        let num_merges = vocab_size - 256;
        log::info!("Starting BPE training: {} merges to compute", num_merges);
        self.merges.clear();

        // ---- Initial pair_counts and pair_positions (parallel) ----
        log::info!("Computing initial pair counts from {} unique sequences", words.len());
        // Fix A: pair_counts is i64; Fix C: pair_positions is side map of Vec<usize>
        let (mut pair_counts, mut pair_positions) = count_pairs_parallel(&words, &counts);

        // ---- Build heap deterministically (Fix B: sort init input) ----
        log::info!("Building heap with {} unique pairs", pair_counts.len());
        // Fix B: sort pairs before pushing so identical-count ties break on pair order
        let mut init_pairs: Vec<(Pair, i64)> = pair_counts.iter().map(|(&p, &c)| (p, c)).collect();
        init_pairs.sort_unstable_by_key(|(pair, _)| *pair);

        let mut heap = OctonaryHeap::with_capacity(init_pairs.len());
        for (pair, count) in init_pairs {
            if count > 0 {
                heap.push(MergeJob { pair, count: count as u64 });
            }
        }

        // ---- Merge loop ----
        log::info!("Starting merge loop");
        let mut merges_done = 0u32;
        let mut last_log_percent = 0u32;

        while merges_done < num_merges {
            let Some(mut top) = heap.pop() else { break; };

            // Fix A: guard cast — clamp negative counts to 0 before u64 cast
            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            let current_u64 = current.max(0) as u64;
            if top.count != current_u64 {
                top.count = current_u64;
                if top.count > 0 {
                    heap.push(top);
                }
                continue;
            }

            // Heap invariant: we never push count-0 jobs, so top.count > 0 here,
            // which implies current > 0. Assert in debug builds to catch drift early.
            debug_assert!(current > 0, "heap invariant violated: non-positive count for top pair");

            // Record merge
            let new_id = 256 + merges_done;
            self.merges.insert(top.pair, new_id);

            let (a, b) = top.pair;

            // Fix C: look up positions from side map (not embedded in heap node)
            let positions = pair_positions.remove(&top.pair).unwrap_or_default();

            let mut local_pos_updates: AHashMap<Pair, Vec<usize>> = AHashMap::new();
            for &word_idx in &positions {
                // Fix D: stale scan guard — skip words where pair is already absent
                if !words[word_idx].ids.windows(2).any(|w| w[0] == a && w[1] == b) {
                    continue;
                }

                // Apply merge to this word and collect pair-count deltas
                let changes = words[word_idx].merge_pair(top.pair, new_id);

                // Fix A: widen delta arithmetic to i64 to prevent overflow
                for (pair, delta) in changes {
                    let delta_total = delta as i64 * counts[word_idx] as i64;
                    if delta_total != 0 {
                        *pair_counts.entry(pair).or_default() += delta_total;
                        if delta > 0 {
                            local_pos_updates.entry(pair).or_default().push(word_idx);
                        }
                    }
                }
            }

            // Fix B+C: merge new positions into pair_positions with sort+dedup.
            // Fix C4: sort updates by pair for deterministic heap push order.
            let mut sorted_updates: Vec<(Pair, Vec<usize>)> = local_pos_updates.into_iter().collect();
            sorted_updates.sort_unstable_by_key(|(pair, _)| *pair);

            for (pair, mut new_positions) in sorted_updates {
                new_positions.sort_unstable();
                new_positions.dedup();

                let existing = pair_positions.entry(pair).or_default();
                existing.extend(new_positions);
                existing.sort_unstable();
                existing.dedup();

                let cnt = *pair_counts.get(&pair).unwrap_or(&0);
                if cnt > 0 {
                    heap.push(MergeJob { pair, count: cnt.max(0) as u64 });
                }
            }

            merges_done += 1;

            // Log progress every 1%
            let current_percent = (merges_done * 100) / num_merges;
            if current_percent > last_log_percent {
                log::info!(
                    "Progress: {}% ({}/{} merges) - Last merge: {:?} -> {} (frequency: {})",
                    current_percent,
                    merges_done,
                    num_merges,
                    top.pair,
                    new_id,
                    top.count
                );
                last_log_percent = current_percent;
            }
        }

        log::info!("Finished training: {} merges completed", merges_done);
    }
}

/// Public methods for the Tokenizer class that will be exposed to Python.
#[pymethods]
impl Tokenizer {
    /// Create a new Tokenizer
    #[new]
    pub fn new() -> Self {
        Self {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").expect("Empty regex should be valid"),
        }
    }

    /// Train from a streaming iterator (parallel ingestion).
    /// We refill a Rust Vec<String> buffer under the GIL, then release the GIL
    /// to do the heavy splitting and counting **in parallel** with rayon.
    ///
    /// Fix C1: uses PyO3 0.27 safe iterator API (try_iter / unbind / it.next())
    /// instead of raw unsafe PyObject_GetIter / PyIter_Next FFI calls.
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, pattern=None))]
    #[pyo3(text_signature = "(self, iterator, vocab_size, buffer_size=8192, pattern=None)")]
    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        pattern: Option<String>,
    ) -> PyResult<()> {
        // Use provided pattern or default to GPT-4 pattern
        let pattern_str = pattern.unwrap_or_else(|| GPT4_PATTERN.to_string());

        // Update the stored pattern and compile it
        self.pattern = pattern_str.clone();
        self.compiled_pattern = Regex::new(&pattern_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid regex pattern: {}", e)))?;

        // C1: acquire a Python iterator via the safe PyO3 0.27 API, then unbind
        // it from the current GIL token so we can rebind it inside the refill closure.
        let py_iter: Py<PyIterator> = iterator.try_iter()?.unbind();

        // Global chunk counts
        let mut counts: AHashMap<CompactString, i32> = AHashMap::new();

        // Temporary buffer we refill under the GIL
        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        log::info!("Processing sequences from iterator (buffer_size: {})", buffer_size);
        let mut total_sequences = 0u64;

        // Helper: refill buf with up to buffer_size strings from the Python iterator.
        // Returns Ok(true) if the iterator is exhausted, Ok(false) otherwise.
        // C1: uses safe it.next() instead of raw PyIter_Next FFI.
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            Python::attach(|py| {
                buf.clear();
                let mut it = py_iter.bind(py).clone();
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    match it.next() {
                        Some(Ok(obj)) => {
                            let s: String = obj.extract()?;
                            buf.push(s);
                        }
                        Some(Err(e)) => return Err(e),
                        None => return Ok(true), // iterator exhausted
                    }
                }
            })
        };

        // Stream ingestion loop: refill under GIL, process without GIL (parallel)
        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }

            total_sequences += buf.len() as u64;
            let pattern = self.compiled_pattern.clone();

            let local: AHashMap<CompactString, i32> = py.detach(|| {
                buf.par_iter()
                    .map(|s| {
                        let mut m: AHashMap<CompactString, i32> = AHashMap::new();
                        for mat in pattern.find_iter(s) {
                            let piece = mat.expect("regex match failed").as_str();
                            *m.entry(CompactString::from(piece)).or_default() += 1;
                        }
                        m
                    })
                    .reduce(
                        || AHashMap::new(),
                        |mut a, b| {
                            for (k, v) in b {
                                *a.entry(k).or_default() += v;
                            }
                            a
                        },
                    )
            });

            // Merge local into global (single-threaded)
            for (k, v) in local {
                *counts.entry(k).or_default() += v;
            }

            if exhausted {
                break;
            }
        }

        log::info!(
            "Processed {} sequences total, {} unique",
            total_sequences,
            counts.len()
        );

        // Fix C3: sort pieces before materializing to get deterministic word-to-index
        // assignment across runs. AHashMap is kept for fast ingestion; sorting happens
        // once here at the boundary.
        let mut sorted_counts: Vec<(CompactString, i32)> = counts.into_iter().collect();
        sorted_counts.sort_unstable_by(|a, b| a.0.as_str().cmp(b.0.as_str()));

        let mut words = Vec::with_capacity(sorted_counts.len());
        let mut cvec = Vec::with_capacity(sorted_counts.len());
        for (chunk, c) in sorted_counts {
            words.push(Word::new(chunk.as_bytes().iter().map(|&b| b as u32).collect()));
            cvec.push(c);
        }

        self.train_core_incremental(words, cvec, vocab_size);
        Ok(())
    }

    /// Return the regex pattern
    pub fn get_pattern(&self) -> String {
        self.pattern.clone()
    }

    /// Return the mergeable ranks (token bytes -> token id / rank)
    pub fn get_mergeable_ranks(&self) -> Vec<(Vec<u8>, u32)> {
        let mut mergeable_ranks = Vec::new();

        // Build vocabulary incrementally from low to high token IDs
        let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|i| vec![i as u8]).collect();
        for (i, bytes) in token_bytes.iter().enumerate() {
            mergeable_ranks.push((bytes.clone(), i as u32));
        }

        // Sort merges by token id (so we can reconstruct bytes progressively)
        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &token_id)| token_id);

        for (&pair, &merged_id) in sorted_merges {
            let (left, right) = pair;
            let mut merged_bytes = token_bytes[left as usize].clone();
            merged_bytes.extend(&token_bytes[right as usize]);
            if token_bytes.len() <= merged_id as usize {
                token_bytes.resize(merged_id as usize + 1, Vec::new());
            }
            token_bytes[merged_id as usize] = merged_bytes.clone();
            mergeable_ranks.push((merged_bytes, merged_id));
        }

        mergeable_ranks
    }

    /// Fix E: O(n log n) encode via lazy-deletion priority queue + doubly-linked index array.
    /// Fix C5: uses OctonaryHeap (8-ary, better cache locality) instead of std BinaryHeap.
    ///
    /// Min-heap keyed by (merge_rank, position): lowest rank = earliest merge.
    /// Deleted positions are marked with ids[pos] = u32::MAX and skipped on pop.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut all_ids = Vec::new();

        for m in self.compiled_pattern.find_iter(text) {
            let chunk = m.expect("regex match failed").as_str();
            let bytes: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();
            let n = bytes.len();

            if n == 0 {
                continue;
            }
            if n == 1 {
                all_ids.push(bytes[0]);
                continue;
            }

            // ids[i]: current token at position i; u32::MAX marks a deleted (merged-away) slot
            let mut ids = bytes;

            // Doubly-linked index arrays for O(1) neighbor lookup and deletion
            // prev[0] = usize::MAX (no left neighbor for first element)
            // next[n-1] = n (sentinel: "past the end")
            let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
            let mut next: Vec<usize> = (1..=n).collect();

            // Fix C5: OctonaryHeap with Reverse wrapper gives a min-heap.
            // Key = (merge_rank, position) — lower rank = earlier merge;
            // position breaks equal-rank ties deterministically (leftmost first).
            let mut heap: OctonaryHeap<Reverse<(u32, usize)>> = OctonaryHeap::new();

            // Push all initial adjacent pairs that have a known merge rank
            for i in 0..n - 1 {
                let pair = (ids[i], ids[i + 1]);
                if let Some(&rank) = self.merges.get(&pair) {
                    heap.push(Reverse((rank, i)));
                }
            }

            while let Some(Reverse((rank, pos))) = heap.pop() {
                // Validate: skip if pos was deleted (marked u32::MAX)
                if ids[pos] == u32::MAX {
                    continue;
                }

                let ni = next[pos];

                // Validate: skip if pos is the last element (no right neighbor in range)
                if ni >= n {
                    continue;
                }

                // Validate: stale entry — the pair at this position no longer matches rank
                let pair = (ids[pos], ids[ni]);
                if self.merges.get(&pair) != Some(&rank) {
                    continue;
                }

                // Perform the merge: ids[pos] = merged token, ids[ni] = deleted
                ids[pos] = rank;
                ids[ni] = u32::MAX; // mark right slot as deleted

                // Update linked list to skip the deleted slot
                let nni = next[ni];
                next[pos] = nni;
                if nni < n {
                    prev[nni] = pos;
                }

                // Push newly valid left-neighbor pair: (prev[pos], pos)
                let pi = prev[pos];
                if pi != usize::MAX {
                    let left_pair = (ids[pi], rank);
                    if let Some(&new_rank) = self.merges.get(&left_pair) {
                        heap.push(Reverse((new_rank, pi)));
                    }
                }

                // Push newly valid right-neighbor pair: (pos, next[pos])
                let new_ni = next[pos];
                if new_ni < n {
                    let right_pair = (rank, ids[new_ni]);
                    if let Some(&new_rank) = self.merges.get(&right_pair) {
                        heap.push(Reverse((new_rank, pos)));
                    }
                }
            }

            // Collect surviving token ids by following the linked-list chain from position 0
            let mut i = 0;
            loop {
                all_ids.push(ids[i]);
                let ni = next[i];
                if ni >= n {
                    break;
                }
                i = ni;
            }
        }

        all_ids
    }
}

#[pymodule]
fn rustbpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); // forwards Rust log to Python's logging
    m.add_class::<Tokenizer>()?;
    Ok(())
}
