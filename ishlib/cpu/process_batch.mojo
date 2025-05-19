from algorithm.functional import parallelize

from ishlib.matcher import (
    SearchableWithIndex,
    ComputedMatchResult,
    Matcher,
    GpuMatcher,
    WhereComputed,
    MatchResult,
)
from ishlib.searcher_settings import SearcherSettings


fn parallel_starts_ends[
    M: Matcher, S: SearchableWithIndex
](
    read matcher: M,
    read settings: SearcherSettings,
    read seqs: Span[S],
    mut output: List[Optional[ComputedMatchResult]],
):
    fn do_matching(index: Int) capturing:
        var target = Pointer(to=seqs[index])
        var result = matcher.first_match(
            target[].buffer_to_search(), matcher.encoded_pattern()
        )
        if result:
            output[target[].original_index()] = ComputedMatchResult(
                result.value(),
                WhereComputed.Cpu,
                index,
            )

    parallelize[do_matching](len(seqs), settings.threads)


fn parallel_starts[
    M: GpuMatcher,
    S: SearchableWithIndex,
    where_computed: WhereComputed = WhereComputed.Cpu,
](
    read matcher: M,
    read settings: SearcherSettings,
    read target_scores: Span[Int32],
    read target_ends: Span[Int32],
    read seqs: Span[S],
    mut outputs: List[Optional[ComputedMatchResult]],
    offset: UInt,
):
    fn do_matching(index: Int) capturing:
        if (
            Float32(target_scores[index]) / matcher.max_alignment_score()
            < matcher.score_threshold()
        ):
            return
        var start = matcher.find_start(
            seqs[index].buffer_to_search()[
                0 : Int(target_ends[index])
            ],  # Search only from the target end backwards
            matcher.encoded_pattern(),
        )
        outputs[seqs[index].original_index()] = ComputedMatchResult(
            MatchResult(start, Int(target_ends[index])),
            where_computed,
            index + offset,
        )

    parallelize[do_matching](len(seqs), settings.threads)
