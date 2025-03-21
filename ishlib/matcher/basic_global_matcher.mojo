"""Needleman-Wunsch global alignment."""
from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix
from ishlib.matcher.alignment.global_aln.basic import (
    needleman_wunsch_parasail,
    needleman_wunsch_full_naive,
)


@value
struct BasicGlobalMatcher(Matcher):
    fn first_match(
        mut self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""
        # var result = needleman_wunsch_full_naive(
        #     haystack,
        #     pattern,
        #     ScoringMatrix.all_ascii_default_matrix(),
        #     match_score=2,
        #     mismatch_score=-2,
        #     gap_open_penalty=-3,
        #     gap_extension_penalty=-1,
        # )
        var result = needleman_wunsch_parasail(
            haystack,
            pattern,
            ScoringMatrix.all_ascii_default_matrix(),
            match_score=2,
            mismatch_score=-2,
            gap_open_penalty=-3,
            gap_extension_penalty=-1,
        )
        if result.score >= len(pattern):
            return MatchResult(
                result.coords.value()[0], result.coords.value()[1]
            )

        return None
