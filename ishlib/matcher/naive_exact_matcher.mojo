"""Naive linear exact match search."""
from ishlib.matcher import Matcher, MatchResult


@value
struct NaiveExactMatcher(Matcher):
    fn first_match(
        mut self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""

        for h in range(0, len(haystack)):
            var matched = True
            for p in range(0, len(pattern)):
                if pattern[p] != haystack[h + p]:
                    matched = False
                    break
            if matched:
                return MatchResult(h, h + len(pattern))

        return None
