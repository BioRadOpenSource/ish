from collections import Optional
from memory import Span


@value
@register_passable("trivial")
struct MatchResult:
    """The start and end of the match, (start, end]."""

    var start: Int
    """Inclusive match start in the haystack."""
    var end: Int
    """Exclusive match end in the haystack."""


trait Matcher(Copyable, Movable):
    """Trait for how matchers must work."""

    fn first_match(
        read self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""
        ...

    @always_inline
    fn convert_ascii_to_encoding(read self, value: UInt8) -> UInt8:
        """Convert an ascii byte to an encoded byte."""
        ...

    @always_inline
    fn convert_encoding_to_ascii(read self, value: UInt8) -> UInt8:
        """Convert an encoded byte to an ascii byte."""
        ...
