from collections import Optional
from gpu.host import DeviceBuffer
from memory import Span, UnsafePointer

from ishlib.matcher.alignment.scoring_matrix import MatrixKind
from ishlib.searcher_settings import SemiGlobalEndsFreeness


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

    fn encoded_pattern(ref self) -> Span[UInt8, __origin_of(self)]:
        """Get the encoded pattern to be matched."""
        ...

    @always_inline
    fn max_alignment_score(read self) -> Int:
        """Return the best possible alignment score given the matrix used for the pattern.
        """
        ...

    @always_inline
    fn score_threshold(read self) -> Float32:
        """Returns the score threshold needed to be concidered a match."""
        ...


trait GpuMatcher(Matcher):
    fn matrix_bytes(read self) -> UnsafePointer[Int8]:
        ...

    fn matrix_len(read self) -> UInt:
        ...

    fn matrix_kind(read self) -> MatrixKind:
        ...

    fn find_start(
        read self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Int:
        """Find the start position of the match for the haystack. Assumes this is a match.
        """
        ...

    @staticmethod
    fn batch_match_coarse[
        max_query_length: UInt,
        max_target_length: UInt,
    ](
        query: DeviceBuffer[DType.uint8],
        ref_buffer: DeviceBuffer[DType.uint8],
        target_ends: DeviceBuffer[DType.uint32],
        score_result_buffer: DeviceBuffer[DType.int32],
        query_end_result_buffer: DeviceBuffer[DType.int32],
        ref_end_result_buffer: DeviceBuffer[DType.int32],
        basic_matrix_values: DeviceBuffer[DType.int8],
        basic_matrix_len: Int,
        matrix_kind: MatrixKind,
        query_len: Int,
        target_ends_len: Int,
        thread_count: Int,
        gap_open: UInt,
        gap_extend: UInt,
        ends_free: SemiGlobalEndsFreeness,
    ):
        """A coarse grain mono-directional match function to be used as a GPU kernel.

        If there is match found then the reverse will need to be run to find the start point.
        """
        ...


trait Searchable(Copyable, Movable, CollectionElement):
    fn buffer_to_search(ref self) -> Span[UInt8, __origin_of(self)]:
        ...


trait SearchableWithIndex(Searchable):
    fn original_index(read self) -> UInt:
        ...


@value
@register_passable
struct WhereComputed:
    var value: Int
    alias Cpu = Self(0)
    alias Gpu = Self(1)

    fn __eq__(read self, read other: Self) -> Bool:
        return self.value == other.value


@value
@register_passable
struct ComputedMatchResult:
    var result: MatchResult
    var where_computed: WhereComputed
    var index: UInt
