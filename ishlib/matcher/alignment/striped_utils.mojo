from memory import memset_zero
from sys import llvm_intrinsic

from ishlib.matcher.alignment import AlignedMemory


@always_inline
fn saturating_sub[
    data: DType, width: Int
](lhs: SIMD[data, width], rhs: SIMD[data, width]) -> SIMD[data, width]:
    """Saturating SIMD subtraction.

    https://llvm.org/docs/LangRef.html#llvm-usub-sat-intrinsics
    https://llvm.org/docs/LangRef.html#llvm-ssub-sat-intrinsics
    """
    constrained[data.is_integral()]()

    @parameter
    if data.is_unsigned():
        return llvm_intrinsic["llvm.usub.sat", __type_of(lhs)](lhs, rhs)
    else:
        return llvm_intrinsic["llvm.ssub.sat", __type_of(lhs)](lhs, rhs)


@always_inline
fn saturating_add[
    data: DType, width: Int
](lhs: SIMD[data, width], rhs: SIMD[data, width]) -> SIMD[data, width]:
    """Saturating SIMD addition.

    https://llvm.org/docs/LangRef.html#llvm-uadd-sat-intrinsics
    https://llvm.org/docs/LangRef.html#llvm-sadd-sat-intrinsics
    """
    constrained[data.is_integral()]()

    @parameter
    if data.is_unsigned():
        return llvm_intrinsic["llvm.uadd.sat", __type_of(lhs)](lhs, rhs)
    else:
        return llvm_intrinsic["llvm.sadd.sat", __type_of(lhs)](lhs, rhs)


@value
@register_passable
struct ScoreSize:
    """Controls the precision used for alignment scores.

    This directly effects the max possible score.
    """

    var value: UInt8

    alias Byte = Self(0)
    """Use only 8-bit (byte) precision for scores."""
    alias Word = Self(1)
    """Use only 16-bit (word) precision for scores."""
    alias Adaptive = Self(2)
    """Use both precisions, adaptive strategy."""

    fn __eq__(read self, read other: Self) -> Bool:
        return self.value == other.value

    fn __str__(read self) -> String:
        if self.value == Self.Byte.value:
            return "byte"
        elif self.value == self.Word.value:
            return "word"
        else:
            return "adaptive"


@value
@register_passable("trivial")
struct AlignmentStartEndResult:
    var score: Int32
    """Alignment score."""
    var query_start: Int32
    """0-based inclusive query start."""
    var query_end: Int32
    """0-based inclusive query end."""
    var target_start: Int32
    """0-based inclusive target start."""
    var target_end: Int32
    """0-based inclusive target end."""


@value
@register_passable("trivial")
struct AlignmentResult:
    var best: AlignmentEnd
    var overflow_detected: Bool

    fn __init__(out self):
        self.best = AlignmentEnd(-1, -1, -1)
        self.overflow_detected = False

    fn __init__(
        out self, best: AlignmentEnd, *, overflow_detected: Bool = False
    ):
        self.best = best
        self.overflow_detected = overflow_detected


@value
@register_passable("trivial")
struct AlignmentEnd:
    var score: Int32
    var reference: Int32  # 0-based
    var query: Int32  # alignment ending position on query, 0-based


@value
struct ProfileVectors[dt: DType, width: Int]:
    var pv_h_store: AlignedMemory[dt, width, width]
    var pv_h_load: AlignedMemory[dt, width, width]
    var pv_e: AlignedMemory[dt, width, width]
    var pv_h_max: AlignedMemory[dt, width, width]
    var zero: SIMD[dt, width]
    var max_column: List[SIMD[dt, 1]]
    var end_query_column: List[Int32]
    var segment_length: Int32
    var query_len: Int32

    fn __init__(out self, query_len: Int32):
        var segment_length = (query_len + width - 1) // width

        var zero = SIMD[dt, width](0)
        # Stores the H values (scores) for the current row of the dynamic programming matrix
        var pv_h_store = AlignedMemory[dt, width, width](Int(segment_length))
        # Contains scores from the previous row that will be loaded for calculation
        var pv_h_load = AlignedMemory[dt, width, width](Int(segment_length))
        # Tracks scores for alignments that end with gaps in the query seq (horizontal gaps in visualization)
        var pv_e = AlignedMemory[dt, width, width](Int(segment_length))
        # Stores the max scores seen in each column for traceback
        var pv_h_max = AlignedMemory[dt, width, width](Int(segment_length))

        # List to record the largest score of each reference position
        var max_column = List[SIMD[dt, 1]]()
        # List to record the alignment query ending position of the largest score of each reference position
        var end_query_column = List[Int32]()

        self.pv_h_store = pv_h_store
        self.pv_h_load = pv_h_load
        self.pv_e = pv_e
        self.pv_h_max = pv_h_max
        self.zero = zero
        self.segment_length = segment_length
        self.query_len = query_len
        self.max_column = max_column
        self.end_query_column = end_query_column

    fn zero_out(mut self):
        memset_zero(self.pv_h_store.ptr, Int(self.segment_length))
        memset_zero(self.pv_h_load.ptr, Int(self.segment_length))
        memset_zero(self.pv_e.ptr, Int(self.segment_length))
        memset_zero(self.pv_h_max.ptr, Int(self.segment_length))

    fn init_columns(mut self, ref_len: Int):
        self.max_column.resize(ref_len, 0)
        self.end_query_column.resize(ref_len, 0)
        memset_zero(self.max_column.unsafe_ptr(), ref_len)
        memset_zero(self.end_query_column.unsafe_ptr(), ref_len)
