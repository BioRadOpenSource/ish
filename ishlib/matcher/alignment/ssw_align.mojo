"""Striped Smith-Waterman Alignment."""

from builtin.math import max
from collections import InlineArray
from math import sqrt
from memory import pack_bits, memset_zero
from sys.intrinsics import likely, unlikely, assume, prefetch
from sys.info import simdwidthof

alias SIMD_U8_WIDTH = simdwidthof[UInt8]()
alias SIMD_U16_WIDTH = simdwidthof[UInt16]()

# fmt: off
alias AA_TO_NUM = InlineArray[UInt8, 128](
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 0,  20, 4,  3,  6,  13, 7,  8,  9,  23, 11, 10, 12, 2,  23,
    14, 5,  1,  15, 16, 23, 19, 17, 22, 18, 21, 23, 23, 23, 23, 23,
    23, 0,  20, 4,  3,  6,  13, 7,  8,  9,  23, 11, 10, 12, 2,  23,
    14, 5,  1,  15, 16, 23, 19, 17, 22, 18, 21, 23, 23, 23, 23, 23
)
"""Table to convert a UInt8 to an AA corresponding to the index in the Blosum50 matrix"""

alias NUM_TO_AA = InlineArray[UInt8, 24](
ord("A"), ord("R"), ord("N"), ord("D"), ord("C"), ord("Q"),
ord("E"), ord("G"), ord("H"), ord("I"), ord("L"), ord("K"), 
ord("M"), ord("F"), ord("P"), ord("S"), ord("T"), ord("W"),
ord("Y"), ord("V"), ord("B"), ord("Z"), ord("X"), ord("*")
)
"""Convert an AA from Blosum50 to ascii"""

# alias BLOSUM50= InlineArray[Int8, 576]( # copied from SSW
#     # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X   *
#      5, -2, -1, -2, -1, -1, -1,  0, -2, -1, -2, -1, -1, -3, -1,  1,  0, -3, -2,  0, -2, -1, -1, -5,  # A
#     -2,  7, -1, -2, -4,  1,  0, -3,  0, -4, -3,  3, -2, -3, -3, -1, -1, -3, -1, -3, -1,  0, -1, -5,  # R
#     -1, -1,  7,  2, -2,  0,  0,  0,  1, -3, -4,  0, -2, -4, -2,  1,  0, -4, -2, -3,  5,  0, -1, -5,  # N
#     -2, -2,  2,  8, -4,  0,  2, -1, -1, -4, -4, -1, -4, -5, -1,  0, -1, -5, -3, -4,  6,  1, -1, -5,  # D
#     -1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1, -3, -3, -1, -5,  # C
#     -1,  1,  0,  0, -3,  7,  2, -2,  1, -3, -2,  2,  0, -4, -1,  0, -1, -1, -1, -3,  0,  4, -1, -5,  # Q
#     -1,  0,  0,  2, -3,  2,  6, -3,  0, -4, -3,  1, -2, -3, -1, -1, -1, -3, -2, -3,  1,  5, -1, -5,  # E
#      0, -3,  0, -1, -3, -2, -3,  8, -2, -4, -4, -2, -3, -4, -2,  0, -2, -3, -3, -4, -1, -2, -1, -5,  # G
#     -2,  0,  1, -1, -3,  1,  0, -2, 10, -4, -3,  0, -1, -1, -2, -1, -2, -3,  2, -4,  0,  0, -1, -5,  # H
#     -1, -4, -3, -4, -2, -3, -4, -4, -4,  5,  2, -3,  2,  0, -3, -3, -1, -3, -1,  4, -4, -3, -1, -5,  # I
#     -2, -3, -4, -4, -2, -2, -3, -4, -3,  2,  5, -3,  3,  1, -4, -3, -1, -2, -1,  1, -4, -3, -1, -5,  # L
#     -1,  3,  0, -1, -3,  2,  1, -2,  0, -3, -3,  6, -2, -4, -1,  0, -1, -3, -2, -3,  0,  1, -1, -5,  # K
#     -1, -2, -2, -4, -2,  0, -2, -3, -1,  2,  3, -2,  7,  0, -3, -2, -1, -1,  0,  1, -3, -1, -1, -5,  # M
#     -3, -3, -4, -5, -2, -4, -3, -4, -1,  0,  1, -4,  0,  8, -4, -3, -2,  1,  4, -1, -4, -4, -1, -5,  # F
#     -1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3, -2, -1, -1, -5,  # P
#      1, -1,  1,  0, -1,  0, -1,  0, -1, -3, -3,  0, -2, -3, -1,  5,  2, -4, -2, -2,  0,  0, -1, -5,  # S
#      0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  2,  5, -3, -2,  0,  0, -1, -1, -5,  # T
#     -3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1,  1, -4, -4, -3, 15,  2, -3, -5, -2, -1, -5,  # W
#     -2, -1, -2, -3, -3, -1, -2, -3,  2, -1, -1, -2,  0,  4, -3, -2, -2,  2,  8, -1, -3, -2, -1, -5,  # Y
#      0, -3, -3, -4, -1, -3, -3, -4, -4,  4,  1, -3,  1, -1, -3, -2,  0, -3, -1,  5, -3, -3, -1, -5,  # V
#     -2, -1,  5,  6, -3,  0,  1, -1,  0, -4, -4,  0, -3, -4, -2,  0,  0, -5, -3, -3,  6,  1, -1, -5,  # B
#     -1,  0,  0,  1, -3,  4,  5, -2,  0, -3, -3,  1, -1, -4, -1,  0, -1, -2, -2, -3,  1,  5, -1, -5,  # Z
#     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -5,  # X
#     -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,  1   # *
# )
# """Blosum50 scoring matrix."""


# Why are they different?

alias BLOSUM50= InlineArray[Int8, 576]( # copied from parasail
# A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X   * */
   5, -2, -1, -2, -1, -1, -1,  0, -2, -1, -2, -1, -1, -3, -1,  1,  0, -3, -2,  0, -2, -1, -1, -5,
  -2,  7, -1, -2, -4,  1,  0, -3,  0, -4, -3,  3, -2, -3, -3, -1, -1, -3, -1, -3, -1,  0, -1, -5,
  -1, -1,  7,  2, -2,  0,  0,  0,  1, -3, -4,  0, -2, -4, -2,  1,  0, -4, -2, -3,  4,  0, -1, -5,
  -2, -2,  2,  8, -4,  0,  2, -1, -1, -4, -4, -1, -4, -5, -1,  0, -1, -5, -3, -4,  5,  1, -1, -5,
  -1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1, -3, -3, -2, -5,
  -1,  1,  0,  0, -3,  7,  2, -2,  1, -3, -2,  2,  0, -4, -1,  0, -1, -1, -1, -3,  0,  4, -1, -5,
  -1,  0,  0,  2, -3,  2,  6, -3,  0, -4, -3,  1, -2, -3, -1, -1, -1, -3, -2, -3,  1,  5, -1, -5,
   0, -3,  0, -1, -3, -2, -3,  8, -2, -4, -4, -2, -3, -4, -2,  0, -2, -3, -3, -4, -1, -2, -2, -5,
  -2,  0,  1, -1, -3,  1,  0, -2, 10, -4, -3,  0, -1, -1, -2, -1, -2, -3,  2, -4,  0,  0, -1, -5,
  -1, -4, -3, -4, -2, -3, -4, -4, -4,  5,  2, -3,  2,  0, -3, -3, -1, -3, -1,  4, -4, -3, -1, -5,
  -2, -3, -4, -4, -2, -2, -3, -4, -3,  2,  5, -3,  3,  1, -4, -3, -1, -2, -1,  1, -4, -3, -1, -5,
  -1,  3,  0, -1, -3,  2,  1, -2,  0, -3, -3,  6, -2, -4, -1,  0, -1, -3, -2, -3,  0,  1, -1, -5,
  -1, -2, -2, -4, -2,  0, -2, -3, -1,  2,  3, -2,  7,  0, -3, -2, -1, -1,  0,  1, -3, -1, -1, -5,
  -3, -3, -4, -5, -2, -4, -3, -4, -1,  0,  1, -4,  0,  8, -4, -3, -2,  1,  4, -1, -4, -4, -2, -5,
  -1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3, -2, -1, -2, -5,
   1, -1,  1,  0, -1,  0, -1,  0, -1, -3, -3,  0, -2, -3, -1,  5,  2, -4, -2, -2,  0,  0, -1, -5,
   0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  2,  5, -3, -2,  0,  0, -1,  0, -5,
  -3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1,  1, -4, -4, -3, 15,  2, -3, -5, -2, -3, -5,
  -2, -1, -2, -3, -3, -1, -2, -3,  2, -1, -1, -2,  0,  4, -3, -2, -2,  2,  8, -1, -3, -2, -1, -5,
   0, -3, -3, -4, -1, -3, -3, -4, -4,  4,  1, -3,  1, -1, -3, -2,  0, -3, -1,  5, -4, -3, -1, -5,
  -2, -1,  4,  5, -3,  0,  1, -1,  0, -4, -4,  0, -3, -4, -2,  0,  0, -5, -3, -4,  5,  2, -1, -5,
  -1,  0,  0,  1, -3,  4,  5, -2,  0, -3, -3,  1, -1, -4, -1,  0, -1, -2, -2, -3,  2,  5, -1, -5,
  -1, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -1, -1, -2, -2, -1,  0, -3, -1, -1, -1, -1, -1, -5,
  -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,  1
)

alias NUM_TO_NT = InlineArray[UInt8, 5](
    ord("A"), ord("C"), ord("G"), ord("T"), ord("N")
)
"""Table to convert an Int8 to the ascii value of a nucleotide"""

alias NT_TO_NUM = InlineArray[UInt8, 128](
		4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
		4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
		4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
		4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
		4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
		4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
		4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
		4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
)
"""Table used to transform nucleotide letters into numbers."""
# fmt: on


@always_inline
fn aa_to_num(owned seq: List[Int8]) -> List[UInt8]:
    var out = List[UInt8](capacity=len(seq))
    for value in seq:
        out.append(AA_TO_NUM[Int(value[])])
    return out


fn aa_to_num(owned seq: List[UInt8]) -> List[UInt8]:
    for value in seq:
        value[] = AA_TO_NUM[Int(value[])]
    return seq


@always_inline
fn num_to_aa(mut seq: List[UInt8]):
    for value in seq:
        value[] = NUM_TO_AA[Int(value[])]


@always_inline
fn nt_to_num(owned seq: List[Int8]) -> List[UInt8]:
    var out = List[UInt8](capacity=len(seq))
    for value in seq:
        out.append(NT_TO_NUM[Int(value[])])
    return out


fn nt_to_num(owned seq: List[UInt8]) -> List[UInt8]:
    for value in seq:
        value[] = NT_TO_NUM[Int(value[])]
    return seq


@always_inline
fn num_to_nt(mut seq: List[UInt8]):
    for value in seq:
        value[] = NUM_TO_NT[Int(value[])]


@value
struct ScoringMatrix:
    # TODO: force this to be an inline list that can be copied onto the profile?
    var values: List[Int8]
    var size: UInt32  # The number of values represented (which is the length of an edge of the square matrix)

    fn set_last_row_to_value(mut self, value: Int8 = 2):
        for i in range((self.size - 1) * self.size, len(self.values)):
            self.values[i] = value

    @staticmethod
    fn blosm50() -> Self:
        var size = sqrt(len(BLOSUM50))
        var values = List(Span(BLOSUM50))
        return Self(values, size)

    @staticmethod
    fn default_matrix(
        size: UInt32, matched: Int8 = 2, mismatched: Int8 = -2
    ) -> Self:
        var values = List[Int8](capacity=Int(size * size))
        for _ in range(0, Int(size * size)):
            values.append(0)
        for i in range(0, size):
            for j in range(0, size):
                if i == j:
                    values[i * size + j] = matched  # match
                else:
                    values[i * size + j] = mismatched  # Mismatch
        return Self(values, size)

    @staticmethod
    fn actg_default_matrix() -> Self:
        return Self.default_matrix(4)

    @staticmethod
    fn actgn_default_matrix() -> Self:
        return Self.default_matrix(5)

    @staticmethod
    fn all_ascii_default_matrix() -> Self:
        return Self.default_matrix(256)

    fn get(read self, i: Int, j: Int) -> Int8:
        return self.values[i * self.size + j]


@value
struct Cigar:
    var seq: List[UInt32]


@value
struct ReferenceDirection:
    """Direction of the reference sequence."""

    var value: UInt8
    alias Forward = Self(0)
    alias Reverse = Self(1)

    fn __eq__(read self, read other: Self) -> Bool:
        return self.value == other.value


@value
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


@value
struct ProfileVectors[dt: DType, width: Int]:
    var pv_h_store: List[SIMD[dt, width]]
    var pv_h_load: List[SIMD[dt, width]]
    var pv_e: List[SIMD[dt, width]]
    var pv_h_max: List[SIMD[dt, width]]
    var zero: SIMD[dt, width]
    var segment_length: Int32
    var query_len: Int32

    fn __init__(out self, query_len: Int32):
        var segment_length = (query_len + width - 1) // width

        var zero = SIMD[dt, width](0)
        # Stores the H values (scores) for the current row of the dynamic programming matrix
        var pv_h_store = List[SIMD[dt, width]](
            capacity=Int(segment_length)
        )  # aka: pvHStore
        # Contains scores from the previous row that will be loaded for calculation
        var pv_h_load = List[SIMD[dt, width]](
            capacity=Int(segment_length)
        )  # aka: pvHLoad
        # Tracks scores for alignments that end with gaps in the query seq (horizontal gaps in visualization)
        var pv_e = List[SIMD[dt, width]](
            capacity=Int(segment_length)
        )  # aka: pvE
        # Stores the max scores seen in each column for traceback
        var pv_h_max = List[SIMD[dt, width]](
            capacity=Int(segment_length)
        )  # aka: pvHmax

        # init all of the above to zeros
        memset_zero(pv_h_store.unsafe_ptr(), Int(segment_length))
        memset_zero(pv_h_load.unsafe_ptr(), Int(segment_length))
        memset_zero(pv_e.unsafe_ptr(), Int(segment_length))
        memset_zero(pv_h_max.unsafe_ptr(), Int(segment_length))

        self.pv_h_store = pv_h_store
        self.pv_h_load = pv_h_load
        self.pv_e = pv_e
        self.pv_h_max = pv_h_max
        self.zero = zero
        self.segment_length = segment_length
        self.query_len = query_len

    fn zero_out(mut self):
        memset_zero(self.pv_h_store.unsafe_ptr(), Int(self.segment_length))
        memset_zero(self.pv_h_load.unsafe_ptr(), Int(self.segment_length))
        memset_zero(self.pv_e.unsafe_ptr(), Int(self.segment_length))
        memset_zero(self.pv_h_max.unsafe_ptr(), Int(self.segment_length))


@value
struct Profile:
    alias ByteVProfile = List[SIMD[DType.uint8, SIMD_U8_WIDTH]]
    alias WordVProfile = List[SIMD[DType.uint16, SIMD_U16_WIDTH]]

    var profile_byte: Optional[Self.ByteVProfile]
    var profile_word: Optional[Self.WordVProfile]
    var byte_vectors: ProfileVectors[DType.uint8, SIMD_U8_WIDTH]
    var word_vectors: ProfileVectors[DType.uint16, SIMD_U16_WIDTH]
    var query_len: Int32
    var alphabet_size: UInt32
    var bias: UInt8

    fn __init__(
        out self,
        query: Span[UInt8],
        read score_matrix: ScoringMatrix,
        score_size: ScoreSize,
    ):
        """Query is expected to be converted to range of [0:score_matrix.size].

        i.e. ACTG should be 0, 1, 2, 3
        """
        # print("Intializing a profile for a query of len", len(query))
        var profile_byte: Optional[Self.ByteVProfile] = None
        var profile_word: Optional[Self.WordVProfile] = None
        var bias: UInt8 = 0
        if score_size == ScoreSize.Byte or score_size == ScoreSize.Adaptive:
            # print("Score size is byte")
            # print(
            #     len(score_matrix.values)
            # )  # Problem is around here, score matrix is garbage
            # Find the bias to use in the substitution matrix
            # The bias will be smallest value in the scoring matrix
            var bias_tmp: Int8 = 0
            for i in range(0, len(score_matrix.values)):
                if score_matrix.values[i] < bias_tmp:
                    bias_tmp = score_matrix.values[i]
            bias = abs(bias_tmp).cast[DType.uint8]()
            # print("Calling generate profile")
            profile_byte = Self.generate_query_profile[
                DType.uint8, SIMD_U8_WIDTH
            ](query, score_matrix, bias)

        if score_size == ScoreSize.Word or score_size == ScoreSize.Adaptive:
            # print("Score size is adaptive")
            # Find the bias to use in the substitution matrix
            # The bias will be smallest value in the scoring matrix
            var bias_tmp: Int8 = 0
            for i in range(0, len(score_matrix.values)):
                if score_matrix.values[i] < bias_tmp:
                    bias_tmp = score_matrix.values[i]
            bias = abs(bias_tmp).cast[DType.uint8]()
            profile_word = Self.generate_query_profile[
                DType.uint16, SIMD_U16_WIDTH
            ](query, score_matrix, bias)

        return Self(
            profile_byte,
            profile_word,
            ProfileVectors[DType.uint8, SIMD_U8_WIDTH](len(query)),
            ProfileVectors[DType.uint16, SIMD_U16_WIDTH](len(query)),
            len(query),
            score_matrix.size,
            bias,
        )

    @staticmethod
    fn generate_query_profile[
        T: DType, size: Int = simdwidthof[T]()
    ](
        query: Span[UInt8], read score_matrix: ScoringMatrix, bias: UInt8
    ) -> List[SIMD[T, size]]:
        """Divide the query into segments."""
        # print("Generating query profile")
        var segment_length = (len(query) + size - 1) // size
        # for i in range(0, len(score_matrix.values)):
        #     print(score_matrix.values[i])
        var profile = List[SIMD[T, size]](
            capacity=Int(score_matrix.size * segment_length)
        )
        for _ in range(0, profile.capacity):
            profile.append(SIMD[T, size](0))

        # Generate query profile and rearrange query sequence and calculate the weight of match/mismatch
        var t_idx = 0
        for nt in range(0, score_matrix.size):
            for i in range(0, segment_length):
                var j = i
                for segment_idx in range(0, size):
                    # if j >= len(query):
                    #     profile[t_idx][segment_idx] = bias.cast[T]()
                    # else:
                    #     profile[t_idx][segment_idx] = (
                    #         score_matrix.get(nt, Int(query[j]))
                    #         + bias.cast[DType.int8]()
                    #     ).cast[T]()
                    # print(profile[t_idx][segment_idx])
                    # ^ non-ternary version
                    profile[t_idx][segment_idx] = (
                        bias if j
                        >= len(query) else (
                            score_matrix.get(nt, Int(query[j]))
                            + bias.cast[DType.int8]()
                        ).cast[DType.uint8]()
                    ).cast[T]()
                    j += segment_length
                t_idx += 1
        return profile


@value
@register_passable("trivial")
struct Alignment:
    var score1: UInt16
    var score2: UInt16
    var ref_begin1: Int32
    var ref_end1: Int32
    var read_begin1: Int32
    var read_end1: Int32
    var ref_end2: Int32


@value
@register_passable("trivial")
struct AlignmentResult:
    var best: AlignmentEnd
    var second_best: AlignmentEnd


@value
@register_passable("trivial")
struct AlignmentEnd:
    var score: UInt16
    var reference: Int32  # 0-based
    var query: Int32  # alignment ending position on query, 0-based


@always_inline
fn saturating_sub[
    data: DType, width: Int
](lhs: SIMD[data, width], rhs: SIMD[data, width]) -> SIMD[data, width]:
    """Saturating SIMD subtraction.

    https://stackoverflow.com/questions/33481295/saturating-subtract-add-for-unsigned-bytes
    """
    constrained[data.is_unsigned()]()
    var resp = lhs - rhs
    resp &= -(resp <= lhs).cast[data]()
    return resp


@always_inline
fn saturating_add[
    data: DType, width: Int
](lhs: SIMD[data, width], rhs: SIMD[data, width]) -> SIMD[data, width]:
    """Saturating SIMD subtraction.

    https://stackoverflow.com/questions/33481295/saturating-subtract-add-for-unsigned-bytes
    """
    constrained[data.is_unsigned()]()
    var resp = lhs + rhs
    resp |= -(resp < lhs).cast[data]()
    return resp


fn ssw_align(
    mut profile: Profile,
    read matrix: ScoringMatrix,
    reference: Span[UInt8],
    query: Span[UInt8],
    mut reverse_profile: Profile,
    *,
    gap_open_penalty: UInt8 = 3,
    gap_extension_penalty: UInt8 = 1,
    return_only_alignment_end: Bool = False,
    mask_length: Int32 = 15,  # for second best score
) -> Optional[Alignment]:
    # Find the alignment scores and ending positions
    var bests: AlignmentResult

    var used_word = False

    if profile.profile_byte:
        # print("Doing Byte SW")
        bests = sw[DType.uint8, SIMD_U8_WIDTH](
            reference,
            ReferenceDirection.Forward,
            profile.query_len,
            gap_open_penalty,
            gap_extension_penalty,
            profile.profile_byte.value(),
            profile.byte_vectors,
            -1,
            profile.bias,
            mask_length,
        )
        if profile.profile_word and bests.best.score == 255:
            bests = sw[DType.uint16, SIMD_U16_WIDTH](
                reference,
                ReferenceDirection.Forward,
                profile.query_len,
                gap_open_penalty.cast[DType.uint16](),
                gap_extension_penalty.cast[DType.uint16](),
                profile.profile_word.value(),
                profile.word_vectors,
                -1,
                profile.bias.cast[DType.uint16](),
                mask_length,
            )
            bests.best.score -= profile.bias.cast[DType.uint16]()
            bests.second_best.score -= profile.bias.cast[DType.uint16]()
            used_word = True
        elif bests.best.score == 255:
            print(
                "Please overflow in alignments detected, please provide a"
                " larger query profile"
            )
            return None

    elif profile.profile_word:
        bests = sw[DType.uint16, SIMD_U16_WIDTH](
            reference,
            ReferenceDirection.Forward,
            profile.query_len,
            gap_open_penalty.cast[DType.uint16](),
            gap_extension_penalty.cast[DType.uint16](),
            profile.profile_word.value(),
            profile.word_vectors,
            -1,
            profile.bias.cast[DType.uint16](),
            mask_length,
        )
        bests.best.score -= profile.bias.cast[DType.uint16]()
        bests.second_best.score -= profile.bias.cast[DType.uint16]()

        used_word = True
    else:
        print("Failed to provide a valid query profile")
        return None

    if bests.best.score <= 0:
        return None

    var score1 = bests.best.score
    var ref_end1 = bests.best.reference
    var read_end1 = bests.best.query

    var score2: UInt16 = 0
    var ref_end2: Int32 = -1

    if mask_length >= 15:
        score2 = bests.second_best.score
        ref_end2 = bests.second_best.reference

    if return_only_alignment_end or ref_end1 <= 0 or read_end1 <= 0:
        return Alignment(
            score1=score1,
            score2=score2,
            ref_begin1=-1,
            ref_end1=ref_end1,
            read_begin1=-1,
            read_end1=read_end1,
            ref_end2=ref_end2,
        )

    # Get the start position
    var bests_rev: AlignmentResult
    # Reverse the query sequence and truncate it
    # print("read_end1", read_end1)
    # print("Reverse and truncate query")
    # print("was len", len(profile.query))
    # var query_reverse = List[UInt8](capacity=Int(read_end1 + 1))
    # var count = 0
    # for nt in query.__reversed__():
    #     if count == Int(read_end1 + 1):
    #         break
    #     query_reverse.append(nt[])
    #     count += 1
    # # print("now ", len(query_reverse))
    # # print("reflen: ", len(reference[: Int(ref_end1 + 1)]))

    # # var query_reverse_truncated = query_reverse[0: Int(read_end1)+1]
    # if not used_word:
    #     var profile = Profile(query_reverse, matrix, ScoreSize.Byte)
    #     # print("Running rev align")
    #     bests_rev = sw[DType.uint8, SIMD_U8_WIDTH](
    #         reference[: Int(ref_end1 + 1)],
    #         ReferenceDirection.Reverse,
    #         len(query_reverse),
    #         gap_open_penalty,
    #         gap_extension_penalty,
    #         profile.profile_byte.value(),
    #         -1,
    #         profile.bias,
    #         mask_length,
    #     )
    # else:
    #     var profile = Profile(query_reverse, matrix, ScoreSize.Word)
    #     bests_rev = sw[DType.uint16, SIMD_U16_WIDTH](
    #         reference[: Int(ref_end1 + 1)],
    #         ReferenceDirection.Reverse,
    #         len(query_reverse),
    #         gap_open_penalty.cast[DType.uint16](),
    #         gap_extension_penalty.cast[DType.uint16](),
    #         profile.profile_word.value(),
    #         -1,
    #         profile.bias.cast[DType.uint16](),
    #         mask_length,
    #     )
    if not used_word:
        # print("Running rev align")
        bests_rev = sw[DType.uint8, SIMD_U8_WIDTH](
            reference,
            ReferenceDirection.Reverse,
            reverse_profile.query_len,
            gap_open_penalty,
            gap_extension_penalty,
            reverse_profile.profile_byte.value(),
            reverse_profile.byte_vectors,
            -1,
            reverse_profile.bias,
            mask_length,
        )
    else:
        bests_rev = sw[DType.uint16, SIMD_U16_WIDTH](
            reference,
            ReferenceDirection.Reverse,
            reverse_profile.query_len,
            gap_open_penalty.cast[DType.uint16](),
            gap_extension_penalty.cast[DType.uint16](),
            reverse_profile.profile_word.value(),
            reverse_profile.word_vectors,
            -1,
            reverse_profile.bias.cast[DType.uint16](),
            mask_length,
        )
    var ref_begin1 = bests_rev.best.reference
    var read_begin1 = read_end1 - bests_rev.best.query
    # if fwd score  > rev score, we might be missing something?

    # Skipping CIGAR for now
    return Alignment(
        score1=score1,
        score2=score2,
        ref_begin1=ref_begin1,
        ref_end1=ref_end1,
        read_begin1=read_begin1,
        read_end1=read_end1,
        ref_end2=ref_end2,
    )

    # return Alignment(
    #     score1=score1,
    #     score2=score2,
    #     ref_begin1=-1,
    #     ref_end1=ref_end1,
    #     read_begin1=-1,
    #     read_end1=read_end1,
    #     ref_end2=ref_end2,
    # )


@export
fn sw[
    dt: DType, width: Int
](
    reference: Span[UInt8],
    reference_direction: ReferenceDirection,
    query_len: Int32,
    gap_open_penalty: SIMD[dt, 1],
    gap_extension_penalty: SIMD[dt, 1],
    profile: Span[SIMD[dt, width]],
    mut p_vecs: ProfileVectors[dt, width],
    terminate: SIMD[dt, 1],
    bias: SIMD[dt, 1],
    mask_length: Int32,
) -> AlignmentResult:
    """Smith-Waterman local alignment.

    Arguments:
        terminate: The best alignment score, used to terminate the matrix calc when locating the alignment beginning point. If this score is set to 0, it will not be used.

    """
    p_vecs.zero_out()
    var max_score = UInt8(0).cast[dt]()
    var end_query: Int32 = query_len - 1
    var end_reference: Int32 = -1  # 0 based best alignment ending point; initialized as isn't aligned -1
    # var segment_length = (query_len + width - 1) // width
    var segment_length = p_vecs.segment_length
    assume(segment_length > 0)

    # Note:
    # H - Score for match / mismatch (diagonal move)
    # E - Score for gap in query (horizontal move)
    # F - Score for gap in reference (vertical move)
    # print("Segment length:", segment_length)
    # print("RefLen: ", len(reference))
    # print("querylen: ", query_len)
    # print("SIMD DType", dt)
    # print("SIMD WIDTH", width)
    # print("bias", bias)
    # for i in range(
    #     0, ScoringMatrix.blosm50().size
    # ):  # TODO: hardcoded, should be length alphabet
    #     print(chr(Int(NUM_TO_AA[i])), ": ", sep="", end="")
    #     for j in range(0, segment_length):
    #         print(profile[i * segment_length + j], ", ", end="")
    #     print()
    # print("Initializing")

    # List to record the largest score of each reference position
    var max_column = List[SIMD[dt, 1]](capacity=len(reference))
    # List to record the alignment query ending position of the largest score of each reference position
    var end_query_column = List[Int32](capacity=len(reference))
    for _ in range(0, len(reference)):
        max_column.append(UInt8(0).cast[dt]())
        end_query_column.append(0)

    var zero = SIMD[dt, width](0)

    var v_gap_open = SIMD[dt, width](gap_open_penalty)  # aka: vGap0
    var v_gap_ext = SIMD[dt, width](gap_extension_penalty)  # aka: vGapE
    var v_bias = SIMD[dt, width](bias)  # aka: vBias
    # Trace the highest scro of the whole SW matrix
    var v_max_score = zero  # aka: vMaxScore
    # Trace the highest score till the previous column
    var v_max_mark = zero  # aka: vMaxMark

    var edge: Int32 = 0
    var begin: Int32 = 0
    var end: Int32 = len(reference)
    var step: Int32 = 1

    # Outer loop to process the reference sequence
    if reference_direction == ReferenceDirection.Reverse:
        begin = len(reference) - 1
        end = -1
        step = -1
    # print("Done with init")
    var i = begin
    while likely(i != end):
        assume(i >= 0)
        # print("Outer loop:", i, "-", chr(Int(NUM_TO_AA[Int(reference[i])])))
        # Initialize to 0, any errors in vH will be corrected in lazy_f
        var e = zero
        # Represents scores for alignments that end with gaps in the reference seq
        var v_f = zero  # aka: vF
        # the max score in the current column
        var v_max_column = zero  # aka: vMaxColumn
        # The score value currently being calculated
        var v_h = p_vecs.pv_h_store[segment_length - 1]  # aka: vH
        v_h = v_h.shift_right[1]()

        # Select the right vector from the query profile
        var profile_idx = reference[i].cast[DType.int32]() * segment_length

        # Swap the two score buffers
        swap(p_vecs.pv_h_load, p_vecs.pv_h_store)
        # print("vH State:", v_h)
        # print("pvHLoad State:", end="")
        # for i in range(0, segment_length):
        #     print(p_vecs.pv_h_load[i], ", ", end="")
        # print()

        # print("pvHStore State:", end="")
        # for i in range(0, segment_length):
        #     print(p_vecs.pv_h_store[i], ", ", end="")
        # print()

        # print("pvE State:", end="")
        # for i in range(0, segment_length):
        #     print(p_vecs.pv_e[i], ", ", end="")
        # print()

        # print("pvHMax State:", end="")
        # for i in range(0, segment_length):
        #     print(p_vecs.pv_h_max[i], ", ", end="")
        # print()

        # print("max_column State:", end="")
        # for i in range(0, len(reference)):
        #     print(max_column[i], ", ", end="")
        # print()

        # print("end_read_column State:", end="")
        # for i in range(0, len(reference)):
        #     print(end_query_column[i], ", ", end="")
        # print()

        # Inner loop to process the query sequence
        for j in range(0, segment_length):
            # print("\tInner loop for query sequence, checcking segment:", j)
            # print("\tSegement J's query profile:", profile[profile_idx + j])
            # Add profile score to
            v_h = saturating_add(v_h, profile[profile_idx + j])
            v_h = saturating_sub(v_h, v_bias)  # adjust for bias
            # print(
            #     "\t\tState of vH after adding profile and subtracting bias: ",
            #     v_h,
            # )

            # Get max from current_cell_score, horizontal gap, and vertical gap score
            # print("\t\t", "Get max from current cell / hgap / vgap")
            e = p_vecs.pv_e[j]
            v_h = max(v_h, e)
            v_h = max(v_h, v_f)
            v_max_column = max(v_max_column, v_h)
            # print("\t\te State: ", e)
            # print("\t\tvMaxColumn State: ", v_max_column)
            # print("\t\tvH State after getting max: ", v_h)

            # Save current_cell_score
            p_vecs.pv_h_store[j] = v_h

            # update vE
            v_h = saturating_sub(v_h, v_gap_open)
            e = saturating_sub(e, v_gap_ext)
            e = max(e, v_h)
            p_vecs.pv_e[j] = e
            # print("\t\te State after update: ", e)

            # update vF
            v_f = saturating_sub(v_f, v_gap_ext)
            v_f = max(v_f, v_h)
            # print("\t\tvF State after update:", v_f)

            # load the next vH
            v_h = p_vecs.pv_h_load[j]
            # print("\t\tnext vH:", v_h)

        # print("\tStarting LazyF")
        # Lazy_F loop, disallows adjacent insertion and then deletion from SWPS3
        # Possible speedup - check if v_f has any updates to start with
        # var break_out = (v_f == zero).reduce_and()
        # var k = 0
        # while not break_out and k < width:
        # k += 1
        var break_out = False
        for _k in range(0, width):
            v_f = v_f.shift_right[1]()
            # print("\tLeft Shift vF:", v_f)
            # print("\tWalking Segments")
            for j in range(0, segment_length):
                v_h = p_vecs.pv_h_store[j]
                v_h = max(v_h, v_f)
                # print("\t\tvH after left shift and max vF", v_h)
                v_max_column = max(v_max_column, v_h)
                # print("\t\t vMaxColumn State:", v_max_column)
                p_vecs.pv_h_store[j] = v_h

                v_h = saturating_sub(v_h, v_gap_open)
                v_f = saturating_sub(v_f, v_gap_ext)
                # print("\t\tnew vH: ", v_h)
                # print("\t\tnew vF: ", v_f)

                # Early termination check
                var v_temp = saturating_sub(v_f, v_h)
                var packed = v_temp == zero
                if unlikely(packed.reduce_and()):
                    # print("\t\tCan terminate early")
                    break_out = True
                    break
            if break_out:
                break
        # print("\t Done with main loops")

        # Check for new max score
        v_max_score = max(v_max_score, v_max_column)
        var equal_vector = v_max_mark == v_max_score
        if not equal_vector.reduce_and():
            # find max score in vector
            var temp = v_max_score.reduce_max()
            v_max_mark = v_max_score

            if likely(temp > max_score):
                max_score = temp
                if (max_score + bias) >= SIMD[dt, 1].MAX:
                    # print("OVERFLOW")
                    break
                end_reference = i

                # Store the column with the highest alignment score in order to trace teh alignment ending position on the query
                for j in range(0, segment_length):
                    p_vecs.pv_h_max[j] = p_vecs.pv_h_store[j]

        # Record the max score of current column
        max_column[i] = v_max_column.reduce_max()
        if unlikely(max_column[i] == terminate):
            # TODO: What's this doing?
            # print("terminate early")
            break

        # Increment the while loop
        i += step

    # Trace the alignment ending position on query
    # var column_len = segment_length * SIMD_U8_WIDTH
    for i in range(0, segment_length):
        for j in range(0, width):
            if p_vecs.pv_h_max[i][j] == max_score:
                var temp = i + j * segment_length
                if temp < Int(end_query):
                    end_query = temp

        # if pv_h_max[i / SIMD_U8_WIDTH][i % SIMD_U8_WIDTH] == max_score:
        #     var temp = i // SIMD_U8_WIDTH + (i % 16) * segment_length
        #     if temp < end_query:
        #         end_query = temp

    # print("pvHMax State:", end="")
    # for i in range(0, segment_length):
    #     print(p_vecs.pv_h_max[i], ", ", end="")
    # print()

    # print("max_column State:", end="")
    # for i in range(0, len(reference)):
    #     print(max_column[i], ", ", end="")
    # print()

    # print("end_read_column State:", end="")
    # for i in range(0, len(reference)):
    #     print(end_query_column[i], ", ", end="")
    # print()
    # Find the most possible 2nd alignment
    var score_0 = max_score + bias if max_score + bias >= 255 else max_score
    var bests = AlignmentResult(
        AlignmentEnd(score_0.cast[DType.uint16](), end_reference, end_query),
        AlignmentEnd(0, 0, 0),
    )

    # Candidate for SIMD?
    edge = (end_reference - mask_length) if (
        end_reference - mask_length
    ) > 0 else 0
    for i in range(0, edge):
        if max_column[i] > bests.second_best.score.cast[dt]():
            bests.second_best.score = max_column[i].cast[DType.uint16]()
            bests.second_best.reference = i

    edge = (
        len(reference) if (end_reference + mask_length)
        > len(reference) else end_reference + mask_length
    )
    for i in range(edge + 1, len(reference)):
        if max_column[i] > bests.second_best.score.cast[dt]():
            bests.second_best.score = max_column[i].cast[DType.uint16]()
            bests.second_best.reference = i

    # print(
    #     "score:",
    #     bests.best.score,
    #     "ref:",
    #     bests.best.reference,
    #     "read:",
    #     bests.best.query,
    # )
    # print(
    #     "score:",
    #     bests.second_best.score,
    #     "ref:",
    #     bests.second_best.reference,
    #     "read:",
    #     bests.second_best.query,
    # )
    return bests
