from collections import InlineArray
from math import sqrt
from memory import UnsafePointer, AddressSpace
from sys.info import alignof

from ishlib.vendor.log import Logger

# fmt: off
alias AA_TO_NUM = InlineArray[UInt8, 256](
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 0,  20, 4,  3,  6,  13, 7,  8,  9,  23, 11, 10, 12, 2,  23,
    14, 5,  1,  15, 16, 23, 19, 17, 22, 18, 21, 23, 23, 23, 23, 23,
    23, 0,  20, 4,  3,  6,  13, 7,  8,  9,  23, 11, 10, 12, 2,  23,
    14, 5,  1,  15, 16, 23, 19, 17, 22, 18, 21, 23, 23, 23, 23, 23,
    # ASCII 128–255: undefined characters → fallback to 23
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23
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

alias BLOSUM62 = InlineArray[Int8, 576]( # Copied from parasail
#  A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X   * */
   4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4,
  -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4,
  -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4,
  -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4,
   0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4,
  -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4,
  -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
   0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4,
  -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4,
  -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4,
  -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4,
  -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4,
  -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4,
  -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4,
  -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4,
   1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4,
   0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4,
  -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4,
  -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4,
   0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4,
  -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4,
  -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
   0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4,
  -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1
)

alias NUM_TO_NT = InlineArray[UInt8, 5](
    ord("A"), ord("C"), ord("G"), ord("T"), ord("N")
)
"""Table to convert an Int8 to the ascii value of a nucleotide"""

alias NT_TO_NUM = InlineArray[UInt8, 256](
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        # ASCII 128–255 → unknown
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
)
"""Table used to transform nucleotide letters into numbers.


Supports ACTGN.
"""

alias ACTGN = InlineArray[Int8, 25](
   # A   C   T   G   N
     2, -2, -2, -2,  2, # A
    -2,  2, -2, -2,  2, # C
    -2, -2,  2, -2,  2, # T
    -2, -2, -2,  2,  2, # G
     2,  2,  2,  2,  2  # N
)

alias ACTGN0 = InlineArray[Int8, 25](
   # A   C   T   G   N
     2, -2, -2, -2,  0, # A
    -2,  2, -2, -2,  0, # C
    -2, -2,  2, -2,  0, # T
    -2, -2, -2,  2,  0, # G
     0,  0,  0,  0,  0  # N
)
# fmt: on


@value
@register_passable
struct MatrixKind(Sized, Stringable, Writable):
    var value: UInt8
    alias ASCII = Self(0)
    alias ACTGN = Self(1)
    alias ACTGN0 = Self(2)
    alias BLOSUM62 = Self(3)

    @staticmethod
    fn from_str(read name: String) raises -> Self:
        if name.lower() == "ascii":
            return Self.ASCII
        elif name.lower() == "actgn":
            return Self.ACTGN
        elif name.lower() == "actgn0":
            return Self.ACTGN0
        elif name.lower() == "blosum62":
            return Self.BLOSUM62
        else:
            raise "Invalid scoring matrix name: " + name

    fn __eq__(read self, read other: Self) -> Bool:
        return self.value == other.value

    fn __str__(read self) -> String:
        var s = String()
        self.write_to(s)
        return s

    fn matrix(read self) -> ScoringMatrix:
        if self == Self.ASCII:
            return ScoringMatrix.all_ascii_default_matrix()
        elif self == Self.ACTGN:
            return ScoringMatrix.actgn_matrix()
        elif self == Self.ACTGN0:
            return ScoringMatrix.actgn0_matrix()
        elif self == Self.BLOSUM62:
            return ScoringMatrix.blosum62()
        else:
            Logger.error(
                "Unsupported scoring matrix kind. Returning ASCII matrix."
            )
            return ScoringMatrix.all_ascii_default_matrix()

    fn write_to[W: Writer](read self, mut writer: W):
        if self == Self.ASCII:
            writer.write("ASCII")
        elif self == Self.ACTGN:
            writer.write("ACTGN")
        elif self == Self.ACTGN0:
            writer.write("ACTGN0")
        elif self == Self.BLOSUM62:
            writer.write("BLOSUM62")
        else:
            writer.write("UNKNOWN")

    fn __len__(read self) -> Int:
        if self == Self.ASCII:
            return 0
        elif self == Self.ACTGN:
            return len(ACTGN)
        elif self == Self.BLOSUM62:
            return len(BLOSUM62)
        else:
            return 0

    fn skip_lookup(read self) -> Bool:
        if self == Self.ASCII:
            return True
        elif self == Self.ACTGN:
            return False
        elif self == Self.ACTGN0:
            return False
        elif self == Self.BLOSUM62:
            return False
        else:
            return True


@value
struct BasicScoringMatrix[
    mut: Bool, //,
    *,
    origin: Origin[mut],
    address_space: AddressSpace = AddressSpace(0),
    alignment: Int = alignof[Int8](),
    no_lookup: Bool = False,
    default_match: Int8 = 2,
    default_mismatch: Int8 = -2,
]:
    """Scoring matrix that allows for easy abstraction over a sequence of bytes.
    """

    var values: UnsafePointer[
        Int8,
        address_space=address_space,
        alignment=alignment,
        mut=mut,
        origin=origin,
    ]
    var len: Int
    var size: Int

    fn __init__(
        out self,
        ptr: UnsafePointer[
            Int8,
            address_space=address_space,
            alignment=alignment,
            mut=mut,
            origin=origin,
        ],
        length: Int,
    ):
        self.values = ptr
        self.len = length
        self.size = sqrt(length)

    fn get(read self, i: Int, j: Int) -> Int8:
        @parameter
        if no_lookup:
            return default_match if i == j else default_mismatch
        else:
            return self.values[i * self.size + j]

    fn __len__(read self) -> Int:
        return self.len


@value
struct ScoringMatrix:
    """The scoring matrix for determining the match/mismatch score between any two values.

    Additionally, the matrix is responsible for knowing what the alphabet it represents is,
    including being able to convert from ascii -> encoding and back.
    """

    # TODO: force this to be an inline list that can be copied onto the profile?
    var values: List[Int8]
    """The square matrix that defines the relationships between elements."""

    var size: UInt32
    """The number of values represented (which is the length of an edge of the square matrix)."""

    var ascii_to_encoding: List[UInt8]
    """This converts ascii to the encoding value. The encoding value will be an "index" in the alphabet."""

    var encoding_to_ascii: List[UInt8]
    """This is the equivalent of the alphabet. Each index is value is the "encoding" and the value at the index is the ASCII value.

    The length of this must be equal to the "size".
    """

    @staticmethod
    fn blosum50() -> Self:
        var size = sqrt(len(BLOSUM50))
        var values = List(Span(BLOSUM50))
        var ascii_to_encoding = List(Span(AA_TO_NUM))
        var encoding_to_ascii = List(Span(NUM_TO_AA))
        return Self(values, size, ascii_to_encoding, encoding_to_ascii)

    @staticmethod
    fn blosum62() -> Self:
        var size = sqrt(len(BLOSUM62))
        var values = List(Span(BLOSUM62))
        var ascii_to_encoding = List(Span(AA_TO_NUM))
        var encoding_to_ascii = List(Span(NUM_TO_AA))
        return Self(values, size, ascii_to_encoding, encoding_to_ascii)

    @staticmethod
    fn actgn_matrix(match_score: Int8 = 2, mismatch_score: Int8 = -2) -> Self:
        var size = sqrt(len(ACTGN))
        var values = List(Span(ACTGN))
        var ascii_to_encoding = List(Span(NT_TO_NUM))
        var encoding_to_ascii = List(Span(NUM_TO_NT))
        return Self(values, size, ascii_to_encoding, encoding_to_ascii)

    @staticmethod
    fn actgn0_matrix(match_score: Int8 = 2, mismatch_score: Int8 = -2) -> Self:
        var size = sqrt(len(ACTGN0))
        var values = List(Span(ACTGN0))
        var ascii_to_encoding = List(Span(NT_TO_NUM))
        var encoding_to_ascii = List(Span(NUM_TO_NT))
        return Self(values, size, ascii_to_encoding, encoding_to_ascii)

    @staticmethod
    fn all_ascii_default_matrix() -> Self:
        # TODO: come up with a zero cost way to handle this for ascii.
        # it's never really converted back and forth where it's being used, so
        # this is just a formality, but it's annoying.
        var values = Self._default_matrix(256)
        var ascii_to_encoding = List[UInt8](capacity=256)
        var encodign_to_acii = List[UInt8](capacity=256)
        for i in range(0, 256):
            ascii_to_encoding.append(i)
            encodign_to_acii.append(i)
        return Self(
            values,
            sqrt(len(values)),
            ascii_to_encoding=ascii_to_encoding,
            encoding_to_ascii=encodign_to_acii,
        )

    @staticmethod
    fn _default_matrix(
        size: UInt32, match_score: Int8 = 2, mismatch_score: Int8 = -2
    ) -> List[Int8]:
        var values = List[Int8](capacity=Int(size * size))
        for _ in range(0, Int(size * size)):
            values.append(0)
        for i in range(0, size):
            for j in range(0, size):
                if i == j:
                    values[i * size + j] = match_score  # match
                else:
                    values[i * size + j] = mismatch_score  # Mismatch
        return values

    fn __len__(read self) -> Int:
        return len(self.values)

    fn get[I: Indexer](read self, i: I, j: I) -> Int8:
        return self.values[Int(i) * self.size + Int(j)]

    fn _set_last_row_to_value(mut self, value: Int8 = 2):
        for i in range((self.size - 1) * self.size, len(self.values)):
            self.values[i] = value

    @always_inline
    fn convert_ascii_to_encoding_and_score(
        read self, seq: Span[UInt8]
    ) -> (List[UInt8], Int):
        """Convert the input seq to the encoding, and also track the optimal alignment score.
        """
        var score = 0
        var out = List[UInt8](capacity=len(seq))
        for value in seq:
            var encoded = self.ascii_to_encoding[Int(value)]
            score += Int(self.get(encoded, encoded))
            out.append(encoded)
        return (out, score)

    @always_inline
    fn convert_ascii_to_encoding[
        origin: MutableOrigin
    ](read self, mut seq: Span[UInt8, origin]):
        for i in range(0, len(seq)):
            seq[i] = self.ascii_to_encoding[Int(seq[i])]

    @always_inline
    fn convert_ascii_to_encoding(read self, seq: Span[UInt8]) -> List[UInt8]:
        var out = List[UInt8](capacity=len(seq))
        for value in seq:
            out.append(self.ascii_to_encoding[Int(value)])
        return out

    @always_inline
    fn convert_ascii_to_encoding(
        read self, owned seq: List[Int8]
    ) -> List[UInt8]:
        var out = List[UInt8](capacity=len(seq))
        for value in seq:
            out.append(self.ascii_to_encoding[Int(value)])
        return out

    @always_inline
    fn convert_ascii_to_encoding(read self, seq: UInt8) -> UInt8:
        return self.ascii_to_encoding[Int(seq)]

    @always_inline
    fn convert_ascii_to_encoding(
        read self, owned seq: List[UInt8]
    ) -> List[UInt8]:
        for ref value in seq:
            value = self.ascii_to_encoding[Int(value)]
        return seq

    @always_inline
    fn convert_encoding_to_ascii(read self, mut seq: List[UInt8]):
        for ref value in seq:
            value = self.encoding_to_ascii[Int(value)]

    @always_inline
    fn convert_encoding_to_ascii(
        read self, read seq: Span[UInt8]
    ) -> List[UInt8]:
        var out = List[UInt8](capacity=len(seq))
        for ref value in seq:
            out.append(self.encoding_to_ascii[Int(value)])
        return out

    @always_inline
    fn convert_encoding_to_ascii(read self, seq: UInt8) -> UInt8:
        return self.encoding_to_ascii[Int(seq)]
