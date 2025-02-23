"""Globally align any two sequences."""
from collections import Optional
from collections.string import StringSlice
from memory import Span

from ishlib.matcher.alignment import AlignmentResult


fn needleman_wunsch(
    seq1: Span[UInt8],
    seq2: Span[UInt8],
    *,
    match_score: Int32 = 1,
    mismatch_score: Int32 = -1,
    gap_penalty: Int32 = -1,
    traceback_alignment: Bool = True,
) -> AlignmentResult:
    """Needleman-Wunsch algorithm for global sequence alignment.

    ```mojo
    from testing import assert_equal
    from ishlib.matcher.alignment.global_align import needleman_wunsch

    var result = needleman_wunsch("ATTGCC".as_bytes(), "ATGC".as_bytes())
    assert_equal(result.score, 2)
    result = needleman_wunsch("ATTGCC".as_bytes(), "AAAA".as_bytes())
    assert_equal(result.score, -4)
    result = needleman_wunsch("GATTACA".as_bytes(), "GCATGCU".as_bytes())
    assert_equal(result.alignment1.value(), "G-ATTACA")
    assert_equal(result.alignment2.value(), "GCA-TGCU")
    assert_equal(result.score, 0)
    ```
    """
    # Create the matrix
    var rows = len(seq1) + 1
    var cols = len(seq2) + 1
    var score_matrix = List[List[Int32]](capacity=rows)
    for _ in range(0, rows):
        var row = List[Int32](capacity=cols)
        for _ in range(0, cols):
            row.append(0)
        score_matrix.append(row)

    # Initialize the first row and column
    for i in range(0, rows):
        score_matrix[i][0] = gap_penalty * i
    for j in range(0, cols):
        score_matrix[0][j] = gap_penalty * j

    # Fill the matrix
    for i in range(1, rows):
        for j in range(1, cols):
            var matched = score_matrix[i - 1][j - 1] + (
                match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score
            )
            var delete = score_matrix[i - 1][j] + gap_penalty
            var insert = score_matrix[i][j - 1] + gap_penalty
            score_matrix[i][j] = max(max(matched, delete), insert)

    if not traceback_alignment:
        return AlignmentResult(
            score_matrix[rows - 1][cols - 1], None, None, None
        )

    # Traceback to find the optimal alignment
    var align1 = List[UInt8]()
    var align2 = List[UInt8]()
    var i = rows - 1
    var j = cols - 1

    var gap_char: UInt8 = ord("-")

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            var matched = score_matrix[i - 1][j - 1] + (
                match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score
            )
            if score_matrix[i][j] == matched:
                align1.append(seq1[i - 1])
                align2.append(seq2[j - 1])
                i -= 1
                j -= 1
                continue

        if i > 0 and score_matrix[i][j] == score_matrix[i - 1][j] + gap_penalty:
            align1.append(seq1[i - 1])
            align2.append(gap_char)
            i -= 1
        else:
            align1.append(gap_char)
            align2.append(seq2[j - 1])
            j -= 1

    # Reverse the alignments since we built them backwards
    var final_align1 = List[UInt8]()
    var final_align2 = List[UInt8]()

    for i in range(len(align1) - 1, -1, -1):
        final_align1.append(align1[i])
        final_align2.append(align2[i])

    return AlignmentResult(
        score_matrix[rows - 1][cols - 1],
        String(StringSlice(unsafe_from_utf8=final_align1)),
        String(StringSlice(unsafe_from_utf8=final_align2)),
        (0, 0),
    )
