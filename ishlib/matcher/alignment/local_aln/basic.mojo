from memory import Span

from ishlib.matcher.alignment import AlignmentResult, TargetSpan

alias DIAG = 1
alias UP = 2
alias LEFT = 3


fn smith_waterman(
    seq1: Span[UInt8],
    seq2: Span[UInt8],
    *,
    match_score: Int16 = 2,
    mismatch_score: Int16 = -1,
    gap_penalty: Int16 = -1,
    traceback_alignment: Bool = True,
) -> AlignmentResult:
    """Smith-Waterman algorithm for local sequence alignment.

    ```mojo
    from testing import assert_equal
    from ishlib.matcher.alignment.local_aln.basic import smith_waterman

    var result = smith_waterman("GATTACA".as_bytes(), "GCATGC".as_bytes())
    assert_equal(result.score, 5)
    assert_equal(result.alignment1.value(), "G-AT")
    assert_equal(result.alignment2.value(), "GCAT")

    ```
    """
    # Create the scoring and traceback matrix
    var rows = len(seq1) + 1
    var cols = len(seq2) + 1

    var score_matrix = List[List[Int16]](capacity=rows)
    var traceback_matrix = List[List[Int16]](capacity=rows)
    for _ in range(0, rows):
        var score_row = List[Int16](capacity=cols)
        var traceback_row = List[Int16](capacity=cols)
        for _ in range(0, cols):
            score_row.append(0)
            traceback_row.append(0)
        score_matrix.append(score_row)
        traceback_matrix.append(traceback_row)

    var max_score: Int16 = 0
    var max_pos = (0, 0)

    # Populate the scoring matrix
    for i in range(1, rows):
        for j in range(1, cols):
            # Calculate the match/mimatch score
            var diag_score: Int16 = 0
            if seq1[i - 1] == seq2[j - 1]:
                diag_score = score_matrix[i - 1][j - 1] + match_score
            else:
                diag_score = score_matrix[i - 1][j - 1] + mismatch_score

            # Calculate gap scores
            var up_score = score_matrix[i - 1][j] + gap_penalty
            var left_score = score_matrix[i][j - 1] + gap_penalty

            # Take the maxium of the possible scores, capping at 0
            score_matrix[i][j] = max(
                0, max(diag_score, max(up_score, left_score))
            )

            # Update traceback matrix
            if score_matrix[i][j] == diag_score:
                traceback_matrix[i][j] = DIAG
            elif score_matrix[i][j] == up_score:
                traceback_matrix[i][j] = UP
            elif score_matrix[i][j] == left_score:
                traceback_matrix[i][j] = LEFT

            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]
                max_pos = (i, j)

    if not traceback_alignment:
        return AlignmentResult(max_score.cast[DType.int32](), None, None, None)

    # Traceback
    var align1 = List[UInt8]()
    var align2 = List[UInt8]()
    i, j = max_pos
    var end = i

    while i > 0 and j > 0 and score_matrix[i][j] > 0:
        if traceback_matrix[i][j] == DIAG:
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == UP:
            align1.append(seq1[i - 1])
            align2.append(ord("-"))
            i -= 1
        else:  # LEFT
            align1.append(ord("-"))
            align2.append(seq2[j - 1])
            j -= 1
    var start = i

    # reverse the alignment
    var final_align1 = List[UInt8](capacity=len(align1))
    var final_align2 = List[UInt8](capacity=len(align2))
    for i in range(len(align1) - 1, -1, -1):
        final_align1.append(align1[i])
        final_align2.append(align2[i])

    return AlignmentResult(
        max_score.cast[DType.int32](),
        String(StringSlice(unsafe_from_utf8=final_align1)),
        String(StringSlice(unsafe_from_utf8=final_align2)),
        TargetSpan(start, end),
    )
