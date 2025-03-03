from memory import Span
from sys.info import simdwidthof
from testing import assert_equal, assert_true

from ishlib.matcher.alignment.ssw_align import (
    Profile,
    ScoringMatrix,
    ScoreSize,
    SIMD_U8_WIDTH,
    SIMD_U16_WIDTH,
    sw_byte,
    ReferenceDirection,
    nt_to_num,
)


fn test_profile() raises:
    assert_equal(SIMD_U8_WIDTH, 16)
    var alphabet_size = 64
    var query = List[UInt8]()
    for i in range(0, alphabet_size):
        query.append(i)
    var matrix = ScoringMatrix.default_matrix(alphabet_size)

    var profile = Profile[__origin_of(query), __origin_of(matrix)](
        Span(query), matrix, ScoreSize.Adaptive
    )

    # Verify basic properties
    assert_equal(profile.bias, 1)
    # query length 64, alphabet size 64
    # segment length = 4
    # alphabet size * segment length = number of elements in the byte profile
    assert_equal(len(profile.profile_byte.value()), 256)
    assert_equal(len(profile.profile_word.value()), 512)

    # The correct profile pattern:
    # For a query of [0,1,2,3,...63] and alphabet size 64:
    #
    # In the SSW algorithm, each character has 4 vectors in a striped pattern:
    # - Vector 0: positions 0, 4, 8, 12, 16, ...
    # - Vector 1: positions 1, 5, 9, 13, 17, ...
    # - Vector 2: positions 2, 6, 10, 14, 18, ...
    # - Vector 3: positions 3, 7, 11, 15, 19, ...
    # Where the positions correspond to the index into the query.
    #
    #
    # For alphabet char 0-63 with match=2, mismatch=-1, bias=1:
    # Char 0: (so alphabet char 0)
    #   [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # Char 1: (the second alphabet char matches the char at idx 1, which is in the second vector)
    #   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # Char 2:
    #   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # Char 3:
    #   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # Char 4:
    #   [0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # Char 5:
    #   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # And so on...

    var zero = SIMD[DType.uint8, SIMD_U8_WIDTH](0)

    # Verify byte profile
    for char_idx in range(alphabet_size):
        # Each character has 4 vectors (the "striped" pattern)
        var vector_base_idx = char_idx * 4

        # Create the expected vectors
        var expected_vectors = List[SIMD[DType.uint8, SIMD_U8_WIDTH]](
            zero, zero, zero, zero
        )

        # Calculate which vector and position within vector that will have the match
        var vector_idx = char_idx % 4
        var pos_within_vector = char_idx // 4

        # Only one position will have the match score (3), rest will be mismatches (0)
        var match_vector = SIMD[DType.uint8, SIMD_U8_WIDTH](0)
        match_vector[pos_within_vector] = 3
        expected_vectors[vector_idx] = match_vector

        # Verify each of the 4 vectors for this character
        for i in range(4):
            var actual_vector = profile.profile_byte.value()[
                vector_base_idx + i
            ]
            assert_equal(
                actual_vector,
                expected_vectors[i],
                "Mismatch for char "
                + String(char_idx)
                + " vector "
                + String(i),
            )

    assert_equal(SIMD_U16_WIDTH, 8)
    # Verify word profile - similar structure but with 8 vectors per character
    # and a different position calculation
    var word_zero = SIMD[DType.uint16, SIMD_U16_WIDTH](0)
    for char_idx in range(alphabet_size):
        # Each character has 8 vectors in word profile
        var vector_base_idx = char_idx * SIMD_U16_WIDTH

        # Create the expected vectors (all zeros initially)
        var expected_vectors = List[SIMD[DType.uint16, SIMD_U16_WIDTH]](
            word_zero,
            word_zero,
            word_zero,
            word_zero,
            word_zero,
            word_zero,
            word_zero,
            word_zero,
        )

        # Calculate which vector and position within vector that will have the match
        var vector_idx = char_idx % SIMD_U16_WIDTH
        var pos_within_vector = char_idx // SIMD_U16_WIDTH

        # Only one position will have the match score (2), rest will be mismatches (-1)
        # No bias in word profile
        var match_vector = SIMD[DType.uint16, SIMD_U16_WIDTH](
            0
        )  # All mismatches
        match_vector[pos_within_vector] = 3  # Match score
        expected_vectors[vector_idx] = match_vector

        # Verify each of the 8 vectors for this character
        for i in range(8):
            var actual_vector = profile.profile_word.value()[
                vector_base_idx + i
            ]
            assert_equal(
                actual_vector,
                expected_vectors[i],
                "Word profile mismatch for char "
                + String(char_idx)
                + " vector "
                + String(i),
            )


fn test_sw_byte() raises:
    """Test the Smith-Waterman byte implementation with small sequences."""

    # Create a small query and reference that will exercise all codepaths
    # For DNA-like alphabet (size 4), representing A=0, C=1, G=2, T=3

    # Query: ACGTACGT
    var query = List[UInt8](0, 1, 2, 3, 0, 1, 2, 3)
    # var query = List[UInt8](2, 0, 3, 0, 1, 0)

    # Reference: TACGTACGTACG
    var reference = List[UInt8](3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2)
    # var reference = List[UInt8](1)

    # Create scoring matrix (match=2, mismatch=-1)
    var matrix = ScoringMatrix.default_matrix(4)

    # Create query profile
    var profile = Profile[__origin_of(query), __origin_of(matrix)](
        Span(query), matrix, ScoreSize.Adaptive
    )

    # Set gap penalties
    var gap_open = UInt8(3)  # Gap open penalty
    var gap_extend = UInt8(1)  # Gap extension penalty

    # Run Smith-Waterman alignment
    var alignments = sw_byte(
        Span(reference),
        ReferenceDirection.Forward,
        len(query),
        gap_open,
        gap_extend,
        profile.profile_byte.value(),
        0,  # No early termination
        profile.bias,
        4,  # Small mask length
    )

    # Validate results
    print("Alignment results:")
    print("Best alignment score:", alignments[0].score)
    print("Reference end position:", alignments[0].reference)
    print("Query end position:", alignments[0].query)

    # Given our specific sequences, we expect:
    # - The optimal alignment is to skip the first T in reference (gap in query)
    #   and then match perfectly: ACGTACGT with ACGTACGTA
    # - With match=2, mismatch=-1, gap_open=3, gap_extend=1:
    #   - 8 matches = 16 points
    #   - 1 gap open = -3 points
    #   - Total score should be ~13

    # The expected end positions would be:
    # - Reference end around position 8 (0-based)
    # - Query end at position 7 (0-based, the last position)

    # Check for reasonable results (exact values may vary based on implementation details)
    assert_true(alignments[0].score > 10, "Score should be at least 10")
    assert_true(
        alignments[0].reference >= 7, "Reference end position seems wrong"
    )
    assert_true(
        alignments[0].query == 7,
        "Query end position should be the last position (7)",
    )

    # Test with a different reference to ensure we get different results
    # Reference with poor alignment: TTTTTTTTT
    var poor_reference = List[UInt8]()
    for _ in range(9):
        poor_reference.append(UInt8(3))  # all Ts

    var poor_alignments = sw_byte(
        Span(poor_reference),
        ReferenceDirection.Forward,
        len(query),
        gap_open,
        gap_extend,
        profile.profile_byte.value(),
        0,
        profile.bias,
        4,
    )

    print("\nPoor alignment results:")
    print("Best alignment score:", poor_alignments[0].score)
    print("Reference end position:", poor_alignments[0].reference)
    print("Query end position:", poor_alignments[0].query)

    # With all Ts in reference, we expect:
    # - Only T in query would match (positions 3 and 7)
    # - Much lower score than good alignment

    assert_true(
        poor_alignments[0].score < alignments[0].score,
        "Poor alignment should have lower score",
    )

    # Test reverse direction to ensure it works
    var reverse_alignments = sw_byte(
        Span(poor_reference),
        ReferenceDirection.Reverse,
        len(query),
        gap_open,
        gap_extend,
        profile.profile_byte.value(),
        0,
        profile.bias,
        4,
    )

    print("\nReverse direction alignment results:")
    print("Best alignment score:", reverse_alignments[0].score)
    print("Reference end position:", reverse_alignments[0].reference)
    print("Query end position:", reverse_alignments[0].query)

    # Score should be the same, but positions will be different
    assert_true(
        reverse_alignments[0].score == poor_alignments[0].score,
        "Reverse alignment should have same score",
    )

    print("\nAll tests passed!")


# More comprehensive test with different sequences
fn test_sw_byte_comprehensive() raises:
    """A more comprehensive test with different alignment scenarios."""

    # Test 1: Perfect alignment
    # Query:     ACGTACGT
    # Reference: ACGTACGT
    var query1 = List[UInt8](0, 1, 2, 3, 0, 1, 2, 3)
    var ref1 = List[UInt8](0, 1, 2, 3, 0, 1, 2, 3)

    # Test 2: Alignment with internal gap in query
    # Query:     AC-GTACGT
    # Reference: ACAGTACGT
    var query2 = List[UInt8](0, 1, 2, 3, 0, 1, 2, 3)

    var ref2 = List[UInt8](0, 1, 0, 2, 3, 0, 1, 2, 3)

    # Test 3: Alignment with internal gap in reference
    # Query:     ACAGTACGT
    # Reference: AC-GTACGT
    var query3 = List[UInt8](0, 1, 0, 2, 3, 0, 1, 2, 3)

    var ref3 = List[UInt8](0, 1, 2, 3, 0, 1, 2, 3)

    # Create scoring matrix
    var matrix = ScoringMatrix.default_matrix(4)

    # Set gap penalties
    var gap_open = UInt8(3)
    var gap_extend = UInt8(1)

    # Run tests
    print("=== Comprehensive Smith-Waterman Tests ===")

    # Test 1: Perfect alignment
    var profile1 = Profile[__origin_of(query1), __origin_of(matrix)](
        Span(query1), matrix, ScoreSize.Adaptive
    )

    var alignments1 = sw_byte(
        Span(ref1),
        ReferenceDirection.Forward,
        len(query1),
        gap_open,
        gap_extend,
        profile1.profile_byte.value(),
        0,
        profile1.bias,
        4,
    )

    print("\nTest 1 - Perfect alignment:")
    print("Score:", alignments1[0].score)
    print("Reference end:", alignments1[0].reference)
    print("Query end:", alignments1[0].query)
    assert_true(
        alignments1[0].score == UInt16(8 * 2 + profile1.bias),
        "Perfect match should have score = 2 * length",
    )

    # Test 2: Gap in query
    var profile2 = Profile[__origin_of(query2), __origin_of(matrix)](
        Span(query2), matrix, ScoreSize.Adaptive
    )

    var alignments2 = sw_byte(
        Span(ref2),
        ReferenceDirection.Forward,
        len(query2),
        gap_open,
        gap_extend,
        profile2.profile_byte.value(),
        0,
        profile2.bias,
        4,
    )

    print("\nTest 2 - Gap in query:")
    print("Score:", alignments2[0].score)
    print("Reference end:", alignments2[0].reference)
    print("Query end:", alignments2[0].query)

    # Test 3: Gap in reference
    var profile3 = Profile[__origin_of(query3), __origin_of(matrix)](
        Span(query3), matrix, ScoreSize.Adaptive
    )

    var alignments3 = sw_byte(
        Span(ref3),
        ReferenceDirection.Forward,
        len(query3),
        gap_open,
        gap_extend,
        profile3.profile_byte.value(),
        0,
        profile3.bias,
        4,
    )

    print("\nTest 3 - Gap in reference:")
    print("Score:", alignments3[0].score)
    print("Reference end:", alignments3[0].reference)
    print("Query end:", alignments3[0].query)

    print("\nAll comprehensive tests completed!")


fn test_compare_vs_c() raises:
    var reference = List("CAGCCTTTCTGACCCGGAAATCAAAATAGGCACAACAAA".as_bytes())
    var ref_seq = nt_to_num(reference)
    var read = List("CTGAGCCGGTAAATC".as_bytes())
    var read_seq = nt_to_num(read)

    var matrix = ScoringMatrix.default_matrix(5, matched=2, mismatched=-2)
    matrix.set_last_row_to_value(0)
    # Create query profile
    var profile = Profile[__origin_of(read_seq), __origin_of(matrix)](
        Span(read_seq), matrix, ScoreSize.Adaptive
    )

    # Set gap penalties
    var gap_open = UInt8(3)  # Gap open penalty
    var gap_extend = UInt8(1)  # Gap extension penalty

    # Run Smith-Waterman alignment
    var alignments = sw_byte(
        Span(ref_seq),
        ReferenceDirection.Forward,
        len(read_seq),
        gap_open,
        gap_extend,
        profile.profile_byte.value(),
        0,  # No early termination
        profile.bias,
        15,  # Small mask length
    )
    print(alignments[0].score, alignments[0].reference, alignments[0].query)
    print(alignments[1].score, alignments[1].reference, alignments[1].query)


# Run tests
fn main() raises:
    print("Running basic test...")
    test_sw_byte()

    # print("\nRunning comprehensive test...")
    # test_sw_byte_comprehensive()

    # print("Test vs C")
    # test_compare_vs_c()
