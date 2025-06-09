from testing import assert_equal

from ishlib.matcher.alignment.striped_utils import (
    saturating_add,
    saturating_sub,
)


def test_saturating_add():
    alias dtypes = List[DType](
        DType.uint8,
        DType.int8,
        DType.uint16,
        DType.int16,
        DType.uint32,
        DType.int32,
    )
    alias widths = List[Int](4, 8, 16, 32, 64, 128)

    @parameter
    for i in range(0, len(dtypes)):

        @parameter
        for j in range(0, len(widths)):
            alias dtype = dtypes[i]
            alias width = widths[j]
            alias MIN = Scalar[dtype].MIN
            alias MAX = Scalar[dtype].MAX

            var lhs = SIMD[dtype, width](MAX)
            var rhs = SIMD[dtype, width](1)
            var expected = SIMD[dtype, width](MAX)
            assert_equal(saturating_add(lhs, rhs), expected)

            expected = SIMD[dtype, width](2)
            assert_equal(saturating_add(rhs, rhs), expected)


def test_saturating_sub():
    alias dtypes = List[DType](
        DType.uint8,
        DType.int8,
        DType.uint16,
        DType.int16,
        DType.uint32,
        DType.int32,
    )
    alias widths = List[Int](4, 8, 16, 32, 64, 128)

    @parameter
    for i in range(0, len(dtypes)):

        @parameter
        for j in range(0, len(widths)):
            alias dtype = dtypes[i]
            alias width = widths[j]
            alias MIN = Scalar[dtype].MIN
            alias MAX = Scalar[dtype].MAX

            var lhs = SIMD[dtype, width](MIN)
            var rhs = SIMD[dtype, width](1)
            var expected = SIMD[dtype, width](MIN)
            assert_equal(saturating_sub(lhs, rhs), expected)

            expected = SIMD[dtype, width](0)
            assert_equal(saturating_sub(rhs, rhs), expected)
