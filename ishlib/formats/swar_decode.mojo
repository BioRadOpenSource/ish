# This is for testing only

from memory import UnsafePointer
from sys     import exit

alias U8x8  = SIMD[DType.uint8 , 8]
alias U16x8 = SIMD[DType.uint16, 8]
alias U32x8 = SIMD[DType.uint32, 8]

@always_inline
fn to_simd(p: UnsafePointer[UInt8]) ->
           UnsafePointer[SIMD[DType.uint8, 1]]:
    return p.bitcast[SIMD[DType.uint8, 1]]()

# 8 ASCII digits
@always_inline
fn decode_8(ptr: UnsafePointer[UInt8]) -> Int:
    var v  = to_simd(ptr).load[width=8](0) - UInt8(ord("0"))
    var w  = v.cast[DType.uint32]()
    var mul = U32x8(10_000_000, 1_000_000, 100_000, 10_000,
                    1_000,       100,       10,       1)
    return Int((w * mul).reduce_add())

# 6 digits (still load 8 lanes)
@always_inline
fn decode_6(ptr: UnsafePointer[UInt8]) -> Int:
    var v  = to_simd(ptr).load[width=8](0) - UInt8(ord("0"))
    var w  = v.cast[DType.uint32]()
    var mul = U32x8(100_000, 10_000, 1_000, 100, 10, 1, 0, 0)
    return Int((w * mul).reduce_add())

# 7 digits: scalar last digit
@always_inline
fn decode_7(ptr: UnsafePointer[UInt8]) -> Int:
    # Read the 7th digit at ptr[6] (offset 6)
    var last_digit = Int(to_simd(ptr.offset(6)).load[width=1](0) - UInt8(ord("0")))
    # decode_6 handles ptr[0] through ptr[5]
    return decode_6(ptr) * 10 + last_digit


# scalar last digit
@always_inline
fn decode_9(ptr: UnsafePointer[UInt8]) -> Int:
    # Read the 9th digit at ptr[8] (offset 8)
    var last_digit = Int(to_simd(ptr.offset(8)).load[width=1](0) - UInt8(ord("0")))
    # decode_8 handles ptr[0] through ptr[7]
    return decode_8(ptr) * 10 + last_digit

# 3-digit fallback
@always_inline
fn decode_3(ptr: UnsafePointer[UInt8]) -> Int:
    var sp = to_simd(ptr)

    var d0 = Int(sp.load[width = 1](0) - UInt8(ord("0")))
    var d1 = Int(sp.load[width = 1](1) - UInt8(ord("0")))
    var d2 = Int(sp.load[width = 1](2) - UInt8(ord("0")))

    return d0 * 100 + d1 * 10 + d2


# zero-pad an Int to a fixed width
fn zpad(value: Int, width: Int) -> String:
    var s = String(value)
    var pad = width - len(s)
    var out = String(capacity = width)
    for _ in range(pad):
        out.write("0")
    out.write(s)
    return out

fn expect(label: String, got: Int, want: Int):
    if got != want:
        print("FAIL {", label, "}  got ", got, "  expected ", want)
        exit(1)
 
fn run_tests():
    # 3-digit
    expect("3-zero",  decode_3("000".unsafe_ptr()), 0)
    expect("3-123",   decode_3("123".unsafe_ptr()), 123)
    expect("3-999",   decode_3("999".unsafe_ptr()), 999)

    # 6-digit
    expect("6-zero",  decode_6("000000".unsafe_ptr()), 0)
    expect("6-00123", decode_6("000123".unsafe_ptr()), 123)
    expect("6-654321",decode_6("654321".unsafe_ptr()), 654321)

    # 7-digit
    expect("7-zero",  decode_7("0000000".unsafe_ptr()), 0)
    expect("7-1234567",decode_7("1234567".unsafe_ptr()), 1234567)

    # 8-digit
    expect("8-zero",  decode_8("00000000".unsafe_ptr()), 0)
    expect("8-87654321",decode_8("87654321".unsafe_ptr()), 87654321)

    # 9-digit
    expect("9-zero",  decode_9("000000000".unsafe_ptr()), 0)
    expect("9-123456789",decode_9("123456789".unsafe_ptr()), 123456789)

    print("All SWAR decode tests passed.")

fn main():
    run_tests()
