from collections import Optional
from collections.string import StringSlice
from memory import memcpy, memset_zero


@value
@register_passable("trivial")
struct TargetSpan:
    var start: Int
    var end: Int


@value
struct AlignmentResult:
    var score: Int32
    var alignment1: Optional[String]
    var alignment2: Optional[String]
    var coords: Optional[TargetSpan]

    fn display_alignments(
        read self, cols: Int = 120
    ) raises -> Optional[String]:
        if not self.alignment1:
            return None
        var ret = String()
        var aln1 = StringSlice(self.alignment1.value())
        var aln2 = StringSlice(self.alignment2.value())
        var i = 0
        ret.write("Score: ", self.score, "\n")
        while i < len(aln1):
            ret.write(aln1[i : min(i + cols, len(aln1))], "\n")
            ret.write("|" * (min(i + cols, len(aln1)) - i), "\n")
            ret.write(aln2[i : min(i + cols, len(aln1))], "\n")
            ret.write("-" * (min(i + cols, len(aln1)) - i), "\n")
            i += cols
        return ret


@always_inline
fn create_reversed(input: Span[UInt8]) -> List[UInt8]:
    var ret = List(input)
    ret.reverse()
    return ret


struct AlignedMemory[dtype: DType, width: Int, alignment: Int](
    Copyable, Movable, Sized
):
    var ptr: UnsafePointer[SIMD[dtype, width], alignment=alignment]
    var length: Int

    fn __init__[zero_mem: Bool = True](out self, length: Int):
        self.ptr = UnsafePointer[
            SIMD[Self.dtype, self.width], alignment = self.alignment
        ].alloc(length)
        self.length = length

        @parameter
        if zero_mem:
            memset_zero(self.ptr, self.length)

    @always_inline
    fn __getitem__[
        I: Indexer
    ](ref self, offset: I) -> ref [self] SIMD[dtype, width]:
        return self.ptr[offset]

    fn __copyinit__(out self, read other: Self):
        self.ptr = UnsafePointer[
            SIMD[Self.dtype, self.width], alignment = self.alignment
        ].alloc(other.length)
        self.length = other.length
        memcpy(self.ptr, other.ptr, self.length)

    fn __moveinit__(out self, owned other: Self):
        self.ptr = other.ptr
        self.length = other.length

    fn __del__(owned self):
        self.ptr.free()

    @always_inline
    fn __len__(read self) -> Int:
        return self.length

    @always_inline
    fn as_span(
        ref self,
    ) -> Span[SIMD[self.dtype, self.width], __origin_of(self)]:
        return rebind[Span[SIMD[self.dtype, self.width], __origin_of(self)]](
            Span(self.ptr, self.length)
        )
