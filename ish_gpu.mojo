from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, warp, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace, external_memory
from memory import stack_allocation
from layout import Layout, LayoutTensor
from math import iota
from sys import sizeof, info
from memory import memcpy
from math import ceildiv
from time.time import perf_counter
from sys import stdout, stderr

from ishlib.matcher.alignment.ssw_align import (
    ssw_align,
    Profile,
    ScoreSize,
    Alignment,
)
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix

from ExtraMojo.io.buffered import BufferedReader, BufferedWriter
from ExtraMojo.bstr.bstr import find

alias UPPER32 = UInt32.MAX.cast[DType.uint64]() << 32
alias PATTERN = "MTEYKLVVVGAGGVGKSALTI".as_bytes()


@value
@register_passable("trivial")
struct LineCoords:
    var start: UInt32
    """0-based inclusive start."""
    var end: UInt32
    """Exclusive end."""

    fn to_u64(self) -> UInt64:
        var value: UInt64 = 0
        value += self.start.cast[DType.uint64]()
        value <<= 32
        value += self.end.cast[DType.uint64]()
        return value

    @staticmethod
    fn from_u64(value: UInt64) -> Self:
        var start = value >> 32
        var end = value ^ UPPER32
        return Self(start.cast[DType.uint32](), end.cast[DType.uint32]())


@value
struct Profiles[SIMD_U8_WIDTH: Int, SIMD_U16_WIDTH: Int]:
    var fwd: Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH]
    var rev: Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH]

    fn __init__(
        out self,
        read record: Span[UInt8],
        read matrix: ScoringMatrix,
        score_size: ScoreSize,
    ):
        self.fwd = Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH](
            record, matrix, score_size
        )
        self.rev = Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH](
            record, matrix, score_size
        )


fn printing_kernel(
    data_buffer: DeviceBuffer[DType.uint8],
    coord_buffer: DeviceBuffer[DType.uint64],
    result_buffer: DeviceBuffer[DType.uint64],
    coord_size: Int,
):
    # print("GPU thread: [", thread_idx.x, thread_idx.y, thread_idx.z, "]")
    var idx = (block_idx.x * block_dim.x) + thread_idx.x
    # print(data_buffer)
    if idx < coord_size:
        var coord = LineCoords.from_u64(coord_buffer[idx])
        var slice = Span[UInt8, __origin_of(data_buffer)](
            ptr=data_buffer.unsafe_ptr().offset(coord.start),
            length=Int(coord.end - coord.start),
        )
        var gc = 0
        for i in range(0, len(slice)):
            if slice[i] == ord("G") or slice[i] == ord("C"):
                gc += 1
        var at = 0
        for i in range(0, len(slice)):
            if slice[i] == ord("A") or slice[i] == ord("T"):
                at += 1
        var start = find(slice, PATTERN)
        if start:
            result_buffer[idx] = LineCoords(
                start.value() + at, start.value() + len(PATTERN) + gc
            ).to_u64()
        else:
            var line_len = coord.end - coord.start
            result_buffer[idx] = LineCoords(
                line_len + at, line_len + gc
            ).to_u64()

    # else:
    #     print("Waste detected at:", block_idx.x, block_dim.x, thread_idx.x)


def main():
    var reader = BufferedReader(
        open("/home/ubuntu/data/uniprot_sprot.fasta", "r")
    )

    var lines = List[List[UInt8]]()
    var coords = List[UInt64]()
    var total_bytes = 0
    while True:
        var buffer = List[UInt8]()
        var bytes_read = reader.read_until(buffer)
        if bytes_read == 0:
            break
        var start = total_bytes
        total_bytes += len(buffer)
        var end = total_bytes
        coords.append(LineCoords(start, end).to_u64())
        lines.append(buffer)

    var writer = BufferedWriter(stderr)

    # CPU bench
    var cpu_start = perf_counter()
    for line in lines:
        var gc = 0
        for i in range(0, len(line[])):
            if line[][i] == ord("G") or line[][i] == ord("C"):
                gc += 1
        var at = 0
        for i in range(0, len(line[])):
            if line[][i] == ord("A") or line[][i] == ord("T"):
                at += 1
        var start = find(line[], PATTERN)
        if start:
            var c = LineCoords(start.value(), start.value() + len(PATTERN))
            writer.write(c.start + at, " ", c.end + gc, "\n")
        else:
            var c = LineCoords(len(line[]), len(line[]))
            writer.write(c.start + at, " ", c.end + gc, "\n")
    var cpu_end = perf_counter()
    print("CPU Time:", cpu_end - cpu_start)

    var ctx = DeviceContext()

    var gpu_start = perf_counter()
    # Allocate data on the host and return a buffer which owns that data
    var host_data_buffer = ctx.enqueue_create_host_buffer[DType.uint8](
        total_bytes
    )
    var host_coord_buffer = ctx.enqueue_create_host_buffer[DType.uint64](
        len(coords)
    )
    ctx.synchronize()

    var dev_data_buffer = ctx.enqueue_create_buffer[DType.uint8](total_bytes)
    var dev_coord_buffer = ctx.enqueue_create_buffer[DType.uint64](len(coords))
    var dev_results = ctx.enqueue_create_buffer[DType.uint64](len(coords))
    ctx.enqueue_memset(dev_results, 0)

    host_data_buffer_ptr = host_data_buffer.unsafe_ptr()

    # Copy lines into host buffer
    var byte_count = 0
    var elem_count = 0
    for line in lines:
        memcpy(
            host_data_buffer_ptr.offset(byte_count),
            line[].unsafe_ptr(),
            len(line[]),
        )
        host_coord_buffer[elem_count] = coords[elem_count]
        byte_count += len(line[])
        elem_count += 1

    host_data_buffer.enqueue_copy_to(dev_data_buffer)
    host_coord_buffer.enqueue_copy_to(dev_coord_buffer)

    # # Ensure the host buffer has finished being created
    ctx.synchronize()

    print("Searching", len(coords))
    alias BLOCK_SIZE = 16
    ctx.enqueue_function[printing_kernel](
        dev_data_buffer,
        dev_coord_buffer,
        dev_results,
        len(coords),
        grid_dim=ceildiv(len(coords), BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    var host_results = ctx.enqueue_create_host_buffer[DType.uint64](len(coords))
    dev_results.enqueue_copy_to(host_results)
    ctx.synchronize()

    for i in range(0, len(coords)):
        var c = LineCoords.from_u64(host_results[i])
        writer.write(c.start, " ", c.end, "\n")

    var gpu_end = perf_counter()
    print("GPU Time:", gpu_end - gpu_start)

    writer.flush()
