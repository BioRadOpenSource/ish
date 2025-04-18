from gpu import thread_idx, block_idx, block_dim, warp, barrier
from gpu.host import (
    DeviceContext,
    DeviceBuffer,
)  # HostBuffer (not in 25.2)
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import ceildiv
from memory import UnsafePointer, stack_allocation, memcpy

from ishlib.gpu.dynamic_2d_matrix import Dynamic2DMatrix, StorageFormat
from ishlib.matcher.alignment.scoring_matrix import (
    BasicScoringMatrix,
    MatrixKind,
)
from ishlib.matcher.alignment.semi_global_aln.basic import (
    semi_global_parasail_gpu,
)


fn gpu_align_coarse[
    matrix_kind: MatrixKind,
    max_query_length: UInt = 200,
    max_target_length: UInt = 1024,
](
    query: DeviceBuffer[DType.uint8],
    ref_buffer: DeviceBuffer[DType.uint8],
    target_ends: DeviceBuffer[DType.uint32],
    score_result_buffer: DeviceBuffer[DType.int32],
    query_end_result_buffer: DeviceBuffer[DType.int32],
    ref_end_result_buffer: DeviceBuffer[DType.int32],
    basic_matrix_values: DeviceBuffer[DType.int8],
    basic_matrix_len: UInt,
    query_len: UInt,
    target_ends_len: UInt,
    thread_count: UInt,
    gap_open: UInt,
    gap_extend: UInt,
):
    alias matrix_skip_lookup = matrix_kind.skip_lookup()
    # Load scoring matrix into shared memory
    var basic_profile_bytes = stack_allocation[
        len(matrix_kind),
        SIMD[DType.int8, 1],
        address_space = AddressSpace.SHARED,
    ]()

    for i in range(0, min(Int(basic_matrix_len), len(matrix_kind))):
        basic_profile_bytes[i] = basic_matrix_values[i]

    var basic_matrix = BasicScoringMatrix[
        address_space = AddressSpace(3), no_lookup=matrix_skip_lookup
    ](basic_profile_bytes, min(Int(basic_matrix_len), len(matrix_kind)))

    # Load query sequence into shared memory - all threads use the same query
    var query_seq_ptr = stack_allocation[
        max_query_length,
        SIMD[DType.uint8, 1],
        address_space = AddressSpace.SHARED,
    ]()

    for i in range(0, min(query_len, max_query_length)):
        query_seq_ptr[i] = query[i]

    barrier()  # Ensure shared memory is fully loaded before proceeding

    # Calculate global thread index
    var thread_id = (block_idx.x * block_dim.x) + thread_idx.x

    # Skip if this thread is outside our desired range
    if thread_id >= thread_count:
        return

    # Process references in a strided pattern
    # Each thread processes references with indices: thread_id, thread_id + thread_count, thread_id + 2*thread_count, etc.
    for idx in range(thread_id, target_ends_len, thread_count):
        # Get the length of this reference sequence
        var target_len = Int(target_ends[idx])

        # Perform the alignment
        var result = semi_global_parasail_gpu[
            DType.int16,
            max_query_length=max_query_length,
            free_query_start_gaps=True,
            free_query_end_gaps=True,
            free_target_start_gaps=True,
            free_target_end_gaps=True,
            matrix_skip_lookup=matrix_skip_lookup,
        ](
            query_seq_ptr,
            query_len,
            ref_buffer.unsafe_ptr(),
            max_target_length,
            target_ends_len,
            idx,
            target_len,
            basic_matrix,
            gap_open_penalty=-Int(gap_open),
            gap_extension_penalty=-Int(gap_extend),
        )

        barrier()

        # Store results
        # TODO: move this to after the loop?
        score_result_buffer[idx] = result.score
        query_end_result_buffer[idx] = Int32(result.query) + 1
        ref_end_result_buffer[idx] = Int32(result.target) + 1
        barrier()
