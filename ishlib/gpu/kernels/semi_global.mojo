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
from ishlib.matcher.alignment.scoring_matrix import BasicScoringMatrix
from ishlib.matcher.alignment.semi_global_aln.basic import (
    semi_global_parasail_gpu,
)


fn gpu_align_coarse[
    max_matrix_length: UInt = 576,
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
):
    alias matrix_skip_lookup = max_matrix_length == 0
    # Load scoring matrix into shared memory
    var basic_profile_bytes = stack_allocation[
        max_matrix_length,
        SIMD[DType.int8, 1],
        address_space = AddressSpace.SHARED,
    ]()

    for i in range(0, min(basic_matrix_len, max_matrix_length)):
        basic_profile_bytes[i] = basic_matrix_values[i]

    var basic_matrix = BasicScoringMatrix[
        address_space = AddressSpace(3), no_lookup=matrix_skip_lookup
    ](basic_profile_bytes, min(max_matrix_length, basic_matrix_len))
    # print("Create matrix of len", basic_matrix_len, matrix_skip_lookup)

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

    # Hard coded parameters
    var gap_open_penalty = -3
    var gap_ext_penalty = -1

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
            gap_open_penalty=gap_open_penalty,
            gap_extension_penalty=gap_ext_penalty,
        )

        barrier()

        # Store results
        # TODO: move this to after the loop?
        score_result_buffer[idx] = result.score
        query_end_result_buffer[idx] = Int32(result.query)
        ref_end_result_buffer[idx] = Int32(result.target)
        barrier()
