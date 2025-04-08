# TODO: This is a copy paste from the fasta reader, make it generic

from algorithm.functional import parallelize
from math import ceildiv
from time.time import perf_counter

from ishlib.cpu.process_batch import (
    parallel_starts_ends as cpu_parallel_starts_ends,
    parallel_starts as cpu_parallel_starts,
)
from ishlib.gpu.searcher_device import (
    SearcherDevice,
)
from ishlib.matcher import (
    GpuMatcher,
    MatchResult,
    Matcher,
    SearchableWithIndex,
    WhereComputed,
    ComputedMatchResult,
)
from ishlib.searcher_settings import SearcherSettings


fn parallel_starts_ends[
    M: GpuMatcher,
    S: SearchableWithIndex,
    max_matrix_length: UInt = 576,
    max_query_length: UInt = 200,
    max_target_length: UInt = 1024,
](
    mut ctxs: List[
        SearcherDevice[
            M.batch_match_coarse[
                max_matrix_length, max_query_length, max_target_length
            ]
        ],
    ],
    read matcher: M,
    read settings: SearcherSettings,
    read seqs: Span[S],
    read cpu_seqs: Span[S],
) raises -> List[Optional[ComputedMatchResult]]:
    var start = perf_counter()
    var output_len = len(seqs) + len(cpu_seqs)
    var outputs = List[Optional[ComputedMatchResult]](capacity=output_len)
    for _ in range(0, output_len):
        outputs.append(None)

    var targets_per_device = ceildiv(len(seqs), len(ctxs))
    # Create the buffers
    for ctx in ctxs:
        ctx[].set_block_info(
            len(seqs),
            len(settings.pattern),
            matcher.matrix_len(),
            max_target_length=max_target_length,
        )
        ctx[].host_create_input_buffers()
    # TODO: Could separate out host buffer creating vs device, etc

    for ctx in ctxs:
        ctx[].synchronize()
    var buffers_created = perf_counter()
    print("Buffer creation time:", buffers_created - start)

    # fill in input data
    var device_id = 0
    for i in range(0, len(seqs), targets_per_device):
        ctxs[device_id].device_create_input_buffers()
        ctxs[device_id].set_host_inputs(
            Span(settings.pattern),
            matcher.matrix_bytes(),
            matcher.matrix_len(),
            seqs[i : min(i + targets_per_device, len(seqs))],
        )
        ctxs[device_id].copy_inputs_to_device()
        device_id += 1

    # This is the slowest part by far, moving it around doesn't seem to help much. I wish we had threads.
    var cpu_start = perf_counter()
    cpu_parallel_starts_ends(matcher, settings, cpu_seqs, outputs)
    var cpu_end = perf_counter()
    print("Long seqs cpu time: ", cpu_end - cpu_start)

    for ctx in ctxs:
        ctx[].synchronize()
    var buffers_filled = perf_counter()
    print("Buffer fill time:", buffers_filled - buffers_created)

    # Launch Kernel
    for ctx in ctxs:
        ctx[].device_create_output_buffers()
        ctx[].launch_kernel()
        ctx[].host_create_output_buffers()

    # Process the long seqs
    # TODO: work on ordering these correctly

    # Get outputs
    for ctx in ctxs:
        ctx[].synchronize()
        ctx[].copy_outputs_to_host()
    var gpu_done = perf_counter()
    print("GPU processing time (with cpu):", gpu_done - buffers_filled)

    for ctx in ctxs:
        ctx[].synchronize()
    var copy_down = perf_counter()
    print("Copy down time:", copy_down - gpu_done)

    # Now check for any matches?
    var total_items = 0
    for ctx in ctxs:
        var end = min(
            total_items + ctx[].block_info.value().num_targets, len(seqs)
        )
        var starts_start = perf_counter()
        cpu_parallel_starts[where_computed = WhereComputed.Gpu](
            matcher,
            settings,
            Span[Int32, __origin_of(ctx[].host_scores.value())](
                ptr=ctx[].host_scores.value().unsafe_ptr(),
                length=ctx[].block_info.value().num_targets,
            ).get_immutable(),
            Span[Int32, __origin_of(ctx[].host_scores.value())](
                ptr=ctx[].host_target_ends.value().unsafe_ptr(),
                length=ctx[].block_info.value().num_targets,
            ).get_immutable(),
            seqs[total_items:end],
            outputs,
            total_items,
        )
        var starts_done = perf_counter()
        print("Starts time:", starts_done - starts_start)
        total_items = end
    var end = perf_counter()
    print("Total time:", end - start)
    return outputs
