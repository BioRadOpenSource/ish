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
from ishlib.vendor.log import Logger


fn parallel_starts_ends[
    M: GpuMatcher,
    S: SearchableWithIndex,
    max_query_length: UInt = 200,
    max_target_length: UInt = 1024,
](
    mut ctxs: List[
        SearcherDevice[
            M.batch_match_coarse[max_query_length, max_target_length]
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
    Logger.timing(
        "Time to create ouput buffer with appends:", perf_counter() - start
    )

    var targets_per_device = ceildiv(len(seqs), len(ctxs))
    Logger.debug(
        "Targets per device:",
        targets_per_device,
        "for ",
        len(ctxs),
        "devices",
        "over",
        len(seqs),
        "seqs",
    )

    # Create the buffers
    var gpu_buffer_create_start = perf_counter()
    for ctx in ctxs:
        ctx[].set_block_info(
            len(seqs),
            len(matcher.encoded_pattern()),
            matcher.matrix_len(),
            matcher.matrix_kind(),
            max_target_length=max_target_length,
        )
        ctx[].host_create_input_buffers()
    # TODO: Could separate out host buffer creating vs device, etc

    for ctx in ctxs:
        ctx[].synchronize()
    var buffers_created = perf_counter()
    Logger.timing(
        "GPU only creatin time:", buffers_created - gpu_buffer_create_start
    )
    Logger.timing("Buffer creation time:", buffers_created - start)

    # fill in input data
    var buffer_fill_start = perf_counter()

    var total_items = 0
    for ctx in ctxs:
        var end = min(
            total_items + ctx[].block_info.value().num_targets, len(seqs)
        )
        ctx[].device_create_input_buffers()
        ctx[].set_host_inputs(
            matcher.encoded_pattern(),
            matcher.matrix_bytes(),
            matcher.matrix_len(),
            seqs[total_items:end],
        )
        ctx[].copy_inputs_to_device()
        total_items += end

    var buffer_fill_end = perf_counter()
    Logger.timing(
        "Time to fill host buffer:", buffer_fill_end - buffer_fill_start
    )

    # This is the slowest part by far, moving it around doesn't seem to help much. I wish we had threads.
    var cpu_start = perf_counter()
    cpu_parallel_starts_ends(matcher, settings, cpu_seqs, outputs)
    var cpu_end = perf_counter()
    Logger.timing("Long seqs cpu time: ", cpu_end - cpu_start)

    for ctx in ctxs:
        ctx[].synchronize()
    var buffers_filled = perf_counter()
    Logger.timing("Buffer fill time:", buffers_filled - buffers_created)

    # Launch Kernel
    for ctx in ctxs:
        ctx[].device_create_output_buffers()
        Logger.debug("Created device output buffers")
        ctx[].launch_kernel()
        Logger.debug("Launched kernel")
        ctx[].host_create_output_buffers()
        Logger.debug("Created host output buffers")

    # Process the long seqs
    # TODO: work on ordering these correctly

    # Get outputs
    for ctx in ctxs:
        ctx[].synchronize()
        ctx[].copy_outputs_to_host()
    var gpu_done = perf_counter()
    Logger.timing("GPU processing time (with cpu):", gpu_done - buffers_filled)

    for ctx in ctxs:
        ctx[].synchronize()
    var copy_down = perf_counter()
    Logger.timing("Copy down time:", copy_down - gpu_done)

    # Now check for any matches?
    total_items = 0
    for ctx in ctxs:
        var end = min(
            total_items + ctx[].block_info.value().num_targets, len(seqs)
        )
        var starts_start = perf_counter()
        cpu_parallel_starts[where_computed = WhereComputed.Gpu](
            matcher,
            settings,
            ctx[].host_scores.value().as_span().get_immutable(),
            ctx[].host_target_ends.value().as_span().get_immutable(),
            seqs[total_items:end],
            outputs,
            total_items,
        )
        var starts_done = perf_counter()
        Logger.timing("Starts time:", starts_done - starts_start)
        total_items = end
    var end = perf_counter()
    Logger.timing("Total time:", end - start)
    return outputs
