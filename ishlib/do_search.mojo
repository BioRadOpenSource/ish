from time.time import perf_counter

from ExtraMojo.io import MovableWriter
from ExtraMojo.io.buffered import BufferedWriter

from ishlib.gpu import has_gpu
from ishlib.line_search_runner import LineSearchRunner
from ishlib.fasta_search_runner import FastaSearchRunner
from ishlib.matcher.naive_exact_matcher import NaiveExactMatcher
from ishlib.matcher.striped_local_matcher import StripedLocalMatcher
from ishlib.matcher.striped_semi_global_matcher import StripedSemiGlobalMatcher
from ishlib.parallel_fasta_search_runner import (
    ParallelFastaSearchRunner,
    GpuParallelFastaSearchRunner,
)
from ishlib.parallel_line_search_runner import (
    ParallelLineSearchRunner,
    GpuParallelLineSearchRunner,
)
from ishlib.matcher.alignment.score_matrix import MatrixKind
from ishlib.searcher_settings import SearcherSettings, SemiGlobalEndsFreeness
from ishlib.vendor.log import Logger


fn do_search[
    W: MovableWriter
](settings: SearcherSettings, owned writer: BufferedWriter[W]) raises:
    Logger.info(
        # fmt: off
        "match_algo:", settings.match_algo, "|",
        "threads:", settings.threads, "|",
        "max_gpus:", settings.max_gpus, "|",
        "record_type:", settings.record_type, "|",
        "matrix_kind:", String(settings.matrix_kind), "|",
        "pattern:", String(StringSlice(unsafe_from_utf8=settings.pattern))
        # fmt: on
    )

    if settings.match_algo == "striped-local":
        if settings.record_type == "line":
            if settings.threads <= 1:
                var runner = LineSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](
                    settings,
                    StripedLocalMatcher(
                        settings.pattern,
                        settings.score_threshold,
                        settings.matrix_kind,
                    ),
                )
                runner.run_search(writer)
            else:
                var runner = ParallelLineSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](
                    settings,
                    StripedLocalMatcher(
                        settings.pattern,
                        settings.score_threshold,
                        settings.matrix_kind,
                    ),
                )
                runner.run_search(writer)
        elif settings.record_type == "fasta":
            if settings.threads <= 1:
                var runner = FastaSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](
                    settings,
                    StripedLocalMatcher(
                        settings.pattern,
                        settings.score_threshold,
                        settings.matrix_kind,
                    ),
                )
                runner.run_search(writer)
            else:
                var runner = ParallelFastaSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](
                    settings,
                    StripedLocalMatcher(
                        settings.pattern,
                        settings.score_threshold,
                        settings.matrix_kind,
                    ),
                )
                runner.run_search(writer)
        else:
            raise String("Invalid record type: {}").format(settings.record_type)
    elif settings.match_algo == "striped-semi-global":
        if settings.record_type == "line":
            if settings.threads <= 1:
                var runner = LineSearchRunner[StripedSemiGlobalMatcher](
                    settings,
                    StripedSemiGlobalMatcher(
                        settings.pattern,
                        settings.score_threshold,
                        settings.sg_ends_free,
                        settings.matrix_kind,
                    ),
                )
                runner.run_search(writer)
            else:
                if settings.max_gpus == 0 or not has_gpu():
                    var runner = ParallelLineSearchRunner[
                        StripedSemiGlobalMatcher
                    ](
                        settings,
                        StripedSemiGlobalMatcher(
                            settings.pattern,
                            settings.score_threshold,
                            settings.sg_ends_free,
                            settings.matrix_kind,
                        ),
                    )
                    runner.run_search(writer)
                else:

                    @parameter
                    if has_gpu():

                        @parameter
                        @always_inline
                        fn choose_striped_semi_global_line_search_runner(
                            qlen: Int,
                        ) raises:
                            alias MAX_QLEN = List(
                                25, 50, 100, 200, 400, 800, 1600
                            )

                            @parameter
                            for i in range(0, len(MAX_QLEN)):
                                alias max_len = MAX_QLEN[i]
                                if qlen <= max_len:
                                    var runner = GpuParallelLineSearchRunner[
                                        StripedSemiGlobalMatcher,
                                        max_query_length=max_len,
                                    ](
                                        settings,
                                        StripedSemiGlobalMatcher(
                                            settings.pattern,
                                            settings.score_threshold,
                                            settings.sg_ends_free,
                                            settings.matrix_kind,
                                        ),
                                    )
                                    runner.run_search(writer)
                                    return

                            # CPU fallback
                            var runner = ParallelLineSearchRunner[
                                StripedSemiGlobalMatcher
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern,
                                    settings.score_threshold,
                                    settings.sg_ends_free,
                                    settings.matrix_kind,
                                ),
                            )
                            runner.run_search(writer)

                        choose_striped_semi_global_line_search_runner(
                            len(settings.pattern)
                        )
                    else:
                        var runner = ParallelLineSearchRunner[
                            StripedSemiGlobalMatcher
                        ](
                            settings,
                            StripedSemiGlobalMatcher(
                                settings.pattern,
                                settings.score_threshold,
                                settings.sg_ends_free,
                                settings.matrix_kind,
                            ),
                        )
                        runner.run_search(writer)
        elif settings.record_type == "fasta":
            if settings.threads <= 1:
                var runner = FastaSearchRunner[StripedSemiGlobalMatcher](
                    settings,
                    StripedSemiGlobalMatcher(
                        settings.pattern,
                        settings.score_threshold,
                        settings.sg_ends_free,
                        settings.matrix_kind,
                    ),
                )
                runner.run_search(writer)
            else:
                if settings.max_gpus == 0 or not has_gpu():
                    var runner = ParallelFastaSearchRunner[
                        StripedSemiGlobalMatcher
                    ](
                        settings,
                        StripedSemiGlobalMatcher(
                            settings.pattern,
                            settings.score_threshold,
                            settings.sg_ends_free,
                            settings.matrix_kind,
                        ),
                    )
                    runner.run_search(writer)
                else:

                    @parameter
                    if has_gpu():

                        @parameter
                        @always_inline
                        fn choose_striped_semi_global_fasta_search_runner(
                            qlen: Int,
                        ) raises:
                            alias MAX_QLEN = List(
                                25, 50, 100, 200, 400, 800, 1600
                            )

                            @parameter
                            for i in range(0, len(MAX_QLEN)):
                                alias max_len = MAX_QLEN[i]
                                if qlen <= max_len:
                                    var runner = GpuParallelFastaSearchRunner[
                                        StripedSemiGlobalMatcher,
                                        max_query_length=max_len,
                                    ](
                                        settings,
                                        StripedSemiGlobalMatcher(
                                            settings.pattern,
                                            settings.score_threshold,
                                            settings.sg_ends_free,
                                            settings.matrix_kind,
                                        ),
                                    )
                                    runner.run_search(writer)
                                    return

                            # CPU fallback
                            var runner = ParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern,
                                    settings.score_threshold,
                                    settings.sg_ends_free,
                                    settings.matrix_kind,
                                ),
                            )
                            runner.run_search(writer)

                        choose_striped_semi_global_fasta_search_runner(
                            len(settings.pattern)
                        )
                    else:
                        var runner = ParallelFastaSearchRunner[
                            StripedSemiGlobalMatcher
                        ](
                            settings,
                            StripedSemiGlobalMatcher(
                                settings.pattern,
                                settings.score_threshold,
                                settings.sg_ends_free,
                                settings.matrix_kind,
                            ),
                        )
                        runner.run_search(writer)
        else:
            raise String("Invalid record type: {}").format(settings.record_type)
    else:
        raise String("Unsupported match algo: {}").format(settings.match_algo)
