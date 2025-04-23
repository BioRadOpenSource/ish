from time.time import perf_counter
from utils import StringSlice

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
            raise "Invalid record type: {}".format(settings.record_type)
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
                        var qlen = len(settings.pattern)
                        if qlen <= 25:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=25,
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
                        elif qlen <= 50:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=50,
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
                        elif qlen <= 100:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=100,
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
                        elif qlen <= 200:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=200,
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
                        elif qlen <= 400:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=400,
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
                        elif qlen <= 800:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=800,
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
                        elif qlen <= 1600:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=1600,
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
                        var qlen = len(settings.pattern)
                        if qlen <= 25:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=25,
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
                        elif qlen <= 50:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=50,
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
                        elif qlen <= 100:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=100,
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
                        elif qlen <= 200:
                            var start = perf_counter()
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=200,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern,
                                    settings.score_threshold,
                                    settings.sg_ends_free,
                                    settings.matrix_kind,
                                ),
                            )
                            Logger.timing(
                                "Setupt time:", perf_counter() - start
                            )
                            runner.run_search(writer)
                            var end = perf_counter()
                            Logger.timing("Time to process: ", end - start)
                        elif qlen <= 400:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=400,
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
                        elif qlen <= 800:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=800,
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
                        elif qlen <= 1600:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_query_length=1600,
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
            raise "Invalid record type: {}".format(settings.record_type)
    else:
        raise "Unsupported match algo: {}".format(settings.match_algo)
