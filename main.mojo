from time.time import perf_counter

from ishlib.gpu import has_gpu
from ishlib.line_search_runner import LineSearchRunner
from ishlib.fasta_search_runner import FastaSearchRunner
from ishlib.matcher.basic_local_matcher import BasicLocalMatcher
from ishlib.matcher.basic_global_matcher import BasicGlobalMatcher
from ishlib.matcher.basic_semi_global_matcher import BasicSemiGlobalMatcher
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
from ishlib.searcher_settings import SearcherSettings
from ishlib.vendor.log import Logger


fn main() raises:
    # TODO: buble up the scoring matrix

    var searcher_settings = SearcherSettings.from_args()
    if not searcher_settings:
        return
    var settings = searcher_settings.value()

    if settings.match_algo == "naive_exact":
        if settings.record_type == "line":
            if settings.threads == 1:
                var runner = LineSearchRunner[NaiveExactMatcher](
                    settings,
                    NaiveExactMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
            else:
                var runner = ParallelLineSearchRunner[NaiveExactMatcher](
                    settings,
                    NaiveExactMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner[NaiveExactMatcher](
                    settings,
                    NaiveExactMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
            else:
                var runner = ParallelFastaSearchRunner[NaiveExactMatcher](
                    settings,
                    NaiveExactMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    elif settings.match_algo == "basic-local":
        if settings.record_type == "line":
            if settings.threads == 1:
                var runner = LineSearchRunner[BasicLocalMatcher](
                    settings,
                    BasicLocalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
            else:
                var runner = ParallelLineSearchRunner[BasicLocalMatcher](
                    settings,
                    BasicLocalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner[BasicLocalMatcher](
                    settings,
                    BasicLocalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
            else:
                var runner = ParallelFastaSearchRunner[BasicLocalMatcher](
                    settings,
                    BasicLocalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    elif settings.match_algo == "striped-local":
        if settings.record_type == "line":
            if settings.threads == 1:
                var runner = LineSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](
                    settings,
                    StripedLocalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
            else:
                var runner = ParallelLineSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](
                    settings,
                    StripedLocalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](
                    settings,
                    StripedLocalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
            else:
                var runner = ParallelFastaSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](
                    settings,
                    StripedLocalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    elif settings.match_algo == "basic-global":
        if settings.record_type == "line":
            if settings.threads == 1:
                var runner = LineSearchRunner(
                    settings,
                    BasicGlobalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
            else:
                var runner = ParallelLineSearchRunner(
                    settings,
                    BasicGlobalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner(
                    settings,
                    BasicGlobalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
            else:
                var runner = ParallelFastaSearchRunner(
                    settings,
                    BasicGlobalMatcher(settings.pattern, settings.matrix_kind),
                )
                runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    elif settings.match_algo == "basic-semi-global":
        if settings.record_type == "line":
            if settings.threads == 1:
                var runner = LineSearchRunner(
                    settings,
                    BasicSemiGlobalMatcher(
                        settings.pattern, settings.matrix_kind
                    ),
                )
                runner.run_search()
            else:
                var runner = ParallelLineSearchRunner(
                    settings,
                    BasicSemiGlobalMatcher(
                        settings.pattern, settings.matrix_kind
                    ),
                )
                runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner(
                    settings,
                    BasicSemiGlobalMatcher(
                        settings.pattern, settings.matrix_kind
                    ),
                )
                runner.run_search()
            else:
                var runner = ParallelFastaSearchRunner(
                    settings,
                    BasicSemiGlobalMatcher(
                        settings.pattern, settings.matrix_kind
                    ),
                )
                runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    elif settings.match_algo == "striped-semi-global":
        if settings.record_type == "line":
            if settings.threads == 1:
                var runner = LineSearchRunner[StripedSemiGlobalMatcher](
                    settings,
                    StripedSemiGlobalMatcher(
                        settings.pattern, settings.matrix_kind
                    ),
                )
                runner.run_search()
            else:
                if settings.max_gpus == 0 or not has_gpu():
                    var runner = ParallelLineSearchRunner[
                        StripedSemiGlobalMatcher
                    ](
                        settings,
                        StripedSemiGlobalMatcher(
                            settings.pattern, settings.matrix_kind
                        ),
                    )
                    runner.run_search()
                else:

                    @parameter
                    if has_gpu():
                        var qlen = len(settings.pattern)
                        if qlen <= 25:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=25,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        elif qlen <= 50:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=50,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        elif qlen <= 100:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=100,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        elif qlen <= 200:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=200,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        elif qlen <= 400:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=400,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        elif qlen <= 800:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=800,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        elif qlen <= 1600:
                            var runner = GpuParallelLineSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=1600,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        else:
                            # CPU fallback
                            var runner = ParallelLineSearchRunner[
                                StripedSemiGlobalMatcher
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                    else:
                        var runner = ParallelLineSearchRunner[
                            StripedSemiGlobalMatcher
                        ](
                            settings,
                            StripedSemiGlobalMatcher(
                                settings.pattern, settings.matrix_kind
                            ),
                        )
                        runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner[StripedSemiGlobalMatcher](
                    settings,
                    StripedSemiGlobalMatcher(
                        settings.pattern, settings.matrix_kind
                    ),
                )
                runner.run_search()
            else:
                if settings.max_gpus == 0 or not has_gpu():
                    var runner = ParallelFastaSearchRunner[
                        StripedSemiGlobalMatcher
                    ](
                        settings,
                        StripedSemiGlobalMatcher(
                            settings.pattern, settings.matrix_kind
                        ),
                    )
                    runner.run_search()
                else:

                    @parameter
                    if has_gpu():
                        var qlen = len(settings.pattern)
                        if qlen <= 25:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=25,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        elif qlen <= 50:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=50,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        elif qlen <= 100:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=100,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        elif qlen <= 200:
                            var start = perf_counter()
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=200,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            Logger.timing(
                                "Setupt time:", perf_counter() - start
                            )
                            runner.run_search()
                            var end = perf_counter()
                            Logger.timing("Time to process: ", end - start)
                        elif qlen <= 400:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=400,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        elif qlen <= 800:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=800,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        elif qlen <= 1600:
                            var runner = GpuParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher,
                                max_matrix_length=576,
                                max_query_length=1600,
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                        else:
                            # CPU fallback
                            var runner = ParallelFastaSearchRunner[
                                StripedSemiGlobalMatcher
                            ](
                                settings,
                                StripedSemiGlobalMatcher(
                                    settings.pattern, settings.matrix_kind
                                ),
                            )
                            runner.run_search()
                    else:
                        var runner = ParallelFastaSearchRunner[
                            StripedSemiGlobalMatcher
                        ](
                            settings,
                            StripedSemiGlobalMatcher(
                                settings.pattern, settings.matrix_kind
                            ),
                        )
                        runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    else:
        raise "Unsupported match algo: {}".format(settings.match_algo)
