from ishlib.searcher_settings import SearcherSettings
from ishlib.line_search_runner import LineSearchRunner
from ishlib.parallel_line_search_runner import ParallelLineSearchRunner
from ishlib.fasta_search_runner import FastaSearchRunner
from ishlib.parallel_fasta_search_runner import ParallelFastaSearchRunner
from ishlib.matcher.basic_global_matcher import BasicGlobalMatcher
from ishlib.matcher.basic_semi_global_matcher import BasicSemiGlobalMatcher
from ishlib.matcher.naive_exact_matcher import NaiveExactMatcher
from ishlib.matcher.basic_local_matcher import BasicLocalMatcher
from ishlib.matcher.striped_local_matcher import StripedLocalMatcher
from ishlib.matcher.striped_semi_global_matcher import StripedSemiGlobalMatcher


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
                    settings, NaiveExactMatcher(settings.pattern)
                )
                runner.run_search()
            else:
                var runner = ParallelLineSearchRunner[NaiveExactMatcher](
                    settings, NaiveExactMatcher(settings.pattern)
                )
                runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner[NaiveExactMatcher](
                    settings, NaiveExactMatcher(settings.pattern)
                )
                runner.run_search()
            else:
                var runner = ParallelFastaSearchRunner[NaiveExactMatcher](
                    settings, NaiveExactMatcher(settings.pattern)
                )
                runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    elif settings.match_algo == "basic-local":
        if settings.record_type == "line":
            if settings.threads == 1:
                var runner = LineSearchRunner[BasicLocalMatcher](
                    settings, BasicLocalMatcher(settings.pattern)
                )
                runner.run_search()
            else:
                var runner = ParallelLineSearchRunner[BasicLocalMatcher](
                    settings, BasicLocalMatcher(settings.pattern)
                )
                runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner[BasicLocalMatcher](
                    settings, BasicLocalMatcher(settings.pattern)
                )
                runner.run_search()
            else:
                var runner = ParallelFastaSearchRunner[BasicLocalMatcher](
                    settings, BasicLocalMatcher(settings.pattern)
                )
                runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    elif settings.match_algo == "striped-local":
        if settings.record_type == "line":
            if settings.threads == 1:
                var runner = LineSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](settings, StripedLocalMatcher(settings.pattern))
                runner.run_search()
            else:
                var runner = ParallelLineSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](settings, StripedLocalMatcher(settings.pattern))
                runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](settings, StripedLocalMatcher(settings.pattern))
                runner.run_search()
            else:
                var runner = ParallelFastaSearchRunner[
                    StripedLocalMatcher[__origin_of(settings.pattern)]
                ](settings, StripedLocalMatcher(settings.pattern))
                runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    elif settings.match_algo == "basic-global":
        if settings.record_type == "line":
            if settings.threads == 1:
                var runner = LineSearchRunner(
                    settings, BasicGlobalMatcher(settings.pattern)
                )
                runner.run_search()
            else:
                var runner = ParallelLineSearchRunner(
                    settings, BasicGlobalMatcher(settings.pattern)
                )
                runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner(
                    settings, BasicGlobalMatcher(settings.pattern)
                )
                runner.run_search()
            else:
                var runner = ParallelFastaSearchRunner(
                    settings, BasicGlobalMatcher(settings.pattern)
                )
                runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    elif settings.match_algo == "basic-semi-global":
        if settings.record_type == "line":
            if settings.threads == 1:
                var runner = LineSearchRunner(
                    settings, BasicSemiGlobalMatcher(settings.pattern)
                )
                runner.run_search()
            else:
                var runner = ParallelLineSearchRunner(
                    settings, BasicSemiGlobalMatcher(settings.pattern)
                )
                runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner(
                    settings, BasicSemiGlobalMatcher(settings.pattern)
                )
                runner.run_search()
            else:
                var runner = ParallelFastaSearchRunner(
                    settings, BasicSemiGlobalMatcher(settings.pattern)
                )
                runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    elif settings.match_algo == "striped-semi-global":
        if settings.record_type == "line":
            if settings.threads == 1:
                var runner = LineSearchRunner[
                    StripedSemiGlobalMatcher[__origin_of(settings.pattern)]
                ](settings, StripedSemiGlobalMatcher(settings.pattern))
                runner.run_search()
            else:
                var runner = ParallelLineSearchRunner[
                    StripedSemiGlobalMatcher[__origin_of(settings.pattern)]
                ](settings, StripedSemiGlobalMatcher(settings.pattern))
                runner.run_search()
        elif settings.record_type == "fasta":
            if settings.threads == 1:
                var runner = FastaSearchRunner[
                    StripedSemiGlobalMatcher[__origin_of(settings.pattern)]
                ](settings, StripedSemiGlobalMatcher(settings.pattern))
                runner.run_search()
            else:
                var runner = ParallelFastaSearchRunner[
                    StripedSemiGlobalMatcher[__origin_of(settings.pattern)]
                ](settings, StripedSemiGlobalMatcher(settings.pattern))
                runner.run_search()
        else:
            raise "Invalid record type: {}".format(settings.record_type)
    else:
        raise "Unsupported match algo: {}".format(settings.match_algo)
