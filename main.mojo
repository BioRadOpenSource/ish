from ishlib.searcher_settings import SearcherSettings
from ishlib.search_runner import SearchRunner
from ishlib.matcher.naive_exact_matcher import NaiveExactMatcher
from ishlib.matcher.sw_local_matcher import SWLocalMatcher


fn main() raises:
    var searcher_settings = SearcherSettings.from_args()
    if not searcher_settings:
        return
    var settings = searcher_settings.value()

    if settings.match_algo == "naive_exact":
        var runner = SearchRunner[NaiveExactMatcher](
            settings, NaiveExactMatcher()
        )
        runner.run_search()
    elif settings.match_algo == "sw_local":
        var runner = SearchRunner[SWLocalMatcher](settings, SWLocalMatcher())
        runner.run_search()
    else:
        raise "Unsupported match algo: {}".format(settings.match_algo)
