from sys import stdout

from ExtraMojo.io.buffered import BufferedWriter

from ishlib.do_search import do_search
from ishlib.searcher_settings import SearcherSettings
from ishlib.vendor.log import Logger


fn main() raises:
    var searcher_settings = SearcherSettings.from_args()
    if not searcher_settings:
        return
    var settings = searcher_settings.value()
    if settings.is_output_stdout():
        do_search(settings, BufferedWriter(stdout))
    else:
        do_search(settings, BufferedWriter(open(settings.output_file, "w")))
