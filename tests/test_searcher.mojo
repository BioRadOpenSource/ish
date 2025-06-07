from pathlib import Path
from testing import assert_equal
from tempfile import TemporaryDirectory

from ExtraMojo.io.buffered import BufferedWriter

from ishlib.do_search import do_search
from ishlib.gpu import has_gpu
from ishlib.matcher.alignment.scoring_matrix import MatrixKind
from ishlib.parallel_line_search_runner import ParallelLineSearchRunner
from ishlib.searcher_settings import (
    SearcherSettings,
    SemiGlobalEndsFreeness as SGEF,
)
from ishlib.vendor.tty_info import Info as TTYInfoResult

alias ASCII_TEXT = """
This is a test file full of hay.
The needle is hidden in here.
I'm not sure what the needle is.
Here's some UTF-8 just in case that makes things really explodeeeeeee 🔥.
And this is just a very long line that goes on for longer than any of the other lines to make sure needle things are working okay on that front as well when the lines get really long.
"""

alias NUC_FASTA_TEST = """
>Human_DNA With a Comment
ACTGACTGACGACGACGACTAATAGNNNNACTGANNNATCATCTAG
GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
ACTGACTGACGACGACGACTAATAGNNNNACTGANgNATCATcTAG
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
ACTGACTGACGACGACGACTAATAGNNNNACTGANNNATCATCTAG
ACTGACTGACGACGACGACTAATAG
>Also_Human_DNA With a Comment
AGTGAGTGAGGAGGAGGAGTAATAGNNNNAGTGANNNATGATGTAG
AAAAAAAGGGGGGGGGGCCCCCCCTTTTTTTTTTTGGGGGNNNNNN
ACTGACTGACGACGGGGGGGGGTAGNNNNACTGANgNATCATcTAG
AAAAACCCCTTTGGGAAACCCCGGAANNAAACCGAAAACCGAAANN
ACAGACAGACGACGACGACAAAAAGNNNNACAGANNNAACAACAAG
GATACAGATACAGATACAGATACA
"""

alias AA_FASTA_TEST = """
>sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=Frog virus 3 (isolate Goorha) GN=FV3-001R PE=4 SV=1 Variant 1
MMMMMMMMMMMMMMMMMMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPS
EKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLD
AKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHL
EKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDD
SFRKIYTDLGWKFTPL
>sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=Frog virus 3 (isolate Goorha) GN=FV3-001R PE=4 SV=1 Variant 2
MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPS
EKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLD
AKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHL
EKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDD
SFRKIYTDLGWKFTPL
>sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=Frog virus 3 (isolate Goorha) GN=FV3-001R PE=4 SV=1 Variant 3
MAFSAEDVLKEYDRRRRMEALLLSLYYYYYYYYYYYYPPPPPRVQVECPKAPVEWNNPPS
EKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLD
AKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHL
EKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDD
SFRKIYTDLGWKFTPL
"""


def test_line_ascii_search():
    var dir = TemporaryDirectory()
    var input = Path(dir.name) / "input.txt"
    input.write_text(ASCII_TEXT)
    var output = Path(dir.name) / "output.txt"

    var settings = List[SearcherSettings]()
    for algo in List(
        # "naive_exact",
        String("striped-local"),
        String("striped-semi-global"),
        # "basic-local",
        # "basic-semi-global",
    ):
        for num_threads in List(0, 1, 2):
            for max_gpus in List(0, 1, 2):
                for sg in List(SGEF.TTTT, SGEF.FFTT):
                    settings.append(
                        SearcherSettings(
                            files=List[Path](input),  # Fill
                            pattern=List("needle".as_bytes()),  # Fill
                            matrix_kind=MatrixKind.ASCII,
                            score_threshold=0.8,
                            output_file=output,  # Fill
                            gap_open_penalty=3,
                            gap_extension_penalty=1,
                            match_algo=String(algo[]),
                            record_type="line",
                            threads=num_threads[],
                            batch_size=268435456,  # default
                            max_gpus=max_gpus[],
                            tty_info=TTYInfoResult(False, 0, 0),
                            sg_ends_free=sg[],
                            verbose=False,
                        )
                    )

    var expected = """The needle is hidden in here.
I'm not sure what the needle is.
And this is just a very long line that goes on for longer than any of the other lines to make sure needle things are working okay on that front as well when the lines get really long.
"""

    for setting in settings:
        var writer = BufferedWriter(open(output, "w"))
        do_search(setting[], writer^)
        assert_equal(
            output.read_text(),
            expected,
            String("{}, {}, {}, {}").format(
                String(setting[].match_algo),
                String(setting[].threads),
                String(setting[].max_gpus),
                setting[].sg_ends_free.__str__(),
            ),
        )


def test_fasta_actgn_search():
    var dir = TemporaryDirectory()
    var input = Path(dir.name) / "input.txt"
    input.write_text(NUC_FASTA_TEST)
    var output = Path(dir.name) / "output.txt"

    var settings = List[SearcherSettings]()
    for algo in List(
        # "naive_exact",
        String("striped-local"),
        String("striped-semi-global"),
        # "basic-local",
        # "basic-semi-global",
    ):
        for num_threads in List(0, 1, 2):
            for max_gpus in List(0, 1, 2):
                for sg in List(SGEF.TTTT, SGEF.FFTT):
                    settings.append(
                        SearcherSettings(
                            files=List[Path](input),  # Fill
                            pattern=List(
                                "TGACGACGACGACTAATAGNNNNACTGANNNAT".as_bytes()
                            ),  # Fill
                            matrix_kind=MatrixKind.ACTGN,
                            score_threshold=0.99,
                            output_file=output,  # Fill
                            gap_open_penalty=3,
                            gap_extension_penalty=1,
                            match_algo=String(algo[]),
                            record_type="fastx",
                            threads=num_threads[],
                            batch_size=268435456,  # default
                            max_gpus=max_gpus[],
                            tty_info=TTYInfoResult(False, 0, 0),
                            sg_ends_free=sg[],
                            verbose=False,
                        )
                    )

    var expected = """>Human_DNA With a Comment
ACTGACTGACGACGACGACTAATAGNNNNACTGANNNATCATCTAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGACTGACTGACGACGACGACTAATAGNNNNACTGANGNATCATCTAGCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCACTGACTGACGACGACGACTAATAGNNNNACTGANNNATCATCTAGACTGACTGACGACGACGACTAATAG
"""

    for setting in settings:
        var writer = BufferedWriter(open(output, "w"))
        do_search(setting[], writer^)
        var found = output.read_text()
        assert_equal(
            found.upper(),
            expected.upper(),
            String("max_gpus {}, num_threads {}, algo {}").format(
                setting[].max_gpus, setting[].threads, setting[].match_algo
            ),
        )


def test_fasta_blosum62_search():
    var dir = TemporaryDirectory()
    var input = Path(dir.name) / "input.txt"
    input.write_text(AA_FASTA_TEST)
    var output = Path(dir.name) / "output.txt"

    var settings = List[SearcherSettings]()
    for algo in List(
        # "naive_exact",
        String("striped-local"),
        String("striped-semi-global"),
        # "basic-local",
        # "basic-semi-global",
    ):
        for num_threads in List(0, 1, 2):
            for max_gpus in List(0, 1, 2):
                for sg in List(SGEF.TTTT, SGEF.FFTT):
                    settings.append(
                        SearcherSettings(
                            files=List[Path](input),  # Fill
                            pattern=List(
                                "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPS"
                                .as_bytes()
                            ),  # Fill
                            matrix_kind=MatrixKind.BLOSUM62,
                            score_threshold=0.95,
                            output_file=output,  # Fill
                            gap_open_penalty=3,
                            gap_extension_penalty=1,
                            match_algo=String(algo[]),
                            record_type="fastx",
                            threads=num_threads[],
                            batch_size=268435456,  # default
                            max_gpus=max_gpus[],
                            tty_info=TTYInfoResult(False, 0, 0),
                            sg_ends_free=sg[],
                            verbose=False,
                        )
                    )

    var expected = """>sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=Frog virus 3 (isolate Goorha) GN=FV3-001R PE=4 SV=1 Variant 2
MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL
"""

    for setting in settings:
        var writer = BufferedWriter(open(output, "w"))
        do_search(setting[], writer^)
        assert_equal(
            output.read_text().upper(),
            expected.upper(),
            String("max_gpus {}, num_threads {}, algo {}").format(
                setting[].max_gpus, setting[].threads, setting[].match_algo
            ),
        )


def main():
    test_line_ascii_search()
    test_fasta_actgn_search()
    test_fasta_blosum62_search()
