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

alias ISSUE_50 = """
>test1
AGCTACGACGACT
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


@fieldwise_init
struct TestCase(Copyable, Movable):
    var in_data: String
    var pattern: List[UInt8]
    var expected: String
    var matrix_kind: MatrixKind
    var record_type: String

    @staticmethod
    fn cases() -> List[Self]:
        return List(
            TestCase(
                in_data=ASCII_TEXT,
                pattern=List("needle".as_bytes()),
                expected="""The needle is hidden in here.
I'm not sure what the needle is.
And this is just a very long line that goes on for longer than any of the other lines to make sure needle things are working okay on that front as well when the lines get really long.
""",
                matrix_kind=MatrixKind.ASCII,
                record_type="line",
            ),
            TestCase(
                in_data=NUC_FASTA_TEST,
                pattern=List("aTGACGACGACGACTAATAGNNNNACTGANNNAT".as_bytes()),
                expected=""">Human_DNA With a Comment
ACTGACTGACGACGACGACTAATAGNNNNACTGANNNATCATCTAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGACTGACTGACGACGACGACTAATAGNNNNACTGANGNATCATCTAGCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCACTGACTGACGACGACGACTAATAGNNNNACTGANNNATCATCTAGACTGACTGACGACGACGACTAATAG
""",
                matrix_kind=MatrixKind.ACTGN,
                record_type="fastx",
            ),
            TestCase(
                in_data=NUC_FASTA_TEST,
                pattern=List("aTGACGACGACGACTAATAGNNNNACTGANNNAT".as_bytes()),
                expected=""">Human_DNA With a Comment
ACTGACTGACGACGACGACTAATAGNNNNACTGANNNATCATCTAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGACTGACTGACGACGACGACTAATAGNNNNACTGANGNATCATCTAGCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCACTGACTGACGACGACGACTAATAGNNNNACTGANNNATCATCTAGACTGACTGACGACGACGACTAATAG
""",
                matrix_kind=MatrixKind.ACTGN0,
                record_type="fastx",
            ),
            # ISSUE 50 seems to be seg-fault related on AVX512
            TestCase(
                in_data=ISSUE_50,
                pattern=List("AGCATCG".as_bytes()),
                expected="""""",
                matrix_kind=MatrixKind.ACTGN,
                record_type="fastx",
            ),
            TestCase(
                in_data=AA_FASTA_TEST,
                pattern=List(
                    "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPS"
                    .as_bytes()
                ),
                expected=""">sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=Frog virus 3 (isolate Goorha) GN=FV3-001R PE=4 SV=1 Variant 2
MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL
""",
                matrix_kind=MatrixKind.BLOSUM62,
                record_type="fastx",
            ),
        )


def test_cases():
    for c in TestCase.cases():
        var dir = TemporaryDirectory()
        var input = Path(dir.name) / "input.txt"
        input.write_text(c.in_data)
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
                                pattern=c.pattern,  # Fill
                                matrix_kind=c.matrix_kind,
                                score_threshold=0.8,
                                output_file=output,  # Fill
                                gap_open_penalty=3,
                                gap_extension_penalty=1,
                                match_algo=String(algo),
                                record_type=c.record_type,
                                threads=num_threads,
                                batch_size=268435456,  # default
                                max_gpus=max_gpus,
                                tty_info=TTYInfoResult(False, 0, 0),
                                sg_ends_free=sg,
                                verbose=False,
                            )
                        )

        for setting in settings:
            var info = String("{}, {}, {}, {}, {}, {}").format(
                String(c.matrix_kind),
                String(c.record_type),
                String(setting.match_algo),
                String(setting.threads),
                String(setting.max_gpus),
                setting.sg_ends_free.__str__(),
            )
            # print("Doing test for:", info)
            var writer = BufferedWriter(open(output, "w"))
            do_search(setting, writer^)
            var out = output.read_text()
            if c.record_type == "fastx":
                assert_equal(out.upper(), c.expected.upper(), info)
            else:
                assert_equal(
                    out,
                    c.expected,
                    String("{}, {}, {}, {}, {}, {}").format(
                        String(c.matrix_kind),
                        String(c.record_type),
                        String(setting.match_algo),
                        String(setting.threads),
                        String(setting.max_gpus),
                        setting.sg_ends_free.__str__(),
                    ),
                )


def main():
    test_cases()
