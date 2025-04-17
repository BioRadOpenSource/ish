from bit.bit import is_power_of_two, next_power_of_two
from collections import Optional
from sys.info import num_physical_cores

from ExtraMojo.cli.parser import OptParser, OptConfig, OptKind

from ishlib.matcher.alignment.scoring_matrix import MatrixKind
from ishlib.vendor.tty_info import TTYInfo, Info as TTYInfoResult, STDOUT_FD


@value
struct SearcherSettings:
    """Settings for the ish searcher."""

    var files: List[String]
    """The files to search for matches."""
    var pattern: List[UInt8]
    """The pattern to search for."""
    var matrix_kind: MatrixKind
    """The scorign matrix to use."""
    var score_threshold: Float32
    """The minimum score needed to return a match."""

    var gap_open_penalty: Int
    var gap_extension_penalty: Int

    var match_algo: String
    var record_type: String
    var threads: UInt
    var batch_size: UInt
    var max_gpus: UInt
    var tty_info: TTYInfoResult

    @staticmethod
    fn from_args() raises -> Optional[Self]:
        var parser = OptParser(
            name="ish", description="Search for inexact patterns in files."
        )
        parser.add_opt(
            OptConfig(
                "pattern",
                OptKind.StringLike,
                default_value=None,
                description="The pattern to search for.",
            )
        )
        parser.add_opt(
            OptConfig(
                "scoring-matrix",
                OptKind.StringLike,
                default_value=String("ascii"),
                # fmt: off
                description=(
                "The scoring matrix to use.\n"
                "\tascii: does no encoding of input bytes, matches are 2, mismatch is -2.\n"
                "\tblosum62: encodes searched inputs as amino acids and uses the classic Blosum62 scoring matrix.\n"
                "\tactgn: encodes searched inputs as nucleotides, matches are 2, mismatch is -2, Ns match anything\n"
                # TODO: support iupac
                )
                # fmt: on
            )
        )
        parser.add_opt(
            OptConfig(
                "score",
                OptKind.FloatLike,
                default_value=String("0.9"),
                description=(
                    "The min score needed to return a match. Results >= this"
                    " value will be returned."
                ),
            )
        )
        parser.add_opt(
            OptConfig(
                "gap-open",
                OptKind.IntLike,
                default_value=String("3"),
                description="Score penalty for opening a gap.",
            )
        )
        parser.add_opt(
            OptConfig(
                "gap-extend",
                OptKind.IntLike,
                default_value=String("1"),
                description="Score penalty for extending a gap.",
            )
        )
        parser.add_opt(
            OptConfig(
                "match-algo",
                OptKind.StringLike,
                default_value=String("ssw"),
                description=(
                    "The algorithm to use for matching: [naive_exact,"
                    " striped-local, basic-local, basic-global,"
                    " basic-semi-global, striped-semi-global]"
                ),
            )
        )
        parser.add_opt(
            OptConfig(
                "record-type",
                OptKind.StringLike,
                default_value=String("line"),
                description="The input record type: [line, fasta]",
            )
        )
        parser.add_opt(
            OptConfig(
                "threads",
                OptKind.IntLike,
                default_value=String(num_physical_cores()),
                description=(
                    "The number of threads to use. Defaults to the number of"
                    " physical cores."
                ),
            )
        )
        parser.add_opt(
            OptConfig(
                "batch-size",
                OptKind.IntLike,
                default_value=String("268435456"),
                # TODO: elaborate on this for GPU batch sizing, with multiple devices.
                description=(
                    "The number of bytes in a parallel processing batch. Note"
                    " that this may use 2-3x this amount to account for"
                    " intermediate transfer buffers."
                ),
            )
        )
        parser.add_opt(
            OptConfig(
                "max-gpus",
                OptKind.IntLike,
                default_value=String("1"),
                description=(
                    "The max number of GPUs to try to use. If set to 0 this"
                    " will ignore any found GPUs. In general, if you have only"
                    " one query then there won't be much using more than 1 GPU."
                    " GPUs won't always be faster than CPU parallelization"
                    " depeding on the profile of data you are working with."
                ),
            )
        )

        parser.expect_at_least_n_args(
            1, "Files to search for the given pattern."
        )

        try:
            var opts = parser.parse_sys_args()
            if opts.get_bool("help"):
                print(opts.get_help_message()[])
                return None

            var pattern = List(opts.get_string("pattern").as_bytes())
            var matrix_kind = MatrixKind.from_str(
                opts.get_string("scoring-matrix")
            )
            var score = opts.get_float("score")
            var gap_open = abs(opts.get_int("gap-open"))
            var gap_extend = abs(opts.get_int("gap-extend"))
            var match_algo = opts.get_string("match-algo")
            var record_type = opts.get_string("record-type")
            var threads = opts.get_int("threads")
            if threads <= 0:
                raise "Threads must be >= 1."
            var batch_size = opts.get_int("batch-size")
            if not is_power_of_two(batch_size):
                var next_power_of_two = next_power_of_two(batch_size)
                raise "Batch size is not a power of two, try: " + String(
                    next_power_of_two
                )
            if batch_size < 1024 * 10:
                raise "Batch size too small, try"
            var files = opts.args
            if len(files) == 0:
                raise "Expected files, found none."

            var max_gpus = opts.get_int("max-gpus")

            var tty = TTYInfo()

            return Self(
                files,
                pattern,
                matrix_kind,
                Float32(score),
                gap_open,
                gap_extend,
                match_algo,
                record_type,
                threads,
                batch_size,
                max_gpus,
                tty.info(STDOUT_FD),
            )
        except e:
            print(parser.help_msg())
            print(e)
            return None
