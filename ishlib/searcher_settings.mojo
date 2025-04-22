from bit.bit import is_power_of_two, next_power_of_two
from collections import Optional
from sys.info import num_physical_cores
from pathlib.path import Path, cwd

from ExtraMojo.cli.parser import OptParser, OptConfig, OptKind

from ishlib.matcher.alignment.scoring_matrix import MatrixKind
from ishlib.vendor.tty_info import TTYInfo, Info as TTYInfoResult, STDOUT_FD
from ishlib.vendor.walk_dir import walk_dir


@value
struct SearcherSettings:
    """Settings for the ish searcher."""

    var files: List[Path]
    """The files to search for matches."""
    var pattern: List[UInt8]
    """The pattern to search for."""
    var matrix_kind: MatrixKind
    """The scorign matrix to use."""
    var score_threshold: Float32
    """The minimum score needed to return a match."""
    var output_file: Path
    """The file to write the output to."""

    var gap_open_penalty: Int
    var gap_extension_penalty: Int

    var match_algo: String
    var record_type: String
    var threads: UInt
    var batch_size: UInt
    var max_gpus: UInt
    var tty_info: TTYInfoResult

    fn is_output_stdout(read self) -> Bool:
        return self.output_file == "/dev/stdout"

    @staticmethod
    fn from_args() raises -> Optional[Self]:
        var parser = OptParser(
            name="ish", description="Search for inexact patterns in files."
        )

        parser.add_opt(
            OptConfig(
                "scoring-matrix",
                OptKind.StringLike,
                default_value=String("ascii"),
                # fmt: off
                description=(
                "The scoring matrix to use.\n"
                "\t\tascii: does no encoding of input bytes, matches are 2, mismatch is -2.\n"
                "\t\tblosum62: encodes searched inputs as amino acids and uses the classic Blosum62 scoring matrix.\n"
                "\t\tactgn: encodes searched inputs as nucleotides, matches are 2, mismatch is -2, Ns match anything\n"
                # TODO: support iupac
                )
                # fmt: on
            )
        )
        parser.add_opt(
            OptConfig(
                "score",
                OptKind.FloatLike,
                default_value=String("0.8"),
                description=(
                    "The min score needed to return a match. Results >= this"
                    " value will be returned. The score is the found alignment"
                    " score / the optimal score for the given scoring matrix"
                    " and gap-open / gap-extend penalty."
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
                default_value=String("striped-semi-global"),
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
                default_value=String("0"),
                description=(
                    "The max number of GPUs to try to use. If set to 0 this"
                    " will ignore any found GPUs. In general, if you have only"
                    " one query then there won't be much using more than 1 GPU."
                    " GPUs won't always be faster than CPU parallelization"
                    " depending on the profile of data you are working with."
                ),
            )
        )
        parser.add_opt(
            OptConfig(
                "output-file",
                OptKind.StringLike,
                default_value=String("/dev/stdout"),
                description=(
                    "The file to write the output to, defaults to stdout."
                ),
            )
        )

        parser.expect_at_least_n_args(
            1,
            (
                "Pattern to search for, then any number of files or directories"
                " to search."
            ),
        )

        try:
            var opts = parser.parse_sys_args()
            if opts.get_bool("help"):
                print(opts.get_help_message()[])
                return None

            if len(opts.args) == 0:
                raise "Missing pattern argument."
            var pattern = List(opts.args[0].as_bytes())

            var output = Path(opts.get_string("output-file"))
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
            var files = List[String]()
            if len(opts.args) == 1:
                files.append(String(cwd()))
            else:
                for i in range(1, len(opts.args)):
                    files.append(opts.args[i])

            var expanded_files = expand_files_to_search(files)

            var max_gpus = opts.get_int("max-gpus")

            var tty = TTYInfo()

            return Self(
                expanded_files,
                pattern,
                matrix_kind,
                Float32(score),
                output,
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


fn expand_files_to_search(files: List[String]) raises -> List[Path]:
    """Expand the passed in files or directories to search."""
    var out = List[Path]()

    for p in files:
        var path = Path(p[])
        if path.is_dir():
            out.extend(walk_dir[ignore_dot_files=True](path))
        else:
            out.append(path)
    return out
