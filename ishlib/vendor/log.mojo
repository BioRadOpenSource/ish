import sys
from sys.param_env import env_get_string
from utils import write_args, StringSlice


@value
struct LogLevel:
    var value: Int
    alias Nolog = Self(100000)
    alias Error = Self(500)
    alias Warn = Self(400)
    alias Info = Self(300)
    alias Timing = Self(200)
    alias Debug = Self(100)

    @staticmethod
    fn from_str(level: StringLiteral) -> Self:
        if level.lower() == "info":
            return Self.Info
        elif level.lower() == "debug":
            return Self.Debug
        elif level.lower() == "error":
            return Self.Error
        elif level.lower() == "warn":
            return Self.Warn
        elif level.lower() == "timing":
            return Self.Timing
        else:
            return Self.Nolog

    fn __is__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.Info:
            writer.write("Info")
        elif self is Self.Debug:
            writer.write("Debug")
        elif self is Self.Error:
            writer.write("Error")
        elif self is Self.Warn:
            writer.write("Warn")
        elif self is Self.Timing:
            writer.write("Timing")
        else:
            writer.write("Uknown")


alias RED = "\x1b[31m"
alias GREEN = "\x1b[32m"
alias YELLOW = "\x1b[33m"
alias PURPLE = "\x1b[35m"
alias BLUE = "\x1b[34m"
alias RESET = "\x1b[0m"


@value
struct Logger[colorize: Bool = True](CollectionElement):
    alias LEVEL: LogLevel = LogLevel.from_str(
        env_get_string["ISH_LOG_LEVEL", "nolog"]()
    )

    @always_inline
    @staticmethod
    fn _is_disabled[level: LogLevel]() -> Bool:
        return Self.LEVEL.value > level.value

    @always_inline
    @staticmethod
    fn info[
        *Ts: Writable
    ](
        *values: *Ts,
        sep: StringSlice[StaticConstantOrigin] = StringSlice(" "),
        end: StringSlice[StaticConstantOrigin] = StringSlice("\n"),
    ):
        @parameter
        if Self._is_disabled[LogLevel.Info]():
            return

        var stderr = sys.stderr

        stderr.write("[")
        LogLevel.Info.write_to(stderr)
        stderr.write("] ")
        write_args(stderr, values, sep=sep, end=end)

    @always_inline
    @staticmethod
    fn timing[
        *Ts: Writable
    ](
        *values: *Ts,
        sep: StringSlice[StaticConstantOrigin] = StringSlice(" "),
        end: StringSlice[StaticConstantOrigin] = StringSlice("\n"),
    ):
        @parameter
        if Self._is_disabled[LogLevel.Timing]():
            return

        var stderr = sys.stderr

        @parameter
        if colorize:
            stderr.write(BLUE)

        stderr.write("[")
        LogLevel.Timing.write_to(stderr)
        stderr.write("] ")
        write_args(stderr, values, sep=sep, end=end)

        @parameter
        if colorize:
            stderr.write(RESET)

    @always_inline
    @staticmethod
    fn debug[
        *Ts: Writable
    ](
        *values: *Ts,
        sep: StringSlice[StaticConstantOrigin] = StringSlice(" "),
        end: StringSlice[StaticConstantOrigin] = StringSlice("\n"),
    ):
        @parameter
        if Self._is_disabled[LogLevel.Debug]():
            return

        var stderr = sys.stderr
        stderr.write("[")
        LogLevel.Debug.write_to(stderr)
        stderr.write("] ")
        write_args(stderr, values, sep=sep, end=end)

    @always_inline
    @staticmethod
    fn error[
        *Ts: Writable
    ](
        *values: *Ts,
        sep: StringSlice[StaticConstantOrigin] = StringSlice(" "),
        end: StringSlice[StaticConstantOrigin] = StringSlice("\n"),
    ):
        @parameter
        if Self._is_disabled[LogLevel.Error]():
            return

        var stderr = sys.stderr

        @parameter
        if colorize:
            stderr.write(RED)
        stderr.write("[")
        LogLevel.Error.write_to(stderr)
        stderr.write("] ")
        write_args(stderr, values, sep=sep, end=end)

        @parameter
        if colorize:
            stderr.write(RESET)

    @always_inline
    @staticmethod
    fn warn[
        *Ts: Writable
    ](
        *values: *Ts,
        sep: StringSlice[StaticConstantOrigin] = StringSlice(" "),
        end: StringSlice[StaticConstantOrigin] = StringSlice("\n"),
    ):
        @parameter
        if Self._is_disabled[LogLevel.Warn]():
            return

        var stderr = sys.stderr

        @parameter
        if colorize:
            stderr.write(YELLOW)
        stderr.write("[")
        LogLevel.Warn.write_to(stderr)
        stderr.write("] ")
        write_args(stderr, values, sep=sep, end=end)

        @parameter
        if colorize:
            stderr.write(RESET)
