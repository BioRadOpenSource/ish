"""
```mojo
fn main() raises:
    var tty = TTYInfo()

    if tty.is_a_tty(STDOUT_FD):
        print("stdout is a TTY")
        (cols, rows) = tty.get_tty_size(STDOUT_FD)
        print("Terminal size (stdout):", cols, "cols x", rows, "rows")
    else:
        print("stdout is not a TTY")

    if tty.is_a_tty(STDERR_FD):
        print("stderr is a TTY")
        (cols, rows) = tty.get_tty_size(STDERR_FD)
        print("Terminal size (stderr):", cols, "cols x", rows, "rows")
    else:
        print("stderr is not a TTY")
```
"""

from sys import ffi
from sys.info import CompilationTarget
from memory import UnsafePointer

alias c_int = Int32
alias c_ulong = UInt64
alias c_ushort = UInt16
alias c_void_ptr = UnsafePointer[UInt8]


alias STDOUT_FD: Int = 1
"""File descriptor for STDOUT."""
alias STDERR_FD: Int = 2
"""File descriptor for STDERR."""


@parameter
fn _TIOCGWINSZ() -> c_ulong:
    """ioctl command for window size."""
    if CompilationTarget.is_macos():
        return 0x40087468
    else:
        return 0x5413


alias _ioctl_fn_type = fn (
    fd: c_int, request: c_ulong, argp: c_void_ptr
) -> c_int
"""ioctl function type"""


@fieldwise_init
@register_passable("trivial")
struct Info:
    """Info on the TTY for a file descriptor."""

    var is_a_tty: Bool
    """Bool indicating if there is an attached TTY or not."""
    var rows: UInt16
    """The number of rows of the TTY (vertacle)."""
    var cols: UInt16
    """The number of cols of the TTY (horizontal)."""


@fieldwise_init
struct TTYInfo:
    var lib_handle: ffi.DLHandle

    @staticmethod
    fn _get_libname() -> StaticString:
        @parameter
        if CompilationTarget.is_macos():
            return "libc.dylib"
        else:
            return "libc.so.6"

    fn __init__(out self) raises:
        self.lib_handle = ffi.DLHandle(Self._get_libname())

    fn info(self, fd: Int) raises -> Info:
        """Collect all the info on the TTY.

        Args:
            fd: The file descriptor to check.

        Returns:
            An Info object.
        """
        var is_tty = self.is_a_tty(fd)
        # (rows, cols) = self.get_tty_size(fd)
        return Info(is_tty, 0, 0)

    fn is_a_tty(self, fd: Int) -> Bool:
        """Check if the file descriptor is attached to a TTY.

        Args:
            fd: The file descriptor to check.

        Returns:
            True if it is attached to a TTY, False if not.
        """
        var func = self.lib_handle.get_function[fn (fd: c_int) -> c_int](
            "isatty"
        )
        return func(c_int(fd)) != 0

    # TODO: this was just lucky and working in 25.2, in 25.3, something has changed.
    # Shelving till we have a repr(C) equivalent
    # fn get_tty_size(self, fd: Int) -> (Int, Int):
    #     """Get the size of a TTY.

    #     If it is not attached the returned rows and cols will be -1.

    #     Args:
    #         fd: The file descriptor to check.

    #     Returns:
    #         (rows, cols) as the size, -1 if not attached.
    #     """
    #     var func = self.lib_handle.get_function[_ioctl_fn_type]("ioctl")

    #     # Allocate a buffer for the struct winsize (4 * 2-byte ushort = 8 bytes)
    #     var buf = InlineArray[UInt8, size=8](fill=0)
    #     var ptr = buf.unsafe_ptr()

    #     var result = func(c_int(fd), _TIOCGWINSZ(), ptr)
    #     if result < 0:
    #         return (-1, -1)

    #     # Read cols and rows from buffer (0-1 = rows, 2-3 = cols)
    #     var rows = Int(UInt16(ptr[0]) | UInt16(ptr[1]) << 8)
    #     var cols = Int(UInt16(ptr[2]) | UInt16(ptr[3]) << 8)
    #     _ = buf
    #     return (cols, rows)
