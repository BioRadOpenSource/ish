"""
Mojo bindings for zlib.

Note that `GZFile` will auto detect compression. If the file is not compressed
it pass through to a normal reader that is quite fast when paired with a
BufferedReader.

```mojo
from ishlib.vendor.zlib import GZFile
from utils import StringSlice

fn main() raises:
    # Example usage
    try:
        # Open a gzip file for reading
        var file = GZFile("example.txt.gz", "rb")

        # Read and print data
        var buffer = List[UInt8](0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        var data = file.unbuffered_read(buffer)
        print("Bytes read:", data)
        print("Read data:", String(StringSlice(unsafe_from_utf8=buffer)))

    except e:
        print("Error:", e)
```
"""

from memory import memset_zero, UnsafePointer
from sys import ffi
from sys.info import CompilationTarget

from ishlib.vendor.kseq import KRead

# Constants for zlib return codes
alias Z_OK = 0
alias Z_STREAM_END = 1
alias Z_NEED_DICT = 2
alias Z_ERRNO = -1
alias Z_STREAM_ERROR = -2
alias Z_DATA_ERROR = -3
alias Z_MEM_ERROR = -4
alias Z_BUF_ERROR = -5
alias Z_VERSION_ERROR = -6

# Type aliases for C types
alias c_void_ptr = UnsafePointer[UInt8]
alias c_char_ptr = UnsafePointer[Int8]
alias c_uint = UInt32
alias c_int = Int32

# Define function signatures for zlib functions
alias gzopen_fn_type = fn (filename: c_char_ptr, mode: c_char_ptr) -> c_void_ptr
alias gzclose_fn_type = fn (file: c_void_ptr) -> c_int
alias gzread_fn_type = fn (
    file: c_void_ptr, buf: c_void_ptr, len: c_uint
) -> c_int


@fieldwise_init
struct ZLib(Movable):
    """Wrapper for zlib library functions."""

    var lib_handle: ffi.DLHandle

    @staticmethod
    fn _get_libname() -> StaticString:
        @parameter
        if CompilationTarget.is_macos():
            return "libz.dylib"
        else:
            return "libz.so"

    fn __init__(out self) raises:
        """Initialize zlib wrapper."""
        self.lib_handle = ffi.DLHandle(Self._get_libname())

    fn gzopen(
        self, mut filename: String, mut mode: String
    ) raises -> c_void_ptr:
        """Open a gzip file."""
        # Get function pointer
        var func = self.lib_handle.get_function[gzopen_fn_type]("gzopen")

        # Call the function
        var result = func(filename.unsafe_cstr_ptr(), mode.unsafe_cstr_ptr())

        return result

    fn gzclose(self, file: c_void_ptr) -> c_int:
        """Close a gzip file."""
        var func = self.lib_handle.get_function[gzclose_fn_type]("gzclose")
        return func(file)

    fn gzread(
        self, file: c_void_ptr, buffer: c_void_ptr, length: c_uint
    ) -> c_int:
        """Read from a gzip file."""
        var func = self.lib_handle.get_function[gzread_fn_type]("gzread")
        return func(file, buffer, length)


struct GZFile(KRead):
    """Helper class for gzip file operations."""

    var handle: c_void_ptr
    var lib: ZLib
    var filename: String
    var mode: String

    fn __init__(out self, filename: String, mode: String) raises:
        """Open a gzip file."""
        self.lib = ZLib()
        # Note: must keep filename and mode because gzopen takes a ref to them and they need to live as long as the file is open.
        self.filename = filename
        self.mode = mode
        self.handle = self.lib.gzopen(self.filename, self.mode)
        if self.handle == c_void_ptr():
            raise Error("Failed to open gzip file: " + filename)

    fn __del__(deinit self):
        """Close the file when the object is destroyed."""
        if self.handle != c_void_ptr():
            _ = self.lib.gzclose(self.handle)

    fn __moveinit__(out self, deinit other: Self):
        self.handle = other.handle
        self.lib = other.lib^
        self.filename = other.filename^
        self.mode = other.mode^

    fn unbuffered_read[
        o: MutableOrigin
    ](mut self, buffer: Span[UInt8, o]) raises -> Int:
        """Read data from the gzip file.

        Args:
            buffer: The buffer to read data into, attempts to fill the buffer.

        Returns:
            The number of bytes read, or an error code if it's less than zero.
        """

        var bytes_read = self.lib.gzread(
            self.handle, buffer.unsafe_ptr(), c_uint(len(buffer))
        )

        if bytes_read < 0:
            raise Error("Error reading from gzip file: " + String(bytes_read))
        return Int(bytes_read)
