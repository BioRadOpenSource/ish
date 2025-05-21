# This code is MIT Licensed by Seth Stadick

from ishlib.vendor.log import Logger
from ishlib.vendor.kseq import BufferedReader, FastxReader
from ishlib.vendor.zlib import GZFile

from ExtraMojo.io.buffered import BufferedWriter

from runtime.asyncrt import DeviceContextPtr, TaskGroup, parallelism_level
from os.atomic import Atomic
from sys import stdout


struct DoubleBuffer[
    T: Copyable & Movable,
    fill: fn (
        mut reader: FastxReader[GZFile, False], mut fill_buffer: List[T]
    ) capturing -> Int,
    process: fn (
        mut writer: BufferedWriter[FileDescriptor], mut process_buffer: List[T]
    ) capturing -> Int,
]:
    var _buffers: InlineArray[List[T], size=2]
    var _current_fill_buffer: UInt
    var _current_process_buffer: UInt

    fn __init__(out self, *, capacity: Int = 0):
        self._buffers = __type_of(self._buffers)(
            List[T](capacity=capacity), List[T](capacity=capacity)
        )
        self._current_fill_buffer = 0
        self._current_process_buffer = 1

    fn _swap_buffers(mut self):
        var tmp = self._current_fill_buffer
        self._current_fill_buffer = self._current_process_buffer
        self._current_process_buffer = tmp

    fn run(mut self, owned file_to_read: String):
        """Alternate between two buffers as the "fill" buffer and "work" buffer.
        """
        var fill_done = Atomic[DType.uint32](0)
        var process_done = Atomic[DType.uint32](1)
        var shutdown = Atomic[DType.uint32](0)

        @parameter
        @always_inline
        async fn thread_filler():
            # TODO: When we have a generic "Read" trait, this can go away and a closure can be passed in to
            # create this.
            var reader: FastxReader[GZFile, read_comment=False]
            try:
                reader = FastxReader[read_comment=False](
                    BufferedReader(GZFile(file_to_read^, "r"))
                )
            except:
                Logger.error("Failed to open", file_to_read, "for reading")
                return

            while True:
                if process_done.load() == 1:
                    _ = process_done.fetch_sub(1)
                    if (
                        fill(reader, self._buffers[self._current_fill_buffer])
                        < 0
                    ):
                        _ = shutdown.fetch_add(1)
                    _ = fill_done.fetch_add(1)

                if shutdown.load() > 0:
                    print("shutting down filler")
                    break
            _ = reader^

        @parameter
        @always_inline
        async fn thread_processor():
            var writer: BufferedWriter[FileDescriptor]
            try:
                writer = BufferedWriter(stdout)
            except:
                Logger.error("Failed to open stdout for writing")
                return

            while True:
                if fill_done.load() == 1:
                    self._swap_buffers()
                    _ = fill_done.fetch_sub(1)
                    if (
                        process(
                            writer, self._buffers[self._current_process_buffer]
                        )
                        < 0
                    ):
                        _ = shutdown.fetch_add(1)
                    _ = process_done.fetch_add(1)

                if shutdown.load() > 0 and fill_done.load() == 0:
                    print("shutting down processor")
                    break

            try:
                writer.flush()
                writer.close()
            except:
                Logger.error("Failed to flush and close stdout")

        var tg = TaskGroup()
        tg.create_task(thread_filler())
        tg.create_task(thread_processor())
        tg.wait()


# def main():
#     var source = List[Int](1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

#     @parameter
#     fn filler(mut fill_buffer: List[Int]) capturing -> Int:
#         if len(source) > 0:
#             fill_buffer.append(source.pop())
#             return 0
#         else:
#             return -1

#     @parameter
#     fn processor(mut process_buffer: List[Int]) capturing -> Int:
#         if len(process_buffer) > 0:
#             print(process_buffer[0])
#             process_buffer.clear()
#         return 0

#     var dblbfr = DoubleBuffer[Int, filler, processor]()
#     dblbfr.run()

#     _ = source
