# This code is MIT Licensed by Seth Stadick

from ishlib.vendor.log import Logger
from ishlib.vendor.kseq import BufferedReader, FastxReader
from ishlib.vendor.zlib import GZFile

from ExtraMojo.io.buffered import BufferedWriter

from runtime.asyncrt import DeviceContextPtr, TaskGroup, parallelism_level
from os.atomic import Atomic
from sys import stdout
from time import perf_counter, sleep


struct BinarySemaphore:
    var state: Atomic[DType.int8]

    fn __init__(out self, initial: Bool):
        self.state = Atomic[DType.int8](Int8(Int(initial)))

    fn check(mut self) -> Bool:
        return Bool(self.state.load())

    fn acquire(self):
        while True:
            var expected = Int8(1)
            if self.state.compare_exchange_weak(expected, Int8(0)):
                return
            sleep(0.001)  # backoff to prevent busy spinning

    fn release(self):
        # Atomic.store doesn't seem to do what I think it should
        self.state.max(1)


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
        print("Par level:", parallelism_level())
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
        var shutdown = Atomic[DType.uint32](0)

        var signal_to_generate = BinarySemaphore(1)
        var signal_to_process = BinarySemaphore(0)

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

            # TODO: this almost works, there is something wrong with shutdown sequencing and the last buffer doesn't get processed

            while True:
                var start = perf_counter()
                if fill(reader, self._buffers[self._current_fill_buffer]) < 0:
                    _ = shutdown.fetch_add(1)

                var end = perf_counter()
                print("Fill in:", end - start)
                print(len(self._buffers[self._current_fill_buffer]))

                # Wait till for signal to generate more data
                signal_to_generate.acquire()

                # Swap buffers
                self._swap_buffers()

                # Send signal to processing to go ahead
                print(
                    "Signal to process",
                    len(self._buffers[self._current_process_buffer]),
                )
                signal_to_process.release()

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

            # this seems like a hack that I can't trust
            var saw_shutdown_last_loop = False

            while True:
                signal_to_process.acquire()

                var start = perf_counter()
                if (
                    process(writer, self._buffers[self._current_process_buffer])
                    < 0
                ):
                    _ = shutdown.fetch_add(1)
                var end = perf_counter()
                print("Process in:", end - start)

                signal_to_generate.release()
                if shutdown.load() > 0:
                    if not signal_to_process.check() and saw_shutdown_last_loop:
                        print("shutting down processor")
                        break
                    saw_shutdown_last_loop = True

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
