from multiprocessing.pool import ThreadPool
from threading import Event, Semaphore, Thread
from typing import Callable

from pdp import Pipeline
from pdp.interface import ComponentDescription, Source, One2One
from pdp.base import SourceExhausted, StopEvent
from loky import ProcessPoolExecutor, Future


def start_iter(q_in, q_out, stop_event: Event, *, transform: Callable, n_workers: int, args, kwargs):
    def source():
        try:
            for value in iter(q_in.get, SourceExhausted()):
                yield value
                q_in.task_done()

            q_in.task_done()
        except StopEvent:
            pass

    def target():
        try:
            for value in transform(source(), *args, **kwargs):
                q_out.put(value)

            # wait for other processes
            q_in.join()
            q_out.put(SourceExhausted())

        except StopEvent:
            pass
        except BaseException:
            stop_event.set()
            raise

    ThreadPool(n_workers, target).close()


def start_loky(q_in, q_out, stop_event: Event, *, transform: Callable, n_workers: int, args, kwargs):
    def target():
        def done(future: Future):
            try:
                q_out.put(future.result())
                q_in.task_done()

            except BaseException:
                stop_event.set()
                raise

            finally:
                counter.release()

        # start worker
        executor = ProcessPoolExecutor(n_workers)
        counter = Semaphore(n_workers)
        wait = True

        try:
            for value in iter(q_in.get, SourceExhausted()):
                counter.acquire()
                executor.submit(transform, value, *args, **kwargs).add_done_callback(done)

            # wait for other processes
            q_in.task_done()
            q_in.join()
            q_out.put(SourceExhausted())

        except StopEvent:
            pass
        except BaseException:
            wait = False
            stop_event.set()
            raise

        finally:
            executor.shutdown(wait=wait)

    Thread(target=target).start()
