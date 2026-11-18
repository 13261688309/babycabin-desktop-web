"""Cancelable absolute HTTP read deadlines, including headers and chunk framing."""

from __future__ import annotations

import http.client
import io
import math
import socket
import threading
import time


class HttpReadDeadline:
    def __init__(self):
        self._cancelled = threading.Event()
        self._until = None

    def begin(self, seconds):
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('HTTP read deadline must be finite and positive')
        self._until = time.monotonic() + seconds
        self.remaining()

    def remaining(self):
        if self._cancelled.is_set():
            raise OSError('HTTP camera read cancelled')
        if self._until is None:
            raise RuntimeError('HTTP read deadline was not started')
        remaining = self._until - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('HTTP camera read deadline exceeded')
        return remaining

    def cancel(self):
        self._cancelled.set()

    def response(self, sock, **options):
        return http.client.HTTPResponse(_ResponseSocket(sock, self), **options)


class _ResponseSocket:
    def __init__(self, sock, deadline):
        self.sock, self.deadline = sock, deadline

    def makefile(self, mode):
        if mode != 'rb':
            raise ValueError('Only binary HTTP response reads are supported')
        return io.BufferedReader(_DeadlineRawReader(self.sock, self.deadline))


class _DeadlineRawReader(io.RawIOBase):
    def __init__(self, sock, deadline):
        super().__init__()
        self._sock, self._deadline = sock, deadline
        # Keep the standard socket file reference alive when HTTPConnection closes
        # a Connection: close socket before its response body has been consumed.
        self._holder = sock.makefile('rb', buffering=0)

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            self._sock.settimeout(min(.1, self._deadline.remaining()))
            try:
                count = self._sock.recv_into(buffer)
            except (socket.timeout, InterruptedError):
                continue
            self._deadline.remaining()
            return count

    def close(self):
        try:
            if not self.closed:
                self._holder.close()
        finally:
            super().close()
