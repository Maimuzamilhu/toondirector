"""Progress on the app's Windows taskbar button, so it shows even when the window is hidden
behind others. Talks to Windows directly (ITaskbarList3); if that fails, it simply does nothing."""
import ctypes
from ctypes import POINTER, WINFUNCTYPE, byref, c_int, c_ulonglong, c_void_p, wintypes

NONE, NORMAL, ERROR = 0, 2, 4
_CLSID = "{56FDF344-FD6D-11d0-958A-006097C9A090}"
_IID = "{EA1AFB91-9E28-4B86-90E9-9E9F8A5EEFAF}"


class _Guid(ctypes.Structure):
    _fields_ = [("a", wintypes.DWORD), ("b", wintypes.WORD), ("c", wintypes.WORD), ("d", ctypes.c_ubyte * 8)]


def _guid(text):
    g = _Guid()
    ctypes.windll.ole32.CLSIDFromString(text, byref(g))
    return g


class Taskbar:
    def __init__(self, hwnd):
        self.hwnd, self.ptr = hwnd, c_void_p()
        self.last = None
        try:
            hr = ctypes.windll.ole32.CoCreateInstance(byref(_guid(_CLSID)), None, 1, byref(_guid(_IID)), byref(self.ptr))
            if hr == 0:
                self._method(3, WINFUNCTYPE(c_int, c_void_p))(self.ptr)  # HrInit
            else:
                self.ptr = None
        except OSError:
            self.ptr = None

    def _method(self, index, proto):
        table = ctypes.cast(self.ptr, POINTER(POINTER(c_void_p))).contents
        return proto(table[index])

    def set(self, state, percent=0):
        now = (state, int(percent))
        if not self.ptr or now == self.last:
            return
        self.last = now
        hwnd = wintypes.HWND(self.hwnd)
        self._method(10, WINFUNCTYPE(c_int, c_void_p, wintypes.HWND, c_int))(self.ptr, hwnd, state)
        if state:
            self._method(9, WINFUNCTYPE(c_int, c_void_p, wintypes.HWND, c_ulonglong, c_ulonglong))(
                self.ptr, hwnd, int(percent * 10), 1000)
