"""Dashboard port binding: a busy port must be detected (not silently shared,
the Windows SO_REUSEADDR trap) and the server must hop to the next free port."""

from http.server import BaseHTTPRequestHandler

from gpu_power_monitor.web.server import create_server


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass


def test_busy_port_hops_to_next_free():
    first, port1 = create_server("127.0.0.1", 18321, _Handler)
    try:
        second, port2 = create_server("127.0.0.1", port1, _Handler)
        try:
            assert port2 != port1, "two dashboard servers must never share a port"
            assert port2 == port1 + 1
        finally:
            second.server_close()
    finally:
        first.server_close()
