import base64
import contextlib
import fcntl
import os
import pty
import pwd
import signal
import struct
import subprocess
import termios
import threading
from dataclasses import dataclass
from typing import Any, Callable


MAX_INPUT_BYTES = 16 * 1024
SESSION_TIMEOUT_SECONDS = 30 * 60
Send = Callable[[str, dict[str, Any]], None]


@dataclass
class TerminalSession:
  id: str
  fd: int
  process: subprocess.Popen
  timeout: threading.Timer


def _size(value: Any, default: int) -> int:
  try:
    return max(1, min(500, int(value)))
  except (TypeError, ValueError):
    return default


class TerminalManager:
  def __init__(self, send: Send):
    self.send = send
    self.sessions: dict[str, TerminalSession] = {}
    self.lock = threading.Lock()

  def _send(self, peer: str, session_id: str, action: str, **payload: Any) -> None:
    self.send(peer, {"type": "event", "name": "terminal", "payload": {"action": action, "sessionId": session_id, **payload}})

  def _spawn(self, cols: Any, rows: Any) -> tuple[int, subprocess.Popen]:
    account = pwd.getpwnam("comma") if os.geteuid() == 0 else pwd.getpwuid(os.geteuid())
    shell = account.pw_shell if os.path.exists(account.pw_shell) else "/bin/bash"
    env = {**os.environ, "HOME": account.pw_dir, "LOGNAME": account.pw_name, "SHELL": shell, "TERM": "xterm-256color", "USER": account.pw_name}
    master_fd, slave_fd = pty.openpty()
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", _size(rows, 24), _size(cols, 80), 0, 0))
    try:
      process = subprocess.Popen(
        [shell, "-l"],
        cwd=account.pw_dir,
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        start_new_session=True,
        user=account.pw_uid if os.geteuid() == 0 else None,
        group=account.pw_gid if os.geteuid() == 0 else None,
      )
      return master_fd, process
    except Exception:
      os.close(master_fd)
      raise
    finally:
      os.close(slave_fd)

  def _close(self, peer: str, session: TerminalSession) -> None:
    with self.lock:
      if self.sessions.get(peer) is not session:
        return
      self.sessions.pop(peer)
    session.timeout.cancel()
    with contextlib.suppress(OSError):
      os.close(session.fd)
    if session.process.poll() is None:
      with contextlib.suppress(ProcessLookupError):
        os.killpg(session.process.pid, signal.SIGTERM)
      try:
        session.process.wait(timeout=1)
      except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
          os.killpg(session.process.pid, signal.SIGKILL)
        session.process.wait(timeout=1)
    self._send(peer, session.id, "closed", exitCode=session.process.poll())

  def _read(self, peer: str, session: TerminalSession) -> None:
    try:
      while data := os.read(session.fd, 4096):
        self._send(peer, session.id, "data", data=base64.b64encode(data).decode())
    except OSError:
      pass  # PTYs raise EIO when the child exits.
    finally:
      self._close(peer, session)

  def _open(self, peer: str, session_id: str, cols: Any, rows: Any) -> None:
    with self.lock:
      if peer in self.sessions:
        raise ValueError("terminal session limit reached")
      fd, process = self._spawn(cols, rows)
      session = TerminalSession(session_id, fd, process, threading.Timer(SESSION_TIMEOUT_SECONDS, lambda: self._close(peer, session)))
      session.timeout.daemon = True
      self.sessions[peer] = session
    session.timeout.start()
    self._send(peer, session_id, "opened")
    threading.Thread(target=self._read, args=(peer, session), name=f"athena_terminal_{session_id}", daemon=True).start()

  def handle(self, peer: str, payload: Any) -> None:
    if not isinstance(payload, dict):
      raise ValueError("terminal payload must be an object")
    action, session_id = payload.get("action"), payload.get("sessionId")
    if not isinstance(session_id, str) or not 1 <= len(session_id) <= 64 or not all(c.isalnum() or c in "_-" for c in session_id):
      raise ValueError("invalid terminal session ID")

    try:
      if action == "open":
        return self._open(peer, session_id, payload.get("cols"), payload.get("rows"))
      with self.lock:
        session = self.sessions.get(peer)
      if session is None or session.id != session_id:
        raise ValueError("terminal session not found")
      if action == "input":
        encoded = payload.get("data")
        if not isinstance(encoded, str) or len(encoded) > MAX_INPUT_BYTES * 2:
          raise ValueError("invalid terminal input")
        data = base64.b64decode(encoded, validate=True)
        if len(data) > MAX_INPUT_BYTES:
          raise ValueError("terminal input is too large")
        os.write(session.fd, data)
      elif action == "resize":
        fcntl.ioctl(session.fd, termios.TIOCSWINSZ, struct.pack("HHHH", _size(payload.get("rows"), 24), _size(payload.get("cols"), 80), 0, 0))
      elif action == "close":
        self._close(peer, session)
      else:
        raise ValueError("unknown terminal action")
    except Exception as error:
      self._send(peer, session_id, "error", error=str(error))
