"""Local Unix-socket embedding daemon and client."""

from __future__ import annotations

import argparse
import base64
import errno
import fcntl
import json
import os
import pathlib
import signal
import socket
import stat
import sys
import threading
import time

from .embedding import BUCKETS, MAX_LEN, Embedder
from .config import SOCKET_PATH, MODEL_DIR, DEFAULT_DEVICE

DIM = 512
IDLE_EXIT_DEFAULT = 1800.0


def default_socket() -> str:
    return str(SOCKET_PATH)


def parse_devices(spec: str) -> tuple[str, ...]:
    """'MULTI:GPU,CPU' 这类复合设备名拆成单个设备名，用于属性查询。"""
    return tuple(d.strip().upper() for d in spec.split(",") if d.strip())


def _encode_reply(vectors, t0: float) -> dict:
    import numpy as np

    v = np.ascontiguousarray(vectors, dtype=np.float32)
    return {
        "ok": True,
        "ms": round((time.perf_counter() - t0) * 1000, 2),
        "n": int(v.shape[0]),
        "dim": int(v.shape[1]),
        "dtype": "float32",
        "vectors": base64.b64encode(v.tobytes()).decode("ascii"),
    }


def _decode_vectors(reply: dict):
    import numpy as np

    v = np.frombuffer(base64.b64decode(reply["vectors"]), dtype=np.float32)
    if reply.get("dim") != DIM or reply.get("dtype") != "float32":
        raise ValueError("Embedding service returned an incompatible vector format")
    vectors = v.reshape(reply["n"], reply["dim"])
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding service returned non-finite vectors")
    return vectors


def run_server(sock_path: str, device: str, idle_exit: float, batch: int) -> None:
    path = pathlib.Path(sock_path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.with_name(path.name + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Embedding service already running: {path}") from None
        if path.exists() or path.is_symlink():
            st = path.lstat()
            if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
                raise RuntimeError(f"Refusing to replace a non-owned socket: {path}")
            with socket.socket(socket.AF_UNIX) as probe:
                probe.settimeout(1)
                try:
                    probe.connect(str(path))
                except OSError as error:
                    if error.errno not in (errno.ECONNREFUSED, errno.ENOENT):
                        raise
                else:
                    raise RuntimeError(f"Socket is in use: {path}")
            path.unlink(missing_ok=True)
        return _run_server(sock_path, device, idle_exit, batch)


def _run_server(sock_path: str, device: str, idle_exit: float, batch: int) -> None:
    t0 = time.perf_counter()
    emb = Embedder(device=device, batch=batch, verbose=False)
    print(
        f"[embd] 嵌入引擎就绪 device={device} batch={batch} "
        f"首桶编译 {time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    # 预热：把查询桶（16）和常用桶都编译掉，否则首次真实请求还要等
    t0 = time.perf_counter()
    for seq in (16, 32):
        try:
            emb._get_compiled(seq)
        except Exception as e:
            print(f"[embd] 预热桶 {seq} 失败（不影响使用）: {e}", flush=True)
    try:
        emb.encode_query("预热")
    except Exception as e:
        print(f"[embd] 预热失败（不影响使用）: {e}", flush=True)
    print(f"[embd] 预热完成 {time.perf_counter() - t0:.2f}s", flush=True)

    pathlib.Path(sock_path).parent.mkdir(parents=True, exist_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o600)
    srv.listen(8)
    srv.settimeout(idle_exit if idle_exit > 0 else None)
    print(f"[embd] 监听 {sock_path}（空闲 {idle_exit:.0f}s 退出，0=常驻）", flush=True)

    stop = threading.Event()

    def _on_signal(signum, _frame):
        print(f"[embd] 收到信号 {signum}，退出", flush=True)
        stop.set()
        try:
            srv.close()
        except OSError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    try:
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                print("[embd] 空闲超时，退出", flush=True)
                break
            except OSError:
                break
            with conn:
                # 设个超时：客户端连上却不发数据时不能把服务端拖死
                conn.settimeout(30.0)
                f = conn.makefile("rwb")
                try:
                    line = f.readline(8 * 1024 * 1024 + 1)
                    if not line:
                        continue
                    try:
                        if len(line) > 8 * 1024 * 1024:
                            raise ValueError("request exceeds 8 MiB")
                        req = json.loads(line)
                        if not isinstance(req, dict):
                            raise ValueError("request must be an object")
                    except Exception as e:
                        payload = {"ok": False, "error": f"bad request: {e}"}
                        req = None
                    if req is not None:
                        cmd = req.get("cmd", "embed")
                        try:
                            if cmd == "ping":
                                payload = {"ok": True}
                            elif cmd == "whoami":
                                payload = {
                                    "ok": True,
                                    "device": device,
                                    "devices": list(parse_devices(device)),
                                    "buckets": list(BUCKETS),
                                    "max_len": MAX_LEN,
                                    "dim": DIM,
                                    "pid": os.getpid(),
                                    "model_dir": str(emb.model_dir.resolve()),
                                }
                            elif cmd == "shutdown":
                                payload = {"ok": True}
                                stop.set()
                            elif cmd == "embed":
                                texts = req.get("texts") or []
                                if (
                                    not isinstance(texts, list)
                                    or not 1 <= len(texts) <= 256
                                    or not all(isinstance(t, str) for t in texts)
                                ):
                                    raise ValueError("texts must contain 1..256 strings")
                                is_query = bool(req.get("is_query", False))
                                t1 = time.perf_counter()
                                vecs = emb.encode(
                                    texts,
                                    is_query=is_query,
                                    normalize=bool(req.get("normalize", True)),
                                )
                                payload = _encode_reply(vecs, t1)
                            else:
                                raise ValueError(f"未知 cmd: {cmd}")
                        except Exception as e:
                            payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    # 统一在这里回包并 flush：早退分支也必须回包，否则客户端会
                    # 一直等到超时（曾因 continue 跳过回包踩过这个坑）。
                    f.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
                    f.flush()
                except (OSError, ValueError):
                    pass  # A disconnected/slow client must not stop the daemon.
                finally:
                    f.close()
    finally:
        try:
            srv.close()
        except OSError:
            pass
        try:
            os.unlink(sock_path)
        except OSError:
            pass
        print("[embd] 已退出", flush=True)


# ── 客户端 ────────────────────────────────────────────────────
class EmbClient:
    """常驻嵌入服务的客户端。

    connect() 只做一次探测；调用方决定失败时是回退直连还是直接报错。
    """

    def __init__(self, sock_path: str | None = None, timeout: float = 120.0):
        self.sock_path = sock_path or default_socket()
        self.timeout = timeout
        self.info: dict | None = None

    def _call(self, req: dict) -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        line = b""
        # 顺序很重要：`with s, s.makefile(...)` 退出时 s 会先关，缓冲区里的
        # 请求就永远发不出去（客户端等回包、服务端等请求，双双卡到超时）。
        # 所以显式 flush，不依赖上下文退出顺序。
        with s:
            s.connect(self.sock_path)
            f = s.makefile("rwb")
            try:
                f.write((json.dumps(req, ensure_ascii=False) + "\n").encode())
                f.flush()
                line = f.readline()
            finally:
                f.close()
        if not line:
            raise ConnectionError("嵌入服务未返回数据")
        return json.loads(line)

    def connect(self) -> dict:
        """确认服务在线并取回设备信息；不可用则抛异常。"""
        info = self._call({"cmd": "whoami"})
        if not info.get("ok"):
            raise ConnectionError(info.get("error", "whoami 失败"))
        self.info = info
        return info

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(self.info.get("devices", ())) if self.info else ()

    def supports(self, device: str) -> bool:
        """客户端请求的设备是否就是这个常驻实例实际用的设备。"""
        return device.strip().upper() in self.devices and self.info.get("model_dir") == str(
            MODEL_DIR.resolve()
        )

    def encode(self, texts, is_query: bool = False):
        """返回 (n, 512) float32 归一化向量。"""
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            import numpy as np

            return np.zeros((0, DIM), dtype=np.float32)
        if len(texts) > 256:
            import numpy as np

            return np.vstack(
                [
                    self.encode(texts[i : i + 256], is_query=is_query)
                    for i in range(0, len(texts), 256)
                ]
            )
        r = self._call({"cmd": "embed", "texts": texts, "is_query": is_query})
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "embed 失败"))
        if r.get("n") != len(texts):
            raise ValueError("Embedding service returned the wrong number of vectors")
        return _decode_vectors(r)

    def encode_query(self, text, **kw):
        return self.encode([text], is_query=True)


class RemoteEmbedder:
    """适配器：对外暴露和 Embedder 一样的 encode / encode_query 接口。"""

    def __init__(self, client: EmbClient, device: str, batch: int):
        self.client = client
        self.device = device
        self.batch = batch

    def encode(self, texts, is_query=False, normalize=True):
        return self.client.encode(texts, is_query=is_query)

    def encode_query(self, text, **kw):
        return self.client.encode([text], is_query=True)


def resolve_embedder(device: str, batch: int, verbose: bool = False):
    """优先用常驻服务；服务不在线、或它跑的设备与请求的设备不一致时回退直连。

    为什么要比设备：常驻实例的设备是启动时固定的。若它跑 GPU 而调用方
    要 NPU，静默用 GPU 会给出与预期不符的耗时/行为，所以宁可回退。
    """
    client = EmbClient()
    try:
        info = client.connect()
    except Exception as e:
        if verbose:
            print(
                f"[embd] 常驻服务不可用（{type(e).__name__}: {e}），回退到进程内 {device}",
                flush=True,
            )
        return Embedder(device=device, batch=batch, verbose=verbose)
    if not client.supports(device):
        if verbose:
            print(
                f"[embd] 常驻服务跑在 {info.get('device')}，"
                f"与请求的 {device} 不一致，回退到进程内 {device}",
                flush=True,
            )
        return Embedder(device=device, batch=batch, verbose=verbose)
    if verbose:
        print(f"[embd] 使用常驻服务 {client.sock_path}（device={info.get('device')}）", flush=True)
    return RemoteEmbedder(client, device, batch)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=default_socket())
    ap.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="CPU / GPU / NPU (default from RECALL_DEVICE, otherwise CPU)",
    )
    ap.add_argument(
        "--idle-exit",
        type=float,
        default=IDLE_EXIT_DEFAULT,
        help="空闲多少秒后退出（0 = 常驻不退）",
    )
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args(argv)
    run_server(a.socket, a.device, a.idle_exit, a.batch)


if __name__ == "__main__":
    main()
