#!/usr/bin/env python3
"""
VoxelCraft Relay + Authority Server
Run this on any machine with a public IP:
    python3 voxelcraft_server.py [port]   (default port 25565)

Clients connect via TCP. Protocol is newline-delimited JSON.
The server is authoritative for block changes and broadcasts
all player positions to all connected clients.

Hosting options (all free tier):
  - Oracle Cloud Free Tier  (Always Free VM)
  - Google Cloud e2-micro
  - Any Linux VPS with open TCP port
"""
import socket, threading, json, time, sys, random, logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("VoxelCraft-Server")

PORT        = int(sys.argv[1]) if len(sys.argv) > 1 else 25565
TICK_RATE   = 20          # position broadcasts per second
MAX_PLAYERS = 16
TIMEOUT     = 30.0        # drop client after this many seconds silence

# ── Player colours (assigned on join) ──────────────────────────────────
PLAYER_COLORS = [
    [1.0, 0.3, 0.3],   # red
    [0.3, 0.8, 0.3],   # green
    [0.3, 0.5, 1.0],   # blue
    [1.0, 0.8, 0.2],   # yellow
    [0.8, 0.3, 1.0],   # purple
    [0.2, 0.9, 0.9],   # cyan
    [1.0, 0.5, 0.1],   # orange
    [1.0, 0.4, 0.8],   # pink
]

class Client:
    _id_counter = 0

    def __init__(self, conn, addr):
        Client._id_counter += 1
        self.pid      = Client._id_counter
        self.conn     = conn
        self.addr     = addr
        self.name     = f"Player{self.pid}"
        self.color    = PLAYER_COLORS[(self.pid - 1) % len(PLAYER_COLORS)]
        self.x = self.y = self.z = 0.0
        self.yaw = self.pitch = 0.0
        self.last_seen = time.time()
        self._buf  = ""
        self._lock = threading.Lock()

    def send(self, msg: dict):
        try:
            data = json.dumps(msg) + "\n"
            with self._lock:
                self.conn.sendall(data.encode())
        except Exception:
            pass

    def recv_lines(self):
        """Non-blocking: return all complete lines received so far."""
        try:
            self.conn.setblocking(False)
            chunk = self.conn.recv(4096).decode(errors="replace")
            self._buf += chunk
        except BlockingIOError:
            pass
        except Exception:
            return None   # connection dead
        lines = self._buf.split("\n")
        self._buf = lines[-1]
        return lines[:-1]


class Server:
    def __init__(self):
        self.clients : dict[int, Client] = {}   # pid -> Client
        self.lock    = threading.Lock()
        # Block change log: list of {x,y,z,block} dicts, for late-joining clients
        self.block_log : list = []

    def broadcast(self, msg: dict, exclude_pid: int = -1):
        with self.lock:
            targets = list(self.clients.values())
        for c in targets:
            if c.pid != exclude_pid:
                c.send(msg)

    def handle_client(self, client: Client):
        log.info(f"[+] {client.name} connected from {client.addr}")

        # Send welcome: assign pid, color, and replay block log
        client.send({"type": "welcome", "pid": client.pid,
                     "color": client.color, "name": client.name})
        # Send all existing block changes so client world matches
        with self.lock:
            log_copy = list(self.block_log)
        for entry in log_copy:
            client.send({"type": "block", **entry})

        # Announce to others
        self.broadcast({"type": "join", "pid": client.pid,
                        "name": client.name, "color": client.color},
                       exclude_pid=client.pid)

        with self.lock:
            self.clients[client.pid] = client

        try:
            while True:
                lines = client.recv_lines()
                if lines is None:
                    break   # disconnected

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    client.last_seen = time.time()
                    t = msg.get("type")

                    if t == "pos":
                        client.x     = msg.get("x", 0)
                        client.y     = msg.get("y", 0)
                        client.z     = msg.get("z", 0)
                        client.yaw   = msg.get("yaw", 0)
                        client.pitch = msg.get("pitch", 0)

                    elif t == "block":
                        entry = {"x": msg["x"], "y": msg["y"], "z": msg["z"],
                                 "block": msg["block"]}
                        with self.lock:
                            # Keep log compact: remove old entry for same pos
                            self.block_log = [
                                e for e in self.block_log
                                if not (e["x"]==entry["x"] and
                                        e["y"]==entry["y"] and
                                        e["z"]==entry["z"])
                            ]
                            if entry["block"] != "air":
                                self.block_log.append(entry)
                        # Broadcast to everyone including sender
                        # (sender already applied locally, but re-apply is harmless)
                        self.broadcast({"type": "block", **entry},
                                       exclude_pid=client.pid)

                    elif t == "chat":
                        text = str(msg.get("text", ""))[:200]
                        self.broadcast({"type": "chat",
                                        "pid": client.pid,
                                        "name": client.name,
                                        "text": text})

                    elif t == "ping":
                        client.send({"type": "pong"})

                # Timeout check
                if time.time() - client.last_seen > TIMEOUT:
                    log.info(f"[!] {client.name} timed out")
                    break

                time.sleep(0.005)

        except Exception as e:
            log.error(f"Client {client.name} error: {e}")
        finally:
            self._remove(client)

    def _remove(self, client: Client):
        with self.lock:
            self.clients.pop(client.pid, None)
        try:
            client.conn.close()
        except Exception:
            pass
        self.broadcast({"type": "leave", "pid": client.pid,
                        "name": client.name})
        log.info(f"[-] {client.name} disconnected  "
                 f"({len(self.clients)} online)")

    def tick_positions(self):
        """Broadcast all player positions at TICK_RATE hz."""
        interval = 1.0 / TICK_RATE
        while True:
            time.sleep(interval)
            with self.lock:
                players = [
                    {"pid": c.pid, "name": c.name, "color": c.color,
                     "x": c.x, "y": c.y, "z": c.z,
                     "yaw": c.yaw, "pitch": c.pitch}
                    for c in self.clients.values()
                ]
            if players:
                msg = {"type": "players", "players": players}
                self.broadcast(msg)

    def run(self):
        tick_t = threading.Thread(target=self.tick_positions, daemon=True)
        tick_t.start()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", PORT))
        sock.listen(MAX_PLAYERS)
        log.info(f"VoxelCraft server listening on 0.0.0.0:{PORT}")

        while True:
            try:
                conn, addr = sock.accept()
                conn.settimeout(TIMEOUT)
                client = Client(conn, addr)
                t = threading.Thread(target=self.handle_client,
                                     args=(client,), daemon=True)
                t.start()
            except Exception as e:
                log.error(f"Accept error: {e}")


if __name__ == "__main__":
    Server().run()