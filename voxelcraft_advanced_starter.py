#!/usr/bin/env python3
"""
VoxelCraft — optimized single-file Minecraft-like voxel game
Dependencies: pip install glfw PyOpenGL PyOpenGL_accelerate numpy

Performance improvements over original:
  1. VBOs (Vertex Buffer Objects) replace display lists — GPU-native draw calls
  2. numpy mesh building — chunk geometry built in vectorized C, not Python loops
  3. Frustum culling — off-screen chunks never sent to GPU
  4. Packed vertex format (interleaved XYZ + RGB) — better GPU cache use
  5. Chunk AABB culling — 6-plane frustum test per chunk, O(1) per chunk
  6. Throttled chunk gen/rebuild — no stutter spikes

Controls:
  WASD          move
  Space         jump / fly up
  Left Shift    fly down
  Left Click    break block
  Right Click   place block
  Scroll Wheel  cycle hotbar
  1-9           select hotbar slot
  F             toggle fly mode
  Tab           cycle place-block type
  I             print debug info to console
  Escape        release mouse / quit
"""
import math, random, sys, ctypes, json, os, threading, queue, socket, time
from typing import Optional
from datetime import datetime

import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
WIN_W, WIN_H  = 1280, 720
CHUNK_SIZE    = 16
CHUNK_HEIGHT  = 128
RENDER_DIST   = 5
GRAVITY       = 0.016
JUMP_VEL      = 0.30
WALK_SPEED    = 0.10
FLY_SPEED     = 0.28
MOUSE_SENS    = 0.13
REACH         = 6.0
HOTBAR_SLOTS  = 9
PLAYER_HW     = 0.30
PLAYER_HEIGHT = 1.80
EYE_HEIGHT    = 1.62

TOP, BOTTOM, NORTH, SOUTH, WEST, EAST = 0, 1, 2, 3, 4, 5

FACE_SHADE = np.array([1.00, 0.50, 0.72, 0.72, 0.82, 0.82], dtype=np.float32)

# (dx, dy, dz) neighbour offsets per face — used in numpy mesh build
FACE_NEIGH = np.array([
    ( 0, 1, 0),  # TOP
    ( 0,-1, 0),  # BOTTOM
    ( 0, 0,-1),  # NORTH
    ( 0, 0, 1),  # SOUTH
    (-1, 0, 0),  # WEST
    ( 1, 0, 0),  # EAST
], dtype=np.int32)

# 4 vertices × 3 coords, per face — shape (6,4,3)
FACE_VERTS = np.array([
    [(0,1,0),(1,1,0),(1,1,1),(0,1,1)],   # TOP
    [(0,0,1),(1,0,1),(1,0,0),(0,0,0)],   # BOTTOM
    [(1,1,0),(1,0,0),(0,0,0),(0,1,0)],   # NORTH
    [(0,1,1),(0,0,1),(1,0,1),(1,1,1)],   # SOUTH
    [(0,1,0),(0,0,0),(0,0,1),(0,1,1)],   # WEST
    [(1,1,1),(1,0,1),(1,0,0),(1,1,0)],   # EAST
], dtype=np.float32)

# ─────────────────────────────────────────────────────────
# BLOCK REGISTRY
# ─────────────────────────────────────────────────────────
# id 0 = air; blocks stored as uint8 ids for numpy arrays
_BLOCK_NAMES = [
    "air","bedrock","stone","cobblestone","dirt","grass","sand","gravel",
    "log","planks","leaves","glass","water","coal_ore","iron_ore","gold_ore",
    "diamond_ore","snow","bricks","obsidian","glowstone","ice","pumpkin",
    "cactus","mossy_stone","shrub",
]
BLOCK_ID   = {n: i for i, n in enumerate(_BLOCK_NAMES)}
NUM_BLOCKS = len(_BLOCK_NAMES)

# per-block properties as numpy arrays (indexed by block id)
# solid[id], transparent[id]
_solid_list = [
    False,True,True,True,True,True,True,True,
    True,True,True,True,False,True,True,True,
    True,True,True,True,True,True,True,
    True,True,False,
]
_transp_list = [
    True,False,False,False,False,False,False,False,
    False,False,True,True,True,False,False,False,
    False,False,False,False,False,True,False,
    False,False,True,
]
SOLID       = np.array(_solid_list,  dtype=np.uint8)
TRANSPARENT = np.array(_transp_list, dtype=np.uint8)

# Colors: shape (NUM_BLOCKS, 3_faces=side/top/bottom, 3_rgb) float32
_COLOR_DATA = {
    "air":         (None,         None,         None),
    "bedrock":     ((.12,.12,.12),(.12,.12,.12),(.12,.12,.12)),
    "stone":       ((.50,.50,.50),(.50,.50,.50),(.50,.50,.50)),
    "cobblestone": ((.47,.47,.47),(.47,.47,.47),(.47,.47,.47)),
    "dirt":        ((.55,.36,.18),(.55,.36,.18),(.55,.36,.18)),
    "grass":       ((.55,.36,.18),(.30,.72,.18),(.55,.36,.18)),
    "sand":        ((.90,.85,.60),(.90,.85,.60),(.90,.85,.60)),
    "gravel":      ((.52,.50,.47),(.52,.50,.47),(.52,.50,.47)),
    "log":         ((.52,.38,.16),(.38,.28,.10),(.38,.28,.10)),
    "planks":      ((.72,.58,.30),(.72,.58,.30),(.72,.58,.30)),
    "leaves":      ((.18,.50,.12),(.20,.55,.14),(.18,.50,.12)),
    "glass":       ((.70,.90,1.00),(.70,.90,1.00),(.70,.90,1.00)),
    "water":       ((.20,.40,.90),(.20,.40,.90),(.20,.40,.90)),
    "coal_ore":    ((.38,.38,.38),(.38,.38,.38),(.38,.38,.38)),
    "iron_ore":    ((.57,.52,.47),(.57,.52,.47),(.57,.52,.47)),
    "gold_ore":    ((.85,.78,.35),(.85,.78,.35),(.85,.78,.35)),
    "diamond_ore": ((.38,.62,.78),(.38,.62,.78),(.38,.62,.78)),
    "snow":        ((.94,.94,.96),(.94,.94,.96),(.94,.94,.96)),
    "bricks":      ((.62,.30,.23),(.62,.30,.23),(.62,.30,.23)),
    "obsidian":    ((.12,.08,.18),(.12,.08,.18),(.12,.08,.18)),
    "glowstone":   ((.90,.80,.45),(.90,.80,.45),(.90,.80,.45)),
    "ice":         ((.70,.80,.95),(.70,.80,.95),(.70,.80,.95)),
    "pumpkin":     ((.85,.55,.20),(.60,.65,.20),(.85,.55,.20)),
    "cactus":      ((.18,.52,.18),(.20,.55,.18),(.18,.52,.18)),
    "mossy_stone": ((.38,.50,.35),(.38,.50,.35),(.38,.50,.35)),
    "shrub":       ((.25,.55,.15),(.25,.55,.15),(.25,.55,.15)),
}

# Build (NUM_BLOCKS, 6_faces, 3_rgb) float32 color table
# face mapping: 0=TOP→top, 1=BOTTOM→bottom, 2-5=sides→side
BLOCK_COLORS = np.zeros((NUM_BLOCKS, 6, 3), dtype=np.float32)
for _name, _cd in _COLOR_DATA.items():
    _bid = BLOCK_ID[_name]
    if _cd[0] is None:
        continue
    _side, _top, _bot = _cd
    for _fi in range(6):
        _c = _top if _fi == 0 else (_bot if _fi == 1 else _side)
        BLOCK_COLORS[_bid, _fi] = [_c[0]*FACE_SHADE[_fi],
                                    _c[1]*FACE_SHADE[_fi],
                                    _c[2]*FACE_SHADE[_fi]]

# Top-face color for HUD display
BLOCK_TINT = np.array([
    BLOCK_COLORS[i, 0] if BLOCK_COLORS[i, 0].any() else [0.8, 0.8, 0.8]
    for i in range(NUM_BLOCKS)
], dtype=np.float32)

WATER_ID = BLOCK_ID["water"]
AIR_ID   = BLOCK_ID["air"]

HOTBAR_DEFAULT = ["grass","dirt","stone","planks","log",
                  "glass","sand","bricks","cobblestone"]

# ─────────────────────────────────────────────────────────
# PERLIN NOISE
# ─────────────────────────────────────────────────────────
class Perlin:
    def __init__(self, seed):
        p = list(range(256))
        random.Random(seed).shuffle(p)
        self.p = p * 2

    def _fade(self, t): return t*t*t*(t*(t*6-15)+10)
    def _lerp(self, a, b, t): return a + t*(b-a)
    def _grad(self, h, x, y):
        return (x+y, -x+y, x-y, -x-y)[h & 3]

    def noise(self, x, y):
        p = self.p
        xi, yi = int(x) & 255, int(y) & 255
        xf, yf = x - int(x), y - int(y)
        u, v   = self._fade(xf), self._fade(yf)
        aa=p[p[xi]+yi]; ab=p[p[xi]+yi+1]
        ba=p[p[xi+1]+yi]; bb=p[p[xi+1]+yi+1]
        x1=self._lerp(self._grad(aa,xf,yf),   self._grad(ba,xf-1,yf),   u)
        x2=self._lerp(self._grad(ab,xf,yf-1), self._grad(bb,xf-1,yf-1), u)
        return self._lerp(x1, x2, v)

    def fbm(self, x, y, oct=6, lac=2.0, gain=0.5):
        v, amp, freq = 0.0, 1.0, 1.0
        for _ in range(oct):
            v += self.noise(x*freq, y*freq) * amp
            amp *= gain; freq *= lac
        return v

# ─────────────────────────────────────────────────────────
# CHUNK  (stores blocks as uint8 numpy array)
# ─────────────────────────────────────────────────────────
class Chunk:
    def __init__(self, cx, cz):
        self.cx, self.cz = cx, cz
        # shape: (CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE) — x,y,z
        self.blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE),
                               dtype=np.uint8)
        self.dirty  = True
        self.vbo_solid: Optional[int] = None
        self.vbo_water: Optional[int] = None
        self.solid_count = 0   # number of vertices in solid VBO
        self.water_count = 0

    def get(self, x, y, z) -> int:
        if 0 <= x < CHUNK_SIZE and 0 <= y < CHUNK_HEIGHT and 0 <= z < CHUNK_SIZE:
            return int(self.blocks[x, y, z])
        return AIR_ID

    def set(self, x, y, z, name_or_id):
        bid = name_or_id if isinstance(name_or_id, int) else BLOCK_ID.get(name_or_id, AIR_ID)
        if 0 <= x < CHUNK_SIZE and 0 <= y < CHUNK_HEIGHT and 0 <= z < CHUNK_SIZE:
            self.blocks[x, y, z] = bid
            self.dirty = True

    def get_name(self, x, y, z) -> str:
        return _BLOCK_NAMES[self.get(x, y, z)]

# ─────────────────────────────────────────────────────────
# WORLD
# ─────────────────────────────────────────────────────────
class World:
    SEA = 9

    def __init__(self, seed=None, world_idx=None):
        self.seed       = seed if seed is not None else random.randint(0, 9_999_999)
        self._world_idx = world_idx   # used in _generate to load saved blocks
        self.chunks : dict = {}
        self.pn     = Perlin(self.seed)
        self.pn2    = Perlin(self.seed + 1)

    def get_chunk(self, cx, cz, gen=True) -> Optional[Chunk]:
        k = (cx, cz)
        if k not in self.chunks:
            if not gen: return None
            self.chunks[k] = self._generate(cx, cz)
        return self.chunks[k]

    def chunk_of(self, wx, wz):
        return math.floor(wx / CHUNK_SIZE), math.floor(wz / CHUNK_SIZE)

    def surface_h(self, wx, wz) -> int:
        n = self.pn.fbm(wx * 0.004, wz * 0.004, oct=5)
        n = (n + 1.0) * 0.5
        return int(4 + n * 58)

    def biome(self, wx, wz) -> str:
        t = self.pn2.fbm(wx * 0.002, wz * 0.002, oct=3)
        if t < -0.4: return "snow"
        if t >  0.4: return "desert"
        if t >  0.1: return "forest"
        return "plains"

    def _generate(self, cx, cz) -> Chunk:
        c   = Chunk(cx, cz)
        # If saved block data exists for this chunk, restore it and skip terrain gen
        if self._world_idx is not None:
            saved = load_chunk(self._world_idx, cx, cz)
            if saved is not None:
                c.blocks = saved
                c.dirty  = True
                return c
        rng = random.Random(self.seed ^ (cx * 100_003 + cz * 999_983))

        heights = {}
        biomes  = {}
        for lx in range(CHUNK_SIZE):
            for lz in range(CHUNK_SIZE):
                wx = cx * CHUNK_SIZE + lx
                wz = cz * CHUNK_SIZE + lz
                h  = self.surface_h(wx, wz)
                bm = self.biome(wx, wz)
                heights[(lx,lz)] = h
                biomes [(lx,lz)] = bm

                c.set(lx, 0, lz, "bedrock")

                for y in range(1, max(1, h - 3)):
                    r = rng.random()
                    if   y < 16 and r < 0.008: blk = "diamond_ore"
                    elif y < 32 and r < 0.015: blk = "gold_ore"
                    elif y < 48 and r < 0.030: blk = "iron_ore"
                    elif            r < 0.050: blk = "coal_ore"
                    else:                      blk = "stone"
                    c.set(lx, y, lz, blk)

                for y in range(max(1, h-3), h-1):
                    c.set(lx, y, lz, "dirt" if bm != "desert" else "sand")

                if h > 0:
                    if h <= self.SEA + 1:
                        c.set(lx, h-1, lz, "sand")
                    elif bm == "snow":
                        c.set(lx, h-1, lz, "snow")
                        if h > 4: c.set(lx, h-2, lz, "dirt")
                    elif bm == "desert":
                        c.set(lx, h-1, lz, "sand")
                        if h > 4: c.set(lx, h-2, lz, "sand")
                    else:
                        c.set(lx, h-1, lz, "grass")

                for y in range(h, self.SEA + 1):
                    c.set(lx, y, lz, "water")

        # trees / features
        for lx in range(2, CHUNK_SIZE - 2):
            for lz in range(2, CHUNK_SIZE - 2):
                h  = heights[(lx,lz)]
                bm = biomes [(lx,lz)]
                if h <= self.SEA + 1: continue

                if bm == "desert" and rng.random() < 0.012:
                    for y in range(h-1, h-1+rng.randint(2,4)):
                        c.set(lx, y, lz, "cactus")

                elif bm in ("plains","forest"):
                    chance = 0.025 if bm == "plains" else 0.06
                    if rng.random() < chance:
                        trunk = rng.randint(4, 6)
                        for y in range(h-1, h-1+trunk):
                            c.set(lx, y, lz, "log")
                        top = h - 1 + trunk
                        for dy in range(-2, 2):
                            r = 2 if dy < 0 else 1
                            for dx in range(-r, r+1):
                                for dz in range(-r, r+1):
                                    if abs(dx)==r and abs(dz)==r: continue
                                    nx, nz = lx+dx, lz+dz
                                    if 0 <= nx < CHUNK_SIZE and 0 <= nz < CHUNK_SIZE:
                                        py = top + dy
                                        if c.get(nx, py, nz) == AIR_ID:
                                            c.set(nx, py, nz, "leaves")
                    elif rng.random() < 0.04:
                        c.set(lx, h-1, lz, "shrub")

        return c

    def get_block_id(self, wx, wy, wz) -> int:
        if wy < 0: return BLOCK_ID["bedrock"]
        if wy >= CHUNK_HEIGHT: return AIR_ID
        cx, cz = self.chunk_of(wx, wz)
        lx = int(wx - cx*CHUNK_SIZE)
        lz = int(wz - cz*CHUNK_SIZE)
        ch = self.get_chunk(cx, cz, gen=False)
        return ch.get(lx, wy, lz) if ch else AIR_ID

    def get_block(self, wx, wy, wz) -> str:
        return _BLOCK_NAMES[self.get_block_id(wx, wy, wz)]

    def set_block(self, wx, wy, wz, name):
        if wy < 0 or wy >= CHUNK_HEIGHT: return
        cx, cz = self.chunk_of(wx, wz)
        lx = int(wx - cx*CHUNK_SIZE)
        lz = int(wz - cz*CHUNK_SIZE)
        ch = self.get_chunk(cx, cz, gen=True)
        ch.set(lx, wy, lz, name)
        # dirty adjacent chunks on borders
        if lx == 0:
            n = self.get_chunk(cx-1, cz, False)
            if n: n.dirty = True
        if lx == CHUNK_SIZE-1:
            n = self.get_chunk(cx+1, cz, False)
            if n: n.dirty = True
        if lz == 0:
            n = self.get_chunk(cx, cz-1, False)
            if n: n.dirty = True
        if lz == CHUNK_SIZE-1:
            n = self.get_chunk(cx, cz+1, False)
            if n: n.dirty = True

    def raycast(self, ox, oy, oz, dx, dy, dz, max_d=REACH):
        def init_axis(o, b, s, d):
            if abs(d) < 1e-9: return float('inf'), float('inf')
            return ((b + (1 if s > 0 else 0)) - o) / d, abs(1.0 / d)

        bx, by, bz = math.floor(ox), math.floor(oy), math.floor(oz)
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        sz = 1 if dz > 0 else -1
        tmx, dtx = init_axis(ox, bx, sx, dx)
        tmy, dty = init_axis(oy, by, sy, dy)
        tmz, dtz = init_axis(oz, bz, sz, dz)
        prev = (bx, by, bz)

        for _ in range(int(max_d / 0.01) + 1):
            if min(tmx, tmy, tmz) > max_d: break
            bid = self.get_block_id(bx, by, bz)
            if SOLID[bid]:
                return (bx,by,bz), prev, _BLOCK_NAMES[bid]
            prev = (bx, by, bz)
            if tmx < tmy and tmx < tmz:
                bx += sx; tmx += dtx
            elif tmy < tmz:
                by += sy; tmy += dty
            else:
                bz += sz; tmz += dtz
        return None, None, None

    def find_surface(self, wx, wz) -> int:
        for y in range(CHUNK_HEIGHT - 1, 0, -1):
            if SOLID[self.get_block_id(int(wx), y, int(wz))]:
                return y + 1
        return 10

# ─────────────────────────────────────────────────────────
# NUMPY MESH BUILDER  ← THE BIG WIN
# ─────────────────────────────────────────────────────────
def _build_padded(chunk: Chunk, world: World) -> np.ndarray:
    """
    (CS+2, CH+2, CS+2) padded block-id volume.
    Border voxels come from neighbouring chunks so face culling is
    correct across chunk boundaries. Built once per mesh call.
    """
    CS, CH = CHUNK_SIZE, CHUNK_HEIGHT
    cx, cz = chunk.cx, chunk.cz
    padded = np.zeros((CS+2, CH+2, CS+2), dtype=np.uint8)
    padded[1:CS+1, 1:CH+1, 1:CS+1] = chunk.blocks
    nc = world.get_chunk(cx-1, cz, gen=False)
    if nc is not None: padded[0,      1:CH+1, 1:CS+1] = nc.blocks[CS-1, :, :]
    nc = world.get_chunk(cx+1, cz, gen=False)
    if nc is not None: padded[CS+1,   1:CH+1, 1:CS+1] = nc.blocks[0,    :, :]
    nc = world.get_chunk(cx, cz-1, gen=False)
    if nc is not None: padded[1:CS+1, 1:CH+1, 0     ] = nc.blocks[:,    :, CS-1]
    nc = world.get_chunk(cx, cz+1, gen=False)
    if nc is not None: padded[1:CS+1, 1:CH+1, CS+1  ] = nc.blocks[:,    :, 0]
    return padded


def build_chunk_mesh(chunk: Chunk, world: World):
    """
    Pure-numpy mesh builder — no Python loops over blocks.
    For each of the 6 face directions:
      1. Compute the neighbour-block slice in one array op
      2. Build a boolean visibility mask with numpy boolean ops
      3. argwhere the mask → list of visible block positions
      4. Broadcast FACE_VERTS offsets to get all 4 verts at once
    Returns (solid: float32 (N,6), water: float32 (M,7)).
    """
    CS, CH = CHUNK_SIZE, CHUNK_HEIGHT
    ox_world = chunk.cx * CS
    oz_world = chunk.cz * CS
    blk = chunk.blocks  # (CS, CH, CS) uint8

    # Empty chunk fast path
    if not blk.any():
        return (np.empty((0, 6), dtype=np.float32),
                np.empty((0, 7), dtype=np.float32))

    padded       = _build_padded(chunk, world)
    blk_solid    = SOLID[blk].view(bool)
    blk_is_water = (blk == WATER_ID)
    blk_opaque   = blk_solid & ~blk_is_water

    solid_parts = []
    water_parts = []

    for fi in range(6):
        dx, dy, dz = int(FACE_NEIGH[fi,0]), int(FACE_NEIGH[fi,1]), int(FACE_NEIGH[fi,2])
        nb = padded[1+dx:1+dx+CS, 1+dy:1+dy+CH, 1+dz:1+dz+CS]

        nb_transp = TRANSPARENT[nb].view(bool)
        nb_same   = (nb == blk)

        # ── Solid faces ──────────────────────────────────
        show = blk_opaque & nb_transp & ~nb_same
        idx  = np.argwhere(show)          # (N, 3)
        if len(idx):
            lx, ly, lz  = idx[:,0], idx[:,1], idx[:,2]
            block_ids   = blk[lx, ly, lz]
            colors      = BLOCK_COLORS[block_ids, fi]          # (N, 3)
            fv          = FACE_VERTS[fi]                       # (4, 3)
            wx = (lx[:,None].astype(np.float32) + ox_world + fv[None,:,0])
            wy = (ly[:,None].astype(np.float32)             + fv[None,:,1])
            wz = (lz[:,None].astype(np.float32) + oz_world  + fv[None,:,2])
            wc = np.repeat(colors[:,None,:], 4, axis=1)        # (N, 4, 3)
            verts = np.concatenate(
                [wx[:,:,None], wy[:,:,None], wz[:,:,None], wc], axis=2
            )                                                  # (N, 4, 6)
            solid_parts.append(verts.reshape(-1, 6))

        # ── Water faces ──────────────────────────────────
        show_w = (blk_is_water
                  & (nb != WATER_ID)
                  & ~(SOLID[nb].view(bool) & ~TRANSPARENT[nb].view(bool)))
        idx_w = np.argwhere(show_w)
        if len(idx_w):
            lx, ly, lz = idx_w[:,0], idx_w[:,1], idx_w[:,2]
            fv = FACE_VERTS[fi].copy()
            if fi == TOP:
                fv[:,1] += 0.1
            wx = (lx[:,None].astype(np.float32) + ox_world + fv[None,:,0])
            wy = (ly[:,None].astype(np.float32)             + fv[None,:,1])
            wz = (lz[:,None].astype(np.float32) + oz_world  + fv[None,:,2])
            wc = np.full((len(lx), 4, 4), [0.20, 0.42, 0.90, 0.60], dtype=np.float32)
            verts = np.concatenate(
                [wx[:,:,None], wy[:,:,None], wz[:,:,None], wc], axis=2
            )
            water_parts.append(verts.reshape(-1, 7))

    solid = (np.concatenate(solid_parts, axis=0)
             if solid_parts else np.empty((0, 6), dtype=np.float32))
    water = (np.concatenate(water_parts, axis=0)
             if water_parts else np.empty((0, 7), dtype=np.float32))
    return solid, water


def upload_vbo(chunk: Chunk, solid: np.ndarray, water: np.ndarray):
    """Upload mesh data to GPU VBOs."""
    # Solid VBO
    if chunk.vbo_solid is None:
        buf = glGenBuffers(1)
        chunk.vbo_solid = int(buf)
    glBindBuffer(GL_ARRAY_BUFFER, chunk.vbo_solid)
    if len(solid):
        glBufferData(GL_ARRAY_BUFFER, solid.nbytes, solid.tobytes(), GL_STATIC_DRAW)
    chunk.solid_count = len(solid)

    # Water VBO
    if chunk.vbo_water is None:
        buf = glGenBuffers(1)
        chunk.vbo_water = int(buf)
    glBindBuffer(GL_ARRAY_BUFFER, chunk.vbo_water)
    if len(water):
        glBufferData(GL_ARRAY_BUFFER, water.nbytes, water.tobytes(), GL_STATIC_DRAW)
    chunk.water_count = len(water)

    glBindBuffer(GL_ARRAY_BUFFER, 0)


def draw_vbo(vbo_id: int, count: int):
    """Draw a VBO with interleaved (xyz, rgb) format — solid geometry."""
    if count == 0: return
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
    stride = 6 * 4  # 6 floats × 4 bytes
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
    glColorPointer (3, GL_FLOAT, stride, ctypes.c_void_p(12))
    glDrawArrays(GL_QUADS, 0, count)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


def draw_vbo_rgba(vbo_id: int, count: int):
    """Draw a VBO with interleaved (xyz, rgba) format — translucent geometry."""
    if count == 0: return
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
    stride = 7 * 4  # 7 floats × 4 bytes
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
    glColorPointer (4, GL_FLOAT, stride, ctypes.c_void_p(12))  # 4-component RGBA
    glDrawArrays(GL_QUADS, 0, count)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


# ─────────────────────────────────────────────────────────
# FRUSTUM CULLING
# ─────────────────────────────────────────────────────────
class Frustum:
    """Extract 6 clip planes from the current MVP matrix."""
    def __init__(self):
        self.planes = np.zeros((6, 4), dtype=np.float64)

    def update(self):
        proj = np.array(glGetDoublev(GL_PROJECTION_MATRIX), dtype=np.float64).reshape(4,4).T
        modl = np.array(glGetDoublev(GL_MODELVIEW_MATRIX),  dtype=np.float64).reshape(4,4).T
        clip = proj @ modl  # (4,4)

        # 6 planes: right,left,top,bottom,near,far
        p = self.planes
        p[0] = clip[3] - clip[0]   # right
        p[1] = clip[3] + clip[0]   # left
        p[2] = clip[3] - clip[1]   # top
        p[3] = clip[3] + clip[1]   # bottom
        p[4] = clip[3] - clip[2]   # near
        p[5] = clip[3] + clip[2]   # far
        # normalize
        norms = np.linalg.norm(p[:, :3], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.planes /= norms

    def chunk_visible(self, cx, cz) -> bool:
        """AABB test: is this chunk's bounding box in the frustum?"""
        wx = cx * CHUNK_SIZE
        wz = cz * CHUNK_SIZE
        # chunk AABB: (wx, 0, wz) → (wx+CS, CH, wz+CS)
        CS, CH = CHUNK_SIZE, CHUNK_HEIGHT
        # 8 corners
        corners = np.array([
            [wx,    0,  wz,    1],
            [wx+CS, 0,  wz,    1],
            [wx,    0,  wz+CS, 1],
            [wx+CS, 0,  wz+CS, 1],
            [wx,    CH, wz,    1],
            [wx+CS, CH, wz,    1],
            [wx,    CH, wz+CS, 1],
            [wx+CS, CH, wz+CS, 1],
        ], dtype=np.float64)

        for plane in self.planes:
            # If all 8 corners are outside this plane, chunk is culled
            dots = corners @ plane
            if np.all(dots < 0):
                return False
        return True

# ─────────────────────────────────────────────────────────
# PLAYER
# ─────────────────────────────────────────────────────────
class Player:
    def __init__(self, x=8.5, z=8.5):
        self.x, self.z = x, z
        self.y   = 80.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.vy  = 0.0
        self.grounded = False
        self.flying   = False

    @property
    def eye(self):
        return self.x, self.y + EYE_HEIGHT, self.z

    def look_dir(self):
        yr = math.radians(self.yaw)
        pr = math.radians(self.pitch)
        return (math.sin(yr)*math.cos(pr), math.sin(pr), -math.cos(yr)*math.cos(pr))

    def forward_h(self):
        yr = math.radians(self.yaw)
        return math.sin(yr), 0.0, -math.cos(yr)

    def right_h(self):
        yr = math.radians(self.yaw)
        return math.cos(yr), 0.0, math.sin(yr)

# ─────────────────────────────────────────────────────────
# INVENTORY
# ─────────────────────────────────────────────────────────
class Inventory:
    def __init__(self):
        self.hotbar   = list(HOTBAR_DEFAULT)
        self.selected = 0

    def scroll(self, d):
        self.selected = (self.selected - int(d)) % HOTBAR_SLOTS

    def select(self, i):
        self.selected = i % HOTBAR_SLOTS

    def current(self) -> str:
        return self.hotbar[self.selected]

# ─────────────────────────────────────────────────────────
# COLLISION
# ─────────────────────────────────────────────────────────
def collides(world: World, x, y, z) -> bool:
    w = PLAYER_HW
    for bx in range(math.floor(x-w), math.ceil(x+w)):
        for by in range(math.floor(y), math.ceil(y + PLAYER_HEIGHT)):
            for bz in range(math.floor(z-w), math.ceil(z+w)):
                if SOLID[world.get_block_id(bx, by, bz)]:
                    return True
    return False

# ─────────────────────────────────────────────────────────
# PHYSICS
# ─────────────────────────────────────────────────────────
def update_physics(world: World, player: Player, keys: dict, dt: float,
                   walk_speed: float = WALK_SPEED, fly_speed: float = FLY_SPEED):
    speed = fly_speed if player.flying else walk_speed
    fx, _, fz = player.forward_h()
    rx, _, rz = player.right_h()
    mx = mz = 0.0

    if keys.get(glfw.KEY_W): mx += fx; mz += fz
    if keys.get(glfw.KEY_S): mx -= fx; mz -= fz
    if keys.get(glfw.KEY_A): mx -= rx; mz -= rz
    if keys.get(glfw.KEY_D): mx += rx; mz += rz

    mag = math.sqrt(mx*mx + mz*mz)
    if mag > 0:
        mx = mx / mag * speed
        mz = mz / mag * speed

    if player.flying:
        my = 0.0
        if keys.get(glfw.KEY_SPACE):      my =  FLY_SPEED
        if keys.get(glfw.KEY_LEFT_SHIFT): my = -FLY_SPEED
        player.vy = 0.0
        if not collides(world, player.x+mx, player.y, player.z):      player.x += mx
        if not collides(world, player.x, player.y+my, player.z):      player.y += my
        if not collides(world, player.x, player.y, player.z+mz):      player.z += mz
    else:
        if mx and not collides(world, player.x+mx, player.y, player.z): player.x += mx
        if mz and not collides(world, player.x, player.y, player.z+mz): player.z += mz
        if keys.get(glfw.KEY_SPACE) and player.grounded:
            player.vy = JUMP_VEL
            player.grounded = False
        player.vy -= GRAVITY
        player.vy  = max(player.vy, -1.2)
        ny = player.y + player.vy
        if not collides(world, player.x, ny, player.z):
            player.y = ny
            player.grounded = False
        else:
            if player.vy < 0: player.grounded = True
            player.vy = 0.0

    player.y = max(0.0, player.y)

# ─────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────
def draw_rect(x, y, w, h, r, g, b, a=1.0):
    glColor4f(r, g, b, a)
    glBegin(GL_QUADS)
    glVertex2f(x,   y);   glVertex2f(x+w, y)
    glVertex2f(x+w, y+h); glVertex2f(x,   y+h)
    glEnd()

def draw_rect_border(x, y, w, h, r, g, b, lw=2.0):
    glColor4f(r, g, b, 1.0)
    glLineWidth(lw)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x,y); glVertex2f(x+w,y); glVertex2f(x+w,y+h); glVertex2f(x,y+h)
    glEnd()

def hud_begin(w, h):
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix(); glLoadIdentity()
    glOrtho(0, w, 0, h, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix(); glLoadIdentity()

def hud_end():
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW);  glPopMatrix()
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)

def draw_crosshair(cx, cy):
    glColor4f(1,1,1,0.9); glLineWidth(2.0)
    glBegin(GL_LINES)
    glVertex2f(cx-10,cy); glVertex2f(cx+10,cy)
    glVertex2f(cx,cy-10); glVertex2f(cx,cy+10)
    glEnd()

def draw_hotbar(inv: Inventory, sw, sh):
    slot_sz, padding = 50, 6
    total_w = HOTBAR_SLOTS * (slot_sz + padding) - padding
    hx = (sw - total_w) // 2
    hy = 14
    for i, blk in enumerate(inv.hotbar):
        sx = hx + i * (slot_sz + padding)
        draw_rect(sx, hy, slot_sz, slot_sz, 0.15,0.15,0.15, 0.80)
        if blk:
            bid = BLOCK_ID.get(blk, 0)
            c = BLOCK_TINT[bid]
            inner = 8
            draw_rect(sx+inner, hy+inner, slot_sz-inner*2, slot_sz-inner*2,
                      float(c[0]), float(c[1]), float(c[2]), 1.0)
        if i == inv.selected:
            draw_rect_border(sx-2, hy-2, slot_sz+4, slot_sz+4, 1,1,1, 2.5)
        else:
            draw_rect_border(sx, hy, slot_sz, slot_sz, 0.5,0.5,0.5, 1.5)

def draw_block_highlight(bx, by, bz):
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(2.0)
    glColor4f(0,0,0,0.6)
    glBegin(GL_LINES)
    edges = [
        (0,0,0),(1,0,0),(1,0,0),(1,0,1),(1,0,1),(0,0,1),(0,0,1),(0,0,0),
        (0,1,0),(1,1,0),(1,1,0),(1,1,1),(1,1,1),(0,1,1),(0,1,1),(0,1,0),
        (0,0,0),(0,1,0),(1,0,0),(1,1,0),(1,0,1),(1,1,1),(0,0,1),(0,1,1),
    ]
    eps = 0.002
    for j in range(0, len(edges), 2):
        a, e = edges[j], edges[j+1]
        glVertex3f(bx+a[0]-eps, by+a[1]-eps, bz+a[2]-eps)
        glVertex3f(bx+e[0]+eps, by+e[1]+eps, bz+e[2]+eps)
    glEnd()
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)

# ─────────────────────────────────────────────────────────
# GAME
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# NETWORK CLIENT
# ─────────────────────────────────────────────────────────
class NetworkClient:
    """
    Async TCP client.  All socket I/O runs on a background thread.
    Main thread calls send_*() freely; incoming messages accumulate
    in self.inbox and are drained by Game.update().
    """
    def __init__(self, host: str, port: int, player_name: str):
        self.host        = host
        self.port        = port
        self.player_name = player_name
        self.pid         = None          # assigned by server on welcome
        self.color       = [1.0,1.0,1.0]
        self.connected   = False
        self.error       = ""
        self.inbox       : queue.Queue = queue.Queue()
        self._sock       = None
        self._buf        = ""
        self._lock       = threading.Lock()
        self._stop       = threading.Event()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    # ── public send helpers ──────────────────────────────
    def send_pos(self, player: "Player"):
        self._send({"type": "pos",
                    "x": round(player.x, 2), "y": round(player.y, 2),
                    "z": round(player.z, 2),
                    "yaw": round(player.yaw, 1),
                    "pitch": round(player.pitch, 1)})

    def send_block(self, x, y, z, block: str):
        self._send({"type": "block", "x": x, "y": y, "z": z, "block": block})

    def send_chat(self, text: str):
        self._send({"type": "chat", "text": text})

    def disconnect(self):
        self._stop.set()
        try:
            if self._sock: self._sock.close()
        except Exception:
            pass

    # ── internal ─────────────────────────────────────────
    def _send(self, msg: dict):
        if not self.connected:
            return
        try:
            data = (json.dumps(msg) + "\n").encode()
            with self._lock:
                self._sock.sendall(data)
        except Exception as e:
            self.error = str(e)
            self.connected = False

    def _run(self):
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=8)
            self._sock.settimeout(None)
            self.connected = True
        except Exception as e:
            self.error = f"Cannot connect to {self.host}:{self.port} — {e}"
            return

        # Send player name
        self._send({"type": "hello", "name": self.player_name})

        buf = ""
        try:
            while not self._stop.is_set():
                try:
                    chunk = self._sock.recv(4096).decode(errors="replace")
                    if not chunk:
                        break
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                            t = msg.get("type")
                            if t == "welcome":
                                self.pid   = msg["pid"]
                                self.color = msg.get("color", self.color)
                            else:
                                self.inbox.put(msg)
                        except Exception:
                            pass
                except OSError:
                    break
        except Exception as e:
            self.error = str(e)
        finally:
            self.connected = False


class Game:
    def __init__(self, seed=None, world_name='World 1', world_idx=0):
        self.world_name = world_name
        self.world_idx  = world_idx
        self.world   = World(seed=seed, world_idx=world_idx)
        self.player  = Player()
        self.inv     = Inventory()
        self.frustum = Frustum()
        self.keys    : dict = {}
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self.captured = False
        self.hit_pos  = None
        self.place_pos = None
        self.window   = None
        self.ww = WIN_W
        self.wh = WIN_H

        # FPS counter
        self._fps_time  = 0.0
        self._fps_frames = 0
        self._fps = 0.0

        # Save feedback: seconds remaining to show "Saved!" flash, -1 = none
        self._save_flash = -1.0

        # Background chunk generation
        # Worker thread generates Chunk objects; main thread inserts + marks dirty
        self._gen_queue   : queue.Queue = queue.Queue()   # (cx, cz) jobs
        self._done_queue  : queue.Queue = queue.Queue()   # completed Chunk objects
        self._in_flight   : set         = set()           # (cx,cz) currently being generated
        self._gen_thread  = threading.Thread(
            target=self._chunk_gen_worker, daemon=True)
        self._gen_thread.start()

        # Per-instance settings (can be changed in ESC menu)
        self.render_dist  = RENDER_DIST   # chunks each direction
        self.walk_speed   = WALK_SPEED
        self.fly_speed    = FLY_SPEED

        # ESC menu state
        self.esc_open      = False
        self.change_world  = False        # set True to return to world select
        # slider drag state: None | 'render_dist' | 'walk_speed' | 'fly_speed'
        self._drag_slider  = None
        self._menu_rects   = {}           # populated each frame by draw_esc_menu

        # Multiplayer
        self.net            : "NetworkClient | None" = None
        # pid -> {name, color, x,y,z, yaw, pitch}
        self.remote_players : dict = {}
        # Chat: list of (timestamp, name, text); shown as HUD overlay
        self.chat_log       : list = []
        self.chat_input     : str  = ""
        self.chat_open      : bool = False
        self._net_tick      : float = 0.0   # accumulator for pos send rate

    def init(self):
        if not glfw.init():
            sys.exit("glfw.init() failed")
        # Allow Mesa software rasterizer on machines without a GPU
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "0")
        glfw.window_hint(glfw.SAMPLES, 0)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        w = glfw.create_window(WIN_W, WIN_H, f"VoxelCraft — {self.world_name}", None, None)
        if not w:
            sys.exit("Could not create window")
        self.window = w
        glfw.make_context_current(w)
        glfw.swap_interval(0)   # uncap FPS for benchmarking (set to 1 for vsync)

        glfw.set_key_callback(w,              self._on_key)
        glfw.set_mouse_button_callback(w,     self._on_mouse_button)
        glfw.set_scroll_callback(w,           self._on_scroll)
        glfw.set_cursor_pos_callback(w,       self._on_cursor)
        glfw.set_framebuffer_size_callback(w, self._on_resize)
        glfw.set_char_callback(w, self._on_char)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glClearColor(0.50, 0.72, 1.00, 1.0)
        glShadeModel(GL_SMOOTH)
        # GL_CULL_FACE intentionally disabled: FACE_VERTS winding is not
        # consistent across all 6 faces so culling drops valid quads.
        # Hidden faces are already removed by the numpy mesh builder.

        print(f"[VoxelCraft] Seed: {self.world.seed}  Generating terrain …")

        # Restore saved player position or spawn on surface
        saved_pos = load_player_pos(self.world_idx)
        if saved_pos:
            self.player.x     = saved_pos["x"]
            self.player.y     = saved_pos["y"]
            self.player.z     = saved_pos["z"]
            self.player.yaw   = saved_pos.get("yaw",   0.0)
            self.player.pitch = saved_pos.get("pitch", 0.0)
            self.player.flying = saved_pos.get("flying", False)
            print(f"[VoxelCraft] Restored position ({self.player.x:.1f}, "
                  f"{self.player.y:.1f}, {self.player.z:.1f})")

        pcx, pcz = self.world.chunk_of(self.player.x, self.player.z)
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                self.world.get_chunk(pcx+dx, pcz+dz, gen=True)

        if not saved_pos:
            self.player.y = float(self.world.find_surface(self.player.x, self.player.z))

        print("[VoxelCraft] Ready.  Click window to capture mouse.")

    def _on_resize(self, win, w, h): self.ww, self.wh = w, h

    def _on_key(self, win, key, sc, action, mods):
        if action == glfw.PRESS:
            self.keys[key] = True
            self._handle_press(key)
        elif action == glfw.RELEASE:
            self.keys[key] = False

    def _handle_press(self, key):
        # Chat input intercepts most keys
        if self.chat_open:
            if key == glfw.KEY_ESCAPE:
                self.chat_open  = False
                self.chat_input = ""
                self._capture_mouse()
            elif key == glfw.KEY_ENTER:
                text = self.chat_input.strip()
                if text and self.net and self.net.connected:
                    self.net.send_chat(text)
                    self._add_chat("You", text)
                self.chat_input = ""
                self.chat_open  = False
                self._capture_mouse()
            elif key == glfw.KEY_BACKSPACE:
                self.chat_input = self.chat_input[:-1]
            return

        if key == glfw.KEY_ESCAPE:
            if self.esc_open:
                self.esc_open = False
                self._capture_mouse()
            elif self.captured:
                self._release_mouse()
                self.esc_open = True
            else:
                glfw.set_window_should_close(self.window, True)

        elif key == glfw.KEY_T and self.captured and not self.esc_open:
            if self.net and self.net.connected:
                self.chat_open = True
                self._release_mouse()
                return

        elif key == glfw.KEY_F:
            self.player.flying = not self.player.flying
            self.player.vy = 0.0
            print(f"Fly: {'ON' if self.player.flying else 'OFF'}")

        elif key == glfw.KEY_I:
            p = self.player
            print(f"Pos:({p.x:.1f},{p.y:.1f},{p.z:.1f})  FPS:{self._fps:.0f}  "
                  f"Chunks:{len(self.world.chunks)}")

        elif key == glfw.KEY_TAB:
            all_blk = [b for b in _BLOCK_NAMES if b not in ("air","water","shrub","bedrock")]
            cur = self.inv.current()
            idx = all_blk.index(cur) if cur in all_blk else 0
            nxt = all_blk[(idx+1) % len(all_blk)]
            self.inv.hotbar[self.inv.selected] = nxt
            print(f"Block: {nxt}")

        for i, k in enumerate([glfw.KEY_1,glfw.KEY_2,glfw.KEY_3,glfw.KEY_4,glfw.KEY_5,
                                glfw.KEY_6,glfw.KEY_7,glfw.KEY_8,glfw.KEY_9]):
            if key == k: self.inv.select(i)

    def _on_mouse_button(self, win, button, action, mods):
        if self.esc_open:
            if button == glfw.MOUSE_BUTTON_LEFT:
                if action == glfw.PRESS:
                    self._menu_click_down()
                elif action == glfw.RELEASE:
                    self._drag_slider = None
            return

        if action != glfw.PRESS: return
        if not self.captured:
            self._capture_mouse(); return

        if button == glfw.MOUSE_BUTTON_LEFT and self.hit_pos:
            bx,by,bz = self.hit_pos
            if self.world.get_block(bx,by,bz) != "bedrock":
                self.world.set_block(bx,by,bz,"air")
                if self.net and self.net.connected:
                    self.net.send_block(bx, by, bz, "air")

        elif button == glfw.MOUSE_BUTTON_RIGHT and self.place_pos:
            px,py,pz = self.place_pos
            blk = self.inv.current()
            if not _player_overlaps(self.player, px,py,pz):
                self.world.set_block(px,py,pz, blk)
                if self.net and self.net.connected:
                    self.net.send_block(px, py, pz, blk)

    def _on_scroll(self, win, xoff, yoff): self.inv.scroll(yoff)

    def _on_cursor(self, win, xpos, ypos):
        if self.esc_open:
            self.mouse_x, self.mouse_y = xpos, ypos
            if self._drag_slider:
                self._menu_drag(xpos)
            return
        if not self.captured:
            self.mouse_x, self.mouse_y = xpos, ypos; return
        dx = xpos - self.mouse_x
        dy = ypos - self.mouse_y
        self.mouse_x, self.mouse_y = xpos, ypos
        self.player.yaw   = (self.player.yaw + dx*MOUSE_SENS) % 360
        self.player.pitch = max(-89.0, min(89.0, self.player.pitch - dy*MOUSE_SENS))

    def _on_char(self, win, codepoint):
        """Receive printable character input for chat."""
        if self.chat_open and len(self.chat_input) < 120:
            self.chat_input += chr(codepoint)

    def _capture_mouse(self):
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        self.mouse_x, self.mouse_y = glfw.get_cursor_pos(self.window)
        self.captured = True

    def _release_mouse(self):
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        self.captured = False

    def _chunk_gen_worker(self):
        """
        Background thread: generate chunk terrain AND build its mesh.
        Only the tiny GPU upload (glBufferData) stays on the main thread.
        This keeps the main thread completely stutter-free.
        """
        while True:
            try:
                cx, cz = self._gen_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                ch = self.world._generate(cx, cz)
                # Build mesh CPU-side in background — pure numpy, no GL calls
                solid, water = build_chunk_mesh(ch, self.world)
                self._done_queue.put((cx, cz, ch, solid, water))
            except Exception as e:
                print(f"[ChunkGen] Error at ({cx},{cz}): {e}")
            finally:
                self._gen_queue.task_done()

    def _update_chunks(self):
        px, pz = self.player.x, self.player.z
        pcx, pcz = self.world.chunk_of(px, pz)
        rd = self.render_dist

        # ── Pull completed chunks from background thread (non-blocking) ──
        # Mesh is pre-built; just upload to GPU and register chunk.
        uploaded = 0
        while not self._done_queue.empty() and uploaded < 2:
            try:
                cx, cz, ch, solid, water = self._done_queue.get_nowait()
                self._in_flight.discard((cx, cz))
                upload_vbo(ch, solid, water)   # GPU upload — must be on main thread
                ch.dirty = False
                self.world.chunks[(cx, cz)] = ch
                uploaded += 1
            except queue.Empty:
                break

        # ── Queue missing chunks to background thread ──
        missing = sorted(
            (dx*dx + dz*dz, pcx+dx, pcz+dz)
            for dx in range(-rd, rd+1)
            for dz in range(-rd, rd+1)
            if (pcx+dx, pcz+dz) not in self.world.chunks
               and (pcx+dx, pcz+dz) not in self._in_flight
        )
        # Enqueue up to 4 nearest missing chunks (thread does the heavy work)
        for _, cx, cz in missing[:4]:
            self._in_flight.add((cx, cz))
            self._gen_queue.put((cx, cz))

        # ── Unload far chunks ──
        to_remove = [k for k in list(self.world.chunks)
                     if abs(k[0]-pcx) > rd+2
                     or abs(k[1]-pcz) > rd+2]
        for k in to_remove:
            ch = self.world.chunks.pop(k, None)
            if ch:
                if ch.vbo_solid: glDeleteBuffers(1, [ch.vbo_solid])
                if ch.vbo_water: glDeleteBuffers(1, [ch.vbo_water])

    def _rebuild_dirty(self):
        # Rebuild dirty chunks (only those modified by player block edits).
        # New chunks arrive pre-meshed from the background thread.
        for ch in self.world.chunks.values():
            if ch.dirty:
                solid, water = build_chunk_mesh(ch, self.world)
                upload_vbo(ch, solid, water)
                ch.dirty = False
                return   # one rebuild per frame max

    def _setup_3d(self):
        w, h = self.ww, self.wh
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70.0, w / max(1,h), 0.05, 400.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        ex, ey, ez = self.player.eye
        dx, dy, dz = self.player.look_dir()
        gluLookAt(ex,ey,ez, ex+dx,ey+dy,ez+dz, 0,1,0)

    def _setup_fog(self):
        far = (self.render_dist-1) * CHUNK_SIZE
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_LINEAR)
        glFogf(GL_FOG_START, far * 0.55)
        glFogf(GL_FOG_END,   far * 0.95)
        glFogfv(GL_FOG_COLOR, [0.50, 0.72, 1.00, 1.0])

    def _render_chunks(self):
        px, pz = self.player.x, self.player.z
        pcx, pcz = self.world.chunk_of(px, pz)

        # Update frustum from current matrices (called after gluLookAt)
        self.frustum.update()

        water_chunks = []
        for (cx,cz), ch in self.world.chunks.items():
            if abs(cx-pcx) > self.render_dist or abs(cz-pcz) > self.render_dist:
                continue
            # ← FRUSTUM CULL: skip chunks outside view
            if not self.frustum.chunk_visible(cx, cz):
                continue
            # Skip chunks with no geometry at all
            if ch.solid_count == 0 and ch.water_count == 0:
                continue
            if ch.vbo_solid and ch.solid_count:
                draw_vbo(ch.vbo_solid, ch.solid_count)
            if ch.vbo_water and ch.water_count:
                water_chunks.append(ch)

        # Translucent water pass (RGBA VBOs, alpha baked in)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for ch in water_chunks:
            draw_vbo_rgba(ch.vbo_water, ch.water_count)
        glDisable(GL_BLEND)

    def _add_chat(self, name: str, text: str):
        """Append to chat log (max 50 messages)."""
        self.chat_log.append((time.time(), name, text))
        if len(self.chat_log) > 50:
            self.chat_log.pop(0)

    def update(self, dt):
        if not self.esc_open and not self.chat_open:
            update_physics(self.world, self.player, self.keys, dt,
                           self.walk_speed, self.fly_speed)
        self._update_chunks()
        self._rebuild_dirty()
        if self._save_flash > 0:
            self._save_flash -= dt
        ex, ey, ez = self.player.eye
        dx, dy, dz = self.player.look_dir()
        hp, pp, hb = self.world.raycast(ex,ey,ez,dx,dy,dz)
        self.hit_pos   = hp
        self.place_pos = pp

        # ── Network tick ─────────────────────────────────
        if self.net:
            # Drain incoming messages
            while not self.net.inbox.empty():
                try:
                    msg = self.net.inbox.get_nowait()
                except queue.Empty:
                    break
                t = msg.get("type")

                if t == "players":
                    for p in msg.get("players", []):
                        pid = p["pid"]
                        if pid != self.net.pid:
                            self.remote_players[pid] = p

                elif t == "block":
                    self.world.set_block(
                        msg["x"], msg["y"], msg["z"], msg["block"])

                elif t == "chat":
                    self._add_chat(msg.get("name","?"), msg.get("text",""))

                elif t == "join":
                    self._add_chat("Server",
                        f"{msg.get('name','?')} joined the game")

                elif t == "leave":
                    pid = msg.get("pid")
                    self.remote_players.pop(pid, None)
                    self._add_chat("Server",
                        f"{msg.get('name','?')} left the game")

            # Send position at ~20 hz
            self._net_tick += dt
            if self._net_tick >= 0.05 and self.net.connected:
                self.net.send_pos(self.player)
                self._net_tick = 0.0

    def render(self):
        self._setup_3d()
        self._setup_fog()
        self._render_chunks()
        self._draw_remote_players()
        if self.hit_pos and not self.esc_open:
            draw_block_highlight(*self.hit_pos)
        hud_begin(self.ww, self.wh)
        draw_hotbar(self.inv, self.ww, self.wh)
        if not self.esc_open and not self.chat_open:
            draw_crosshair(self.ww//2, self.wh//2)
        if self.esc_open:
            self._draw_esc_menu()
        self._draw_chat_hud()
        if self.net and not self.net.connected and self.net.error:
            _gl_label(self.ww//2, self.wh//2 + 30,
                      "DISCONNECTED: " + self.net.error[:40],
                      scale=1.0, center=True, color=(1.0, 0.3, 0.3))
        hud_end()
        glfw.swap_buffers(self.window)

    def _draw_remote_players(self):
        """Draw each remote player as a coloured box with a name tag."""
        if not self.remote_players:
            return
        glDisable(GL_TEXTURE_2D)
        for p in self.remote_players.values():
            px, py, pz = p["x"], p["y"], p["z"]
            r, g, b = p.get("color", [0.8, 0.8, 0.8])
            hw = PLAYER_HW
            ph = PLAYER_HEIGHT
            glColor3f(r, g, b)
            glBegin(GL_QUADS)
            # 6 faces of player AABB
            faces = [
                # top
                [(px-hw,py+ph,pz-hw),(px+hw,py+ph,pz-hw),
                 (px+hw,py+ph,pz+hw),(px-hw,py+ph,pz+hw)],
                # bottom
                [(px-hw,py,pz+hw),(px+hw,py,pz+hw),
                 (px+hw,py,pz-hw),(px-hw,py,pz-hw)],
                # front
                [(px-hw,py,pz+hw),(px+hw,py,pz+hw),
                 (px+hw,py+ph,pz+hw),(px-hw,py+ph,pz+hw)],
                # back
                [(px+hw,py,pz-hw),(px-hw,py,pz-hw),
                 (px-hw,py+ph,pz-hw),(px+hw,py+ph,pz-hw)],
                # left
                [(px-hw,py,pz-hw),(px-hw,py,pz+hw),
                 (px-hw,py+ph,pz+hw),(px-hw,py+ph,pz-hw)],
                # right
                [(px+hw,py,pz+hw),(px+hw,py,pz-hw),
                 (px+hw,py+ph,pz-hw),(px+hw,py+ph,pz+hw)],
            ]
            for face in faces:
                for vx,vy,vz in face:
                    glVertex3f(vx, vy, vz)
            glEnd()
            # Outline
            glColor3f(0,0,0)
            glLineWidth(1.2)
            glBegin(GL_LINE_LOOP)
            glVertex3f(px-hw,py+ph,pz-hw); glVertex3f(px+hw,py+ph,pz-hw)
            glVertex3f(px+hw,py+ph,pz+hw); glVertex3f(px-hw,py+ph,pz+hw)
            glEnd()

    def _draw_chat_hud(self):
        """Draw chat log and input box."""
        now   = time.time()
        sw, sh = self.ww, self.wh
        y_base = 120
        line_h = 18
        # Show last 8 messages fading out after 10s (unless chat open)
        visible = [
            (ts, name, text) for (ts, name, text) in self.chat_log
            if self.chat_open or (now - ts) < 10.0
        ][-8:]
        for i, (ts, name, text) in enumerate(visible):
            age   = now - ts
            alpha = 1.0 if self.chat_open else max(0.0, 1.0 - (age - 7.0) / 3.0)
            if alpha <= 0: continue
            label = f"{name}: {text}"[:60]
            draw_rect(10, y_base + i*line_h - 2, min(len(label)*7+8, sw-20),
                      line_h, 0, 0, 0, 0.45 * alpha)
            _gl_label(14, y_base + i*line_h, label,
                      scale=1.0, center=False,
                      color=(0.95*alpha, 0.95*alpha, 0.95*alpha))

        if self.chat_open:
            # Input box
            box_y = y_base + len(visible) * line_h + 6
            draw_rect(10, box_y, sw - 20, 22, 0.05, 0.05, 0.10, 0.90)
            draw_rect_border(10, box_y, sw - 20, 22, 0.6, 0.6, 0.9, 1.5)
            cursor = "_" if int(now * 2) % 2 == 0 else ""
            _gl_label(14, box_y + 4,
                      "SAY: " + self.chat_input + cursor,
                      scale=1.0, center=False,
                      color=(0.9, 0.9, 1.0))


    # ─── ESC MENU HELPERS ─────────────────────────────────
    def save_world(self):
        """Save all loaded chunks + player position, show feedback flash."""
        save_all_chunks(self.world_idx, self.world)
        save_player_pos(self.world_idx, self.player)
        self._save_flash = 2.5   # show "SAVED" for 2.5 seconds

    def _menu_click_down(self):
        mx, my_raw = self.mouse_x, self.mouse_y
        # flip Y: OpenGL HUD is bottom-up, glfw cursor is top-down
        my = self.wh - my_raw

        r = self._menu_rects
        # check buttons
        if _pt_in(mx, my, r.get("resume", (0,0,0,0))):
            self.esc_open = False
            self._capture_mouse()
        elif _pt_in(mx, my, r.get("save", (0,0,0,0))):
            self.save_world()
        elif _pt_in(mx, my, r.get("change_world", (0,0,0,0))):
            self.change_world = True
        elif _pt_in(mx, my, r.get("quit", (0,0,0,0))):
            glfw.set_window_should_close(self.window, True)
        # check sliders (start drag)
        for name in ("render_dist", "walk_speed", "fly_speed"):
            if _pt_in(mx, my, r.get(f"slider_{name}", (0,0,0,0))):
                self._drag_slider = name

    def _menu_drag(self, mouse_x_screen):
        name = self._drag_slider
        if not name: return
        r = self._menu_rects.get(f"slider_{name}")
        if not r: return
        sx, sy, sw, sh = r
        t = max(0.0, min(1.0, (mouse_x_screen - sx) / sw))
        if name == "render_dist":
            self.render_dist = max(2, min(12, round(2 + t * 10)))
        elif name == "walk_speed":
            self.walk_speed = round(0.05 + t * 0.45, 3)
        elif name == "fly_speed":
            self.fly_speed  = round(0.10 + t * 0.90, 3)

    def _draw_esc_menu(self):
        sw, sh = self.ww, self.wh

        # dim background
        draw_rect(0, 0, sw, sh, 0, 0, 0, 0.55)

        # panel
        pw, ph = 420, 420
        px = (sw - pw) // 2
        py = (sh - ph) // 2
        draw_rect(px, py, pw, ph, 0.08, 0.08, 0.12, 0.97)
        draw_rect_border(px, py, pw, ph, 0.40, 0.40, 0.60, 2.0)

        rects = {}
        cx = px + pw // 2

        # title
        _gl_label(cx, py + ph - 38, "PAUSED", scale=2.0, center=True)

        # divider
        y = py + ph - 58
        draw_rect(px+20, y, pw-40, 1, 0.4, 0.4, 0.6, 0.8)

        # ── sliders ──
        sliders = [
            ("render_dist", "Render Distance",
             self.render_dist, 2, 12, True),
            ("walk_speed",  "Walk Speed",
             self.walk_speed, 0.05, 0.50, False),
            ("fly_speed",   "Fly Speed",
             self.fly_speed,  0.10, 1.00, False),
        ]
        sy_start = py + ph - 90
        for i, (name, label, val, vmin, vmax, is_int) in enumerate(sliders):
            ys = sy_start - i * 90
            # label + value
            disp = str(int(val)) if is_int else f"{val:.2f}"
            _gl_label(px+30, ys, f"{label}: {disp}", scale=1.2, center=False)
            # track
            tx, ty, tw, th = px+30, ys-28, pw-60, 14
            draw_rect(tx, ty, tw, th, 0.20, 0.20, 0.30, 1.0)
            # fill
            t = (val - vmin) / (vmax - vmin)
            draw_rect(tx, ty, int(tw*t), th, 0.40, 0.55, 0.95, 1.0)
            # thumb
            thumb_x = tx + int(tw*t) - 7
            draw_rect(thumb_x, ty-3, 14, th+6, 0.75, 0.85, 1.00, 1.0)
            draw_rect_border(thumb_x, ty-3, 14, th+6, 0.50, 0.70, 1.0, 1.5)
            # hit zone for drag (generous)
            rects[f"slider_{name}"] = (tx, ty-6, tw, th+12)

        # ── buttons: 2 rows of 2 ──
        bw, bh = 185, 40
        gap = 12
        total_btn_w = bw*2 + gap
        bx0 = cx - total_btn_w // 2

        row0_y = py + 84   # top row (quit / change world)
        row1_y = py + 28   # bottom row (save / resume) — drawn lower in panel

        buttons = [
            # (id,            label,          color-rgb,           row_y)
            ("resume",       "Resume",       (0.25, 0.55, 0.30),  row1_y),
            ("save",         "Save World",   (0.20, 0.50, 0.55),  row1_y),
            ("change_world", "Change World", (0.25, 0.35, 0.65),  row0_y),
            ("quit",         "Quit",         (0.55, 0.20, 0.20),  row0_y),
        ]
        mx_gl = self.mouse_x
        my_gl = self.wh - self.mouse_y   # flip Y

        for j, (bid, label, (r,g,b), row_y) in enumerate(buttons):
            col = j % 2
            bx = bx0 + col*(bw+gap)
            hovering = _pt_in(mx_gl, my_gl, (bx, row_y, bw, bh))
            br = min(r+0.12,1) if hovering else r
            bg_ = min(g+0.12,1) if hovering else g
            bb  = min(b+0.12,1) if hovering else b
            draw_rect(bx, row_y, bw, bh, br, bg_, bb, 1.0)
            draw_rect_border(bx, row_y, bw, bh, 0.7, 0.7, 0.7, 1.5)
            _gl_label(bx + bw//2, row_y + bh//2 - 5, label, scale=1.2, center=True)
            rects[bid] = (bx, row_y, bw, bh)

        # ── "SAVED!" flash ──
        if self._save_flash > 0:
            alpha = min(1.0, self._save_flash)   # fade out in last second
            flash_x = cx
            flash_y  = py + 140
            _gl_label(flash_x, flash_y, "SAVED!", scale=1.8, center=True,
                      color=(0.40*alpha, 1.0*alpha, 0.55*alpha))

        self._menu_rects = rects

    def run(self):
        self.init()
        prev = glfw.get_time()
        while not glfw.window_should_close(self.window) and not self.change_world:
            now = glfw.get_time()
            dt  = min(now - prev, 0.05)
            prev = now

            # FPS counter
            self._fps_frames += 1
            self._fps_time   += dt
            if self._fps_time >= 1.0:
                self._fps = self._fps_frames / self._fps_time
                glfw.set_window_title(self.window,
                    f"VoxelCraft — {self.world_name}  FPS: {self._fps:.0f}")
                self._fps_frames = 0
                self._fps_time   = 0.0

            glfw.poll_events()
            self.update(dt)
            self.render()

        # Disconnect from multiplayer server
        if self.net:
            self.net.disconnect()

        # Auto-save all chunks + player position before exiting
        print("[VoxelCraft] Auto-saving world…")
        save_all_chunks(self.world_idx, self.world)
        save_player_pos(self.world_idx, self.player)

        # Cleanup VBOs
        for ch in self.world.chunks.values():
            if ch.vbo_solid: glDeleteBuffers(1, [ch.vbo_solid])
            if ch.vbo_water: glDeleteBuffers(1, [ch.vbo_water])
        if not self.change_world:
            glfw.terminate()
            print("[VoxelCraft] Goodbye!")
        # If change_world is True, caller (main loop) handles re-launch


def _pt_in(mx, my, rect) -> bool:
    """Point-in-rect test. rect = (x, y, w, h) in HUD coords."""
    x, y, w, h = rect
    return x <= mx <= x+w and y <= my <= y+h

# Minimal bitmap font: 5x5 pixel chars drawn as tiny GL_QUADS
# Each char is a list of (row, col) lit pixels, row 0=top
_FONT = {
    'A':[(0,1),(0,2),(0,3),(1,0),(1,4),(2,0),(2,1),(2,2),(2,3),(2,4),(3,0),(3,4),(4,0),(4,4)],
    'B':[(0,0),(0,1),(0,2),(0,3),(1,0),(1,4),(2,0),(2,1),(2,2),(2,3),(3,0),(3,4),(4,0),(4,1),(4,2),(4,3)],
    'C':[(0,1),(0,2),(0,3),(1,0),(2,0),(3,0),(4,1),(4,2),(4,3)],
    'D':[(0,0),(0,1),(0,2),(1,0),(1,3),(2,0),(2,3),(3,0),(3,3),(4,0),(4,1),(4,2)],
    'E':[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(2,1),(2,2),(3,0),(4,0),(4,1),(4,2),(4,3)],
    'F':[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(2,1),(2,2),(3,0),(4,0)],
    'G':[(0,1),(0,2),(0,3),(1,0),(2,0),(2,2),(2,3),(3,0),(3,3),(4,1),(4,2),(4,3)],
    'H':[(0,0),(0,4),(1,0),(1,4),(2,0),(2,1),(2,2),(2,3),(2,4),(3,0),(3,4),(4,0),(4,4)],
    'I':[(0,0),(0,1),(0,2),(1,1),(2,1),(3,1),(4,0),(4,1),(4,2)],
    'J':[(0,2),(0,3),(1,3),(2,3),(3,0),(3,3),(4,1),(4,2)],
    'K':[(0,0),(0,3),(1,0),(1,2),(2,0),(2,1),(3,0),(3,2),(4,0),(4,3)],
    'L':[(0,0),(1,0),(2,0),(3,0),(4,0),(4,1),(4,2),(4,3)],
    'M':[(0,0),(0,4),(1,0),(1,1),(1,3),(1,4),(2,0),(2,2),(2,4),(3,0),(3,4),(4,0),(4,4)],
    'N':[(0,0),(0,4),(1,0),(1,1),(1,4),(2,0),(2,2),(2,4),(3,0),(3,3),(3,4),(4,0),(4,4)],
    'O':[(0,1),(0,2),(0,3),(1,0),(1,4),(2,0),(2,4),(3,0),(3,4),(4,1),(4,2),(4,3)],
    'P':[(0,0),(0,1),(0,2),(0,3),(1,0),(1,4),(2,0),(2,1),(2,2),(2,3),(3,0),(4,0)],
    'Q':[(0,1),(0,2),(0,3),(1,0),(1,4),(2,0),(2,4),(3,0),(3,2),(3,4),(4,1),(4,2),(4,3),(4,4)],
    'R':[(0,0),(0,1),(0,2),(0,3),(1,0),(1,4),(2,0),(2,1),(2,2),(2,3),(3,0),(3,2),(4,0),(4,3)],
    'S':[(0,1),(0,2),(0,3),(1,0),(2,1),(2,2),(2,3),(3,4),(4,1),(4,2),(4,3)],
    'T':[(0,0),(0,1),(0,2),(0,3),(0,4),(1,2),(2,2),(3,2),(4,2)],
    'U':[(0,0),(0,4),(1,0),(1,4),(2,0),(2,4),(3,0),(3,4),(4,1),(4,2),(4,3)],
    'V':[(0,0),(0,4),(1,0),(1,4),(2,0),(2,4),(3,1),(3,3),(4,2)],
    'W':[(0,0),(0,4),(1,0),(1,4),(2,0),(2,2),(2,4),(3,0),(3,1),(3,3),(3,4),(4,0),(4,4)],
    'X':[(0,0),(0,4),(1,1),(1,3),(2,2),(3,1),(3,3),(4,0),(4,4)],
    'Y':[(0,0),(0,4),(1,1),(1,3),(2,2),(3,2),(4,2)],
    'Z':[(0,0),(0,1),(0,2),(0,3),(0,4),(1,3),(2,2),(3,1),(4,0),(4,1),(4,2),(4,3),(4,4)],
    '0':[(0,1),(0,2),(0,3),(1,0),(1,4),(2,0),(2,2),(2,4),(3,0),(3,4),(4,1),(4,2),(4,3)],
    '1':[(0,1),(0,2),(1,2),(2,2),(3,2),(4,1),(4,2),(4,3)],
    '2':[(0,1),(0,2),(0,3),(1,4),(2,2),(2,3),(3,1),(4,0),(4,1),(4,2),(4,3),(4,4)],
    '3':[(0,0),(0,1),(0,2),(0,3),(1,4),(2,2),(2,3),(3,4),(4,0),(4,1),(4,2),(4,3)],
    '4':[(0,0),(0,3),(1,0),(1,3),(2,0),(2,1),(2,2),(2,3),(3,3),(4,3)],
    '5':[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(2,1),(2,2),(2,3),(3,4),(4,0),(4,1),(4,2),(4,3)],
    '6':[(0,1),(0,2),(0,3),(1,0),(2,0),(2,1),(2,2),(2,3),(3,0),(3,4),(4,1),(4,2),(4,3)],
    '7':[(0,0),(0,1),(0,2),(0,3),(0,4),(1,4),(2,3),(3,2),(4,2)],
    '8':[(0,1),(0,2),(0,3),(1,0),(1,4),(2,1),(2,2),(2,3),(3,0),(3,4),(4,1),(4,2),(4,3)],
    '9':[(0,1),(0,2),(0,3),(1,0),(1,4),(2,1),(2,2),(2,3),(2,4),(3,4),(4,1),(4,2),(4,3)],
    '.':[(4,1),(4,2)],
    ':':[(1,1),(1,2),(3,1),(3,2)],
    ' ':[],
    '-':[(2,0),(2,1),(2,2),(2,3),(2,4)],
    '/':[(0,4),(1,3),(2,2),(3,1),(4,0)],
}

def _gl_label(x, y, text, scale=1.0, center=False, color=(0.90, 0.90, 0.95)):
    """Draw a string using the bitmap font at HUD position (x, y)."""
    text = text.upper()
    ps   = max(1, int(2 * scale))   # pixel size
    cw   = (5 * ps) + ps            # char width incl spacing
    total_w = len(text) * cw
    cx = x - total_w // 2 if center else x
    glColor3f(*color)
    glBegin(GL_QUADS)
    for ci, ch in enumerate(text):
        pixels = _FONT.get(ch, [])
        ox = cx + ci * cw
        for (row, col) in pixels:
            px2 = ox + col * ps
            py2 = y + (4 - row) * ps   # flip row so row0=top
            glVertex2f(px2,      py2)
            glVertex2f(px2+ps,   py2)
            glVertex2f(px2+ps,   py2+ps)
            glVertex2f(px2,      py2+ps)
    glEnd()

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def _player_overlaps(player: Player, bx, by, bz) -> bool:
    w = PLAYER_HW
    return (player.x-w < bx+1 and player.x+w > bx and
            player.y     < by+1 and player.y+PLAYER_HEIGHT > by and
            player.z-w < bz+1 and player.z+w > bz)

# ─────────────────────────────────────────────────────────
# WORLD SAVE / SELECT
# ─────────────────────────────────────────────────────────
SAVE_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voxelcraft_worlds.json")
CHUNKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voxelcraft_chunks")
NUM_WORLDS = 3

def _world_chunks_dir(world_idx: int) -> str:
    """Return (and create) the directory that stores chunks for world_idx."""
    d = os.path.join(CHUNKS_DIR, str(world_idx))
    os.makedirs(d, exist_ok=True)
    return d

def save_chunk(world_idx: int, chunk: "Chunk"):
    """Persist a single chunk's block array to disk as a compressed .npz."""
    d = _world_chunks_dir(world_idx)
    path = os.path.join(d, f"{chunk.cx}_{chunk.cz}.npz")
    np.savez_compressed(path, blocks=chunk.blocks)

def load_chunk(world_idx: int, cx: int, cz: int) -> Optional[np.ndarray]:
    """Load and return the block array for (cx,cz), or None if not saved."""
    path = os.path.join(CHUNKS_DIR, str(world_idx), f"{cx}_{cz}.npz")
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)["blocks"].astype(np.uint8)
    except Exception:
        return None

def save_all_chunks(world_idx: int, world: "World"):
    """Save every loaded chunk to disk."""
    for ch in world.chunks.values():
        save_chunk(world_idx, ch)
    print(f"[VoxelCraft] Saved {len(world.chunks)} chunks for world {world_idx}.")

def save_player_pos(world_idx: int, player: "Player"):
    """Persist player position, look direction, and fly mode to disk."""
    d = _world_chunks_dir(world_idx)
    path = os.path.join(d, "player.json")
    data = {
        "x": player.x, "y": player.y, "z": player.z,
        "yaw": player.yaw, "pitch": player.pitch,
        "flying": player.flying,
    }
    with open(path, "w") as f:
        json.dump(data, f)

def load_player_pos(world_idx: int) -> Optional[dict]:
    """Return saved player state dict, or None if no save exists."""
    path = os.path.join(CHUNKS_DIR, str(world_idx), "player.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def load_worlds() -> list:
    """Return list of 3 world dicts with keys: name, seed, last_played."""
    default = [
        {"name": f"World {i+1}", "seed": None, "last_played": None}
        for i in range(NUM_WORLDS)
    ]
    if not os.path.exists(SAVE_FILE):
        return default
    try:
        with open(SAVE_FILE) as f:
            data = json.load(f)
        worlds = data.get("worlds", default)
        # Pad/trim to exactly NUM_WORLDS
        while len(worlds) < NUM_WORLDS:
            worlds.append({"name": f"World {len(worlds)+1}", "seed": None, "last_played": None})
        return worlds[:NUM_WORLDS]
    except Exception:
        return default

def save_worlds(worlds: list):
    with open(SAVE_FILE, "w") as f:
        json.dump({"worlds": worlds}, f, indent=2)

def touch_world(worlds: list, idx: int):
    """Assign a seed if new, update last_played timestamp."""
    if worlds[idx]["seed"] is None:
        worlds[idx]["seed"] = random.randint(0, 9_999_999)
    worlds[idx]["last_played"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_worlds(worlds)

# ─────────────────────────────────────────────────────────
# WORLD SELECT GUI  (tkinter — stdlib, no extra install)
# ─────────────────────────────────────────────────────────
def world_select_gui() -> tuple:
    """
    Show a Tkinter world-select window.
    Returns (world_index, seed, world_name) or raises SystemExit if cancelled.
    """
    import tkinter as tk
    from tkinter import simpledialog, messagebox

    worlds = load_worlds()
    chosen = [None]   # mutable container so inner funcs can write to it

    root = tk.Tk()
    root.title("VoxelCraft — Select World")
    root.resizable(False, False)
    root.configure(bg="#1a1a2e")

    # ── styles ──
    BG       = "#1a1a2e"
    CARD_BG  = "#16213e"
    ACCENT   = "#0f3460"
    BTN_BG   = "#533483"
    BTN_HOV  = "#7b52ab"
    TEXT     = "#e0e0e0"
    SUBTEXT  = "#9090a0"
    PLAYED   = "#6fcf97"
    FONT_H   = ("Segoe UI", 13, "bold")
    FONT_S   = ("Segoe UI", 9)
    FONT_BTN = ("Segoe UI", 10, "bold")
    FONT_TIT = ("Segoe UI", 18, "bold")

    tk.Label(root, text="⛰  VoxelCraft", font=FONT_TIT,
             bg=BG, fg=TEXT).pack(pady=(22, 4))
    tk.Label(root, text="Select a world to play", font=FONT_S,
             bg=BG, fg=SUBTEXT).pack(pady=(0, 6))

    # Multiplayer button at the top
    def open_multiplayer():
        chosen[0] = "multiplayer"
        root.quit()

    mp_btn = tk.Button(root, text="🌐  Multiplayer", font=FONT_BTN,
                       bg="#1a4a6a", fg=TEXT, relief="flat",
                       padx=16, pady=7, cursor="hand2",
                       command=open_multiplayer)
    mp_btn.pack(pady=(0, 12))
    mp_btn.bind("<Enter>", lambda e: mp_btn.config(bg="#2a6a9a"))
    mp_btn.bind("<Leave>", lambda e: mp_btn.config(bg="#1a4a6a"))

    frame = tk.Frame(root, bg=BG)
    frame.pack(padx=30, pady=0)

    def refresh_cards():
        for w in frame.winfo_children():
            w.destroy()
        for i, world in enumerate(worlds):
            card = tk.Frame(frame, bg=CARD_BG, bd=0, relief="flat",
                            padx=16, pady=12)
            card.grid(row=i, column=0, sticky="ew", pady=6)
            card.columnconfigure(0, weight=1)

            # world name (editable label trick: click to rename)
            name_var = tk.StringVar(value=world["name"])

            name_lbl = tk.Label(card, textvariable=name_var, font=FONT_H,
                                bg=CARD_BG, fg=TEXT, anchor="w", cursor="xterm")
            name_lbl.grid(row=0, column=0, sticky="w")

            if world["seed"] is None:
                sub = "New world — click Play to generate"
                sub_fg = SUBTEXT
            else:
                played = world["last_played"] or "never"
                sub = f"Seed: {world['seed']}   •   Last played: {played}"
                sub_fg = PLAYED

            tk.Label(card, text=sub, font=FONT_S,
                     bg=CARD_BG, fg=sub_fg, anchor="w").grid(row=1, column=0, sticky="w")

            btn_frame = tk.Frame(card, bg=CARD_BG)
            btn_frame.grid(row=0, column=1, rowspan=2, padx=(20,0))

            def make_play(idx):
                def play():
                    chosen[0] = idx
                    root.quit()
                return play

            def make_rename(idx, var):
                def rename():
                    new_name = simpledialog.askstring(
                        "Rename World",
                        f"New name for world {idx+1}:",
                        initialvalue=worlds[idx]["name"],
                        parent=root
                    )
                    if new_name and new_name.strip():
                        worlds[idx]["name"] = new_name.strip()
                        save_worlds(worlds)
                        var.set(new_name.strip())
                return rename

            def make_reset(idx):
                def reset():
                    if messagebox.askyesno(
                        "Reset World",
                        f"Reset world '{worlds[idx]['name']}'?\n"
                        "This will generate a brand-new world.",
                        parent=root
                    ):
                        worlds[idx]["seed"]        = None
                        worlds[idx]["last_played"] = None
                        save_worlds(worlds)
                        refresh_cards()
                return reset

            play_btn = tk.Button(btn_frame, text="▶  Play", font=FONT_BTN,
                                 bg=BTN_BG, fg=TEXT, relief="flat",
                                 padx=14, pady=6, cursor="hand2",
                                 command=make_play(i))
            play_btn.pack(side="left", padx=(0,6))
            play_btn.bind("<Enter>", lambda e, b=play_btn: b.config(bg=BTN_HOV))
            play_btn.bind("<Leave>", lambda e, b=play_btn: b.config(bg=BTN_BG))

            ren_btn = tk.Button(btn_frame, text="✎", font=FONT_BTN,
                                bg=ACCENT, fg=TEXT, relief="flat",
                                padx=8, pady=6, cursor="hand2",
                                command=make_rename(i, name_var))
            ren_btn.pack(side="left", padx=(0,4))
            ren_btn.bind("<Enter>", lambda e, b=ren_btn: b.config(bg="#1a4a7a"))
            ren_btn.bind("<Leave>", lambda e, b=ren_btn: b.config(bg=ACCENT))

            rst_btn = tk.Button(btn_frame, text="↺", font=FONT_BTN,
                                bg="#5c2a2a", fg=TEXT, relief="flat",
                                padx=8, pady=6, cursor="hand2",
                                command=make_reset(i))
            rst_btn.pack(side="left")
            rst_btn.bind("<Enter>", lambda e, b=rst_btn: b.config(bg="#8b3a3a"))
            rst_btn.bind("<Leave>", lambda e, b=rst_btn: b.config(bg="#5c2a2a"))

    refresh_cards()

    tk.Label(root, text="", bg=BG).pack(pady=10)

    def on_close():
        chosen[0] = None
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Centre on screen
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    ww = root.winfo_width()
    wh = root.winfo_height()
    root.geometry(f"+{(sw-ww)//2}+{(sh-wh)//2}")

    root.mainloop()
    root.destroy()

    if chosen[0] is None:
        sys.exit(0)

    if chosen[0] == "multiplayer":
        return "multiplayer", None, None

    idx = chosen[0]
    touch_world(worlds, idx)
    return idx, worlds[idx]["seed"], worlds[idx]["name"]

# ─────────────────────────────────────────────────────────
# MULTIPLAYER CONNECT GUI
# ─────────────────────────────────────────────────────────
def multiplayer_connect_gui():
    """
    Show server address + player name dialog.
    Returns (host, port, player_name) or None if cancelled.
    """
    import tkinter as tk
    from tkinter import messagebox

    result = [None]

    root = tk.Tk()
    root.title("VoxelCraft — Multiplayer")
    root.resizable(False, False)
    root.configure(bg="#1a1a2e")

    BG       = "#1a1a2e"
    CARD_BG  = "#16213e"
    TEXT     = "#e0e0e0"
    SUBTEXT  = "#9090a0"
    BTN_BG   = "#533483"
    BTN_HOV  = "#7b52ab"
    FONT_H   = ("Segoe UI", 13, "bold")
    FONT_S   = ("Segoe UI", 9)
    FONT_BTN = ("Segoe UI", 10, "bold")
    FONT_TIT = ("Segoe UI", 18, "bold")

    tk.Label(root, text="🌐  Multiplayer", font=FONT_TIT,
             bg=BG, fg=TEXT).pack(pady=(22, 4))
    tk.Label(root, text="Connect to a VoxelCraft server", font=FONT_S,
             bg=BG, fg=SUBTEXT).pack(pady=(0, 18))

    card = tk.Frame(root, bg=CARD_BG, padx=24, pady=20)
    card.pack(padx=30, pady=(0, 10))

    tk.Label(card, text="Server Address", font=FONT_S, bg=CARD_BG,
             fg=SUBTEXT, anchor="w").grid(row=0, column=0, sticky="w")
    host_var = tk.StringVar(value="")
    host_entry = tk.Entry(card, textvariable=host_var, font=FONT_H,
                          bg="#0a0a1e", fg=TEXT, insertbackground=TEXT,
                          relief="flat", width=28)
    host_entry.grid(row=1, column=0, pady=(2, 12), ipady=6)
    host_entry.focus_set()

    tk.Label(card, text="Port", font=FONT_S, bg=CARD_BG,
             fg=SUBTEXT, anchor="w").grid(row=2, column=0, sticky="w")
    port_var = tk.StringVar(value="25565")
    port_entry = tk.Entry(card, textvariable=port_var, font=FONT_H,
                          bg="#0a0a1e", fg=TEXT, insertbackground=TEXT,
                          relief="flat", width=28)
    port_entry.grid(row=3, column=0, pady=(2, 12), ipady=6)

    tk.Label(card, text="Your Name", font=FONT_S, bg=CARD_BG,
             fg=SUBTEXT, anchor="w").grid(row=4, column=0, sticky="w")
    import random as _r
    name_var = tk.StringVar(value=f"Player{_r.randint(100,999)}")
    name_entry = tk.Entry(card, textvariable=name_var, font=FONT_H,
                          bg="#0a0a1e", fg=TEXT, insertbackground=TEXT,
                          relief="flat", width=28)
    name_entry.grid(row=5, column=0, pady=(2, 4), ipady=6)

    def do_connect():
        h = host_var.get().strip()
        n = name_var.get().strip() or "Player"
        try:
            p = int(port_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Port must be a number", parent=root)
            return
        if not h:
            messagebox.showerror("Error", "Enter a server address", parent=root)
            return
        result[0] = (h, p, n)
        root.quit()

    def do_cancel():
        root.quit()

    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(pady=(0, 20))

    conn_btn = tk.Button(btn_frame, text="Connect", font=FONT_BTN,
                         bg=BTN_BG, fg=TEXT, relief="flat",
                         padx=18, pady=8, cursor="hand2",
                         command=do_connect)
    conn_btn.pack(side="left", padx=8)
    conn_btn.bind("<Enter>", lambda e: conn_btn.config(bg=BTN_HOV))
    conn_btn.bind("<Leave>", lambda e: conn_btn.config(bg=BTN_BG))

    back_btn = tk.Button(btn_frame, text="Back", font=FONT_BTN,
                         bg="#333355", fg=TEXT, relief="flat",
                         padx=18, pady=8, cursor="hand2",
                         command=do_cancel)
    back_btn.pack(side="left", padx=8)

    root.bind("<Return>", lambda e: do_connect())
    root.protocol("WM_DELETE_WINDOW", do_cancel)

    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw-root.winfo_width())//2}+{(sh-root.winfo_height())//2}")
    root.mainloop()
    root.destroy()
    return result[0]


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        idx, seed, name = world_select_gui()

        if idx == "multiplayer":
            conn_info = multiplayer_connect_gui()
            if conn_info is None:
                continue   # user hit Back → return to world select
            host, port, player_name = conn_info
            # Join as client: use world idx 0 (local copy for chunk gen)
            game = Game(seed=None, world_name=f"MP:{host}", world_idx=0)
            game.net = NetworkClient(host, port, player_name)
            # Wait briefly for connection
            import time as _t
            _t.sleep(0.8)
            if not game.net.connected:
                # Show error briefly then loop back
                import tkinter as tk
                from tkinter import messagebox
                _r = tk.Tk(); _r.withdraw()
                messagebox.showerror("Connection Failed",
                    game.net.error or f"Could not connect to {host}:{port}")
                _r.destroy()
                continue
        else:
            game = Game(seed=seed, world_name=name, world_idx=idx)

        game.run()
        if not game.change_world:
            break
        try:
            glfw.destroy_window(game.window)
        except Exception:
            pass