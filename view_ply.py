"""
Standalone viser viewer for a point_cloud.ply exported by demo.py's --export_results
(x, y, z, r, g, b, confidence per vertex -- see export_point_cloud_ply in demo.py).
Loads the file back and opens the same kind of browser-based viewer demo.py uses live.
"""

import argparse
import time

import numpy as np
import viser

parser = argparse.ArgumentParser()
parser.add_argument("ply_path", type=str)
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--point_size", type=float, default=0.01)
args = parser.parse_args()


def load_ply(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        assert f.readline().strip() == b"format ascii 1.0"
        n_vertex = int(f.readline().split()[-1])
        while f.readline().strip() != b"end_header":
            pass
        data = np.loadtxt(f, max_rows=n_vertex)
    points = data[:, 0:3].astype(np.float32)
    colors = data[:, 3:6].astype(np.uint8)
    confidence = data[:, 6] if data.shape[1] > 6 else None
    return points, colors, confidence


points, colors, confidence = load_ply(args.ply_path)
print(f"Loaded {points.shape[0]} points from {args.ply_path}")
if confidence is not None:
    print(f"Confidence: mean={confidence.mean():.3f} min={confidence.min():.3f} max={confidence.max():.3f}")

server = viser.ViserServer(port=args.port)
server.scene.add_point_cloud(
    name="/point_cloud",
    points=points,
    colors=colors,
    point_size=args.point_size,
)

print(f"Viewer running at http://localhost:{args.port}")
while True:
    time.sleep(1)
