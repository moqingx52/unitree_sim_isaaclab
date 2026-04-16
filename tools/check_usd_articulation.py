#!/usr/bin/env python3
"""Quick USD inspector for units, joints, and articulation APIs."""

import argparse
from pathlib import Path


def main() -> int:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        print("[ERROR] pxr module not found in current Python environment.")
        print("[HINT] Run this script with Isaac Sim Python, for example:")
        print("       /path/to/isaac-sim/python.sh tools/check_usd_articulation.py <usd_path>")
        return 2

    parser = argparse.ArgumentParser(description="Inspect a USD asset for articulation readiness.")
    parser.add_argument("usd_path", type=str, help="Path to the USD/USD[A|C|Z] file.")
    args = parser.parse_args()

    usd_path = Path(args.usd_path).expanduser().resolve()
    if not usd_path.exists():
        print(f"[ERROR] File not found: {usd_path}")
        return 1

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print(f"[ERROR] Failed to open USD stage: {usd_path}")
        return 1

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    print(f"[INFO] USD: {usd_path}")
    print(f"[INFO] metersPerUnit={meters_per_unit}, upAxis={up_axis}")

    articulation_roots = []
    rigid_bodies = []
    revolute_joints = []
    all_joints = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim.GetPath().pathString)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim.GetPath().pathString)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute_joints.append(prim.GetPath().pathString)
        if prim.IsA(UsdPhysics.Joint):
            all_joints.append(prim.GetPath().pathString)

    print(f"[INFO] articulation roots: {len(articulation_roots)}")
    for path in articulation_roots:
        print(f"  - {path}")

    print(f"[INFO] rigid bodies: {len(rigid_bodies)}")
    print(f"[INFO] joints: {len(all_joints)}")
    print(f"[INFO] revolute joints: {len(revolute_joints)}")

    if revolute_joints:
        print("[INFO] revolute joint candidates:")
        for path in revolute_joints:
            print(f"  - {path}")

    if not all_joints:
        print("[WARN] No physics joints found. This USD cannot open the door as articulation yet.")
    elif not articulation_roots:
        print("[WARN] Joints exist but no ArticulationRootAPI found; authoring may still be incomplete.")
    else:
        print("[OK] USD has joints and articulation root candidates.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
