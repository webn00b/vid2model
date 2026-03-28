#!/usr/bin/env python3
from __future__ import annotations

from vid2model_lib import (
    CHILDREN,
    JOINTS,
    LM,
    MAP_TO_POINTS,
    MODEL_URLS,
    JointDef,
    build_rest_offsets,
    bvh_hierarchy_lines,
    channel_headers,
    convert_video_to_bvh,
    ensure_pose_model,
    euler_zxy_from_matrix,
    extract_pose_points,
    frame_channels,
    main,
    normalize,
    parse_args,
    rotation_align,
    write_bvh,
    write_csv,
    write_json,
    write_npz,
    write_trc,
)


if __name__ == "__main__":
    raise SystemExit(main())
