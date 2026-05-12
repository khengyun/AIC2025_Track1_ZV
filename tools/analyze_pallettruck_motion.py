import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass


@dataclass
class MoveSegment:
    start_frame: int
    end_frame: int
    start_xyz: list
    end_xyz: list
    distance_m: float


@dataclass
class ObjectSummary:
    split: str
    scene: str
    object_id: str
    num_observations: int
    first_frame: int
    last_frame: int
    start_xyz: list
    end_xyz: list
    net_displacement_m: float
    path_length_m: float
    max_step_m: float
    is_moving: bool
    move_segments: list


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze PalletTruck motion directly from AIC 2025 Track 1 "
            "ground_truth.json files."
        )
    )
    parser.add_argument(
        "--data-root",
        default="dataset/MTMC_Tracking_2025",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to scan. Default: train val",
    )
    parser.add_argument(
        "--scene-name",
        action="append",
        default=None,
        help="Optional scene filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--target-class",
        default="PalletTruck",
        help="Object class to analyze.",
    )
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=0.5,
        help="Object-level movement threshold in meters.",
    )
    parser.add_argument(
        "--step-threshold",
        type=float,
        default=0.1,
        help="Consecutive-observation step threshold in meters.",
    )
    parser.add_argument(
        "--min-segment-distance",
        type=float,
        default=0.3,
        help="Minimum total distance in meters for a reported move segment.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional report directory for CSV and JSON outputs.",
    )
    return parser.parse_args()


def distance_xyz(a, b):
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def format_xyz(xyz):
    return f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"


def xyz_to_csv(xyz):
    return f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}"


def normalize_location(location):
    if not isinstance(location, list) or len(location) < 3:
        return None
    try:
        return [float(location[0]), float(location[1]), float(location[2])]
    except (TypeError, ValueError):
        return None


def load_target_observations(gt_path, target_class):
    with open(gt_path, "r") as f:
        data = json.load(f)

    observations = defaultdict(list)
    for frame_key, objects in data.items():
        try:
            frame = int(frame_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(objects, list):
            continue

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if obj.get("object type") != target_class:
                continue
            location = normalize_location(obj.get("3d location"))
            if location is None:
                continue
            object_id = str(obj.get("object id", "unknown"))
            observations[object_id].append((frame, location))

    for object_id in observations:
        observations[object_id].sort(key=lambda item: item[0])
    return observations


def find_move_segments(observations, step_threshold, min_segment_distance):
    segments = []
    current_start = None
    current_end = None
    current_distance = 0.0

    for idx in range(1, len(observations)):
        step_distance = distance_xyz(observations[idx - 1][1], observations[idx][1])

        if step_distance >= step_threshold:
            if current_start is None:
                current_start = idx - 1
                current_distance = 0.0
            current_end = idx
            current_distance += step_distance
            continue

        if current_start is not None and current_distance >= min_segment_distance:
            start_frame, start_xyz = observations[current_start]
            end_frame, end_xyz = observations[current_end]
            segments.append(
                MoveSegment(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_xyz=start_xyz,
                    end_xyz=end_xyz,
                    distance_m=current_distance,
                )
            )
        current_start = None
        current_end = None
        current_distance = 0.0

    if current_start is not None and current_distance >= min_segment_distance:
        start_frame, start_xyz = observations[current_start]
        end_frame, end_xyz = observations[current_end]
        segments.append(
            MoveSegment(
                start_frame=start_frame,
                end_frame=end_frame,
                start_xyz=start_xyz,
                end_xyz=end_xyz,
                distance_m=current_distance,
            )
        )

    return segments


def summarize_object(
    split,
    scene,
    object_id,
    observations,
    movement_threshold,
    step_threshold,
    min_segment_distance,
):
    first_frame, start_xyz = observations[0]
    last_frame, end_xyz = observations[-1]
    net_displacement = distance_xyz(start_xyz, end_xyz)

    path_length = 0.0
    max_step = 0.0
    for idx in range(1, len(observations)):
        step = distance_xyz(observations[idx - 1][1], observations[idx][1])
        path_length += step
        max_step = max(max_step, step)

    move_segments = find_move_segments(
        observations,
        step_threshold=step_threshold,
        min_segment_distance=min_segment_distance,
    )
    is_moving = (
        net_displacement >= movement_threshold
        or path_length >= movement_threshold
        or len(move_segments) > 0
    )

    return ObjectSummary(
        split=split,
        scene=scene,
        object_id=object_id,
        num_observations=len(observations),
        first_frame=first_frame,
        last_frame=last_frame,
        start_xyz=start_xyz,
        end_xyz=end_xyz,
        net_displacement_m=net_displacement,
        path_length_m=path_length,
        max_step_m=max_step,
        is_moving=is_moving,
        move_segments=[asdict(segment) for segment in move_segments],
    )


def segment_to_text(segment):
    return (
        f"{segment['start_frame']}->{segment['end_frame']} "
        f"{format_xyz(segment['start_xyz'])}->{format_xyz(segment['end_xyz'])} "
        f"{segment['distance_m']:.3f}m"
    )


def print_scene_report(split, scene, summaries, target_class):
    if not summaries:
        print(f"[{split}/{scene}] no {target_class}")
        return

    print(f"[{split}/{scene}] {target_class} objects: {len(summaries)}")
    for summary in summaries:
        status = "MOVING" if summary.is_moving else "STATIC"
        print(
            f"  id={summary.object_id} {status} "
            f"obs={summary.num_observations} "
            f"frames={summary.first_frame}->{summary.last_frame} "
            f"net={summary.net_displacement_m:.3f}m "
            f"path={summary.path_length_m:.3f}m "
            f"max_step={summary.max_step_m:.3f}m"
        )
        if summary.is_moving:
            if summary.move_segments:
                for segment in summary.move_segments:
                    print(f"    move {segment_to_text(segment)}")
            else:
                print("    move segment: none above min-segment-distance")


def get_scene_names(split_path, scene_filter):
    if scene_filter is not None:
        return sorted(scene_filter)
    if not os.path.isdir(split_path):
        return []
    return sorted(
        name
        for name in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, name))
    )


def write_reports(summaries, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pallettruck_motion_summary.csv")
    json_path = os.path.join(out_dir, "pallettruck_motion_summary.json")

    columns = [
        "split",
        "scene",
        "object_id",
        "num_observations",
        "first_frame",
        "last_frame",
        "start_xyz",
        "end_xyz",
        "net_displacement_m",
        "path_length_m",
        "max_step_m",
        "is_moving",
        "move_segments",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            row = asdict(summary)
            row["start_xyz"] = xyz_to_csv(summary.start_xyz)
            row["end_xyz"] = xyz_to_csv(summary.end_xyz)
            row["net_displacement_m"] = f"{summary.net_displacement_m:.6f}"
            row["path_length_m"] = f"{summary.path_length_m:.6f}"
            row["max_step_m"] = f"{summary.max_step_m:.6f}"
            row["move_segments"] = json.dumps(summary.move_segments)
            writer.writerow(row)

    with open(json_path, "w") as f:
        json.dump([asdict(summary) for summary in summaries], f, indent=2)

    print("\nReports written:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")


def main():
    args = parse_args()
    scene_filter = set(args.scene_name) if args.scene_name else None
    all_summaries = []
    scenes_with_target = []
    scenes_without_target = []

    print(
        f"Analyzing class={args.target_class} "
        f"data_root={args.data_root} "
        f"splits={','.join(args.splits)}"
    )
    print(
        f"thresholds: movement={args.movement_threshold:.3f}m, "
        f"step={args.step_threshold:.3f}m, "
        f"min_segment={args.min_segment_distance:.3f}m\n"
    )

    for split in args.splits:
        split_path = os.path.join(args.data_root, split)
        if not os.path.isdir(split_path):
            print(f"Warning: split directory not found: {split_path}")
            continue

        scene_names = get_scene_names(split_path, scene_filter)
        if not scene_names:
            print(f"Warning: no scenes found for split {split}")
            continue

        for scene in scene_names:
            scene_path = os.path.join(split_path, scene)
            gt_path = os.path.join(scene_path, "ground_truth.json")
            if not os.path.exists(gt_path):
                print(f"Warning: missing ground_truth.json: {gt_path}")
                scenes_without_target.append((split, scene))
                continue

            observations_by_object = load_target_observations(gt_path, args.target_class)
            scene_summaries = []
            for object_id in sorted(observations_by_object):
                observations = observations_by_object[object_id]
                if not observations:
                    continue
                scene_summaries.append(
                    summarize_object(
                        split=split,
                        scene=scene,
                        object_id=object_id,
                        observations=observations,
                        movement_threshold=args.movement_threshold,
                        step_threshold=args.step_threshold,
                        min_segment_distance=args.min_segment_distance,
                    )
                )

            if scene_summaries:
                scenes_with_target.append((split, scene))
            else:
                scenes_without_target.append((split, scene))

            print_scene_report(split, scene, scene_summaries, args.target_class)
            all_summaries.extend(scene_summaries)

    print("\nSummary")
    print(f"  Scenes with {args.target_class}: {len(scenes_with_target)}")
    for split, scene in scenes_with_target:
        print(f"    {split}/{scene}")
    print(f"  Scenes without {args.target_class}: {len(scenes_without_target)}")
    for split, scene in scenes_without_target:
        print(f"    {split}/{scene}")
    print(f"  Objects analyzed: {len(all_summaries)}")

    if args.out_dir:
        write_reports(all_summaries, args.out_dir)


if __name__ == "__main__":
    main()
