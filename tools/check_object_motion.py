import argparse
import csv
import json
import math
import os
from collections import defaultdict


CLASS_NAMES = [
    "Person",
    "Forklift",
    "NovaCarter",
    "Transporter",
    "FourierGR1T2",
    "AgilityDigit",
    "PalletTruck",
]
CLASS_TO_LABEL = {name: label for label, name in enumerate(CLASS_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate object motion summary CSV from ground_truth.json files."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root directory containing <split>/<scene>/ground_truth.json.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to scan.",
    )
    parser.add_argument(
        "--class-name",
        default="all",
        help='Class name to analyze, or "all" for all classes.',
    )
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=0.5,
        help="Total displacement threshold in meters for object-level movement.",
    )
    parser.add_argument(
        "--step-threshold",
        type=float,
        default=0.1,
        help="Consecutive-frame step threshold in meters for moving intervals.",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/object_motion_summary.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def distance_xyz(a, b):
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def normalize_location(location):
    if not isinstance(location, list) or len(location) < 3:
        return None
    try:
        return [float(location[0]), float(location[1]), float(location[2])]
    except (TypeError, ValueError):
        return None


def load_ground_truth(gt_path):
    with open(gt_path, "r") as f:
        data = json.load(f)

    observations_by_class = defaultdict(lambda: defaultdict(list))
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
            class_name = obj.get("object type")
            if class_name not in CLASS_TO_LABEL:
                continue
            location = normalize_location(obj.get("3d location"))
            if location is None:
                continue
            object_id = str(obj.get("object id", "unknown"))
            observations_by_class[class_name][object_id].append((frame, location))

    for objects_by_id in observations_by_class.values():
        for object_id in objects_by_id:
            objects_by_id[object_id].sort(key=lambda item: item[0])

    return observations_by_class


def find_moving_intervals(observations, step_threshold):
    intervals = []
    start_frame = None
    end_frame = None
    has_moving_step = False

    for idx in range(1, len(observations)):
        prev_frame, prev_xyz = observations[idx - 1]
        frame, xyz = observations[idx]
        step_distance = distance_xyz(prev_xyz, xyz)
        is_moving_step = step_distance >= step_threshold
        is_consecutive = frame == prev_frame + 1

        if is_moving_step:
            has_moving_step = True
            if start_frame is None or not is_consecutive:
                if start_frame is not None:
                    intervals.append((start_frame, end_frame))
                start_frame = frame
            end_frame = frame
            continue

        if start_frame is not None:
            intervals.append((start_frame, end_frame))
            start_frame = None
            end_frame = None

    if start_frame is not None:
        intervals.append((start_frame, end_frame))

    return intervals, has_moving_step


def summarize_object(observations, movement_threshold, step_threshold):
    first_frame, start_xyz = observations[0]
    last_frame, end_xyz = observations[-1]
    total_displacement = distance_xyz(start_xyz, end_xyz)
    intervals, has_moving_step = find_moving_intervals(observations, step_threshold)
    is_moving = total_displacement >= movement_threshold or has_moving_step
    return {
        "first_frame": first_frame,
        "last_frame": last_frame,
        "is_moving": is_moving,
        "intervals": intervals,
    }


def merge_intervals(intervals):
    if not intervals:
        return []

    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def format_intervals(intervals):
    if not intervals:
        return "None"
    return ";".join(f"{start}->{end}" for start, end in intervals)


def summarize_class_scene(
    split,
    scene,
    class_name,
    objects_by_id,
    movement_threshold,
    step_threshold,
):
    object_summaries = [
        summarize_object(observations, movement_threshold, step_threshold)
        for observations in objects_by_id.values()
        if observations
    ]
    if not object_summaries:
        return None

    all_intervals = []
    for summary in object_summaries:
        all_intervals.extend(summary["intervals"])
    moving_intervals = merge_intervals(all_intervals)
    moving_object_count = sum(1 for summary in object_summaries if summary["intervals"])
    static_object_count = len(object_summaries) - moving_object_count
    fr_start = min(summary["first_frame"] for summary in object_summaries)
    fr_end = max(summary["last_frame"] for summary in object_summaries)

    return {
        "split": split,
        "scene": scene,
        "class_label": CLASS_TO_LABEL[class_name],
        "class_name": class_name,
        "num_objects": len(object_summaries),
        "present_frame_range": f"{fr_start}->{fr_end}",
        "fr_start": fr_start,
        "fr_end": fr_end,
        "move": "TRUE" if moving_object_count > 0 else "FALSE",
        "moving_object_count": moving_object_count,
        "static_object_count": static_object_count,
        "moving_intervals": format_intervals(moving_intervals),
    }


def get_target_classes(class_name):
    if class_name == "all":
        return CLASS_NAMES
    if class_name not in CLASS_TO_LABEL:
        raise ValueError(
            f"Unknown class name: {class_name}. Expected one of: {', '.join(CLASS_NAMES)}"
        )
    return [class_name]


def get_scene_names(split_path):
    if not os.path.isdir(split_path):
        print(f"Warning: split directory not found: {split_path}")
        return []
    return sorted(
        name
        for name in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, name))
    )


def write_csv(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    columns = [
        "split",
        "scene",
        "class_label",
        "class_name",
        "num_objects",
        "present_frame_range",
        "fr_start",
        "fr_end",
        "move",
        "moving_object_count",
        "static_object_count",
        "moving_intervals",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    target_classes = get_target_classes(args.class_name)
    rows = []

    for split in args.splits:
        split_path = os.path.join(args.data_root, split)
        for scene in get_scene_names(split_path):
            gt_path = os.path.join(split_path, scene, "ground_truth.json")
            if not os.path.exists(gt_path):
                print(f"Warning: missing ground_truth.json: {gt_path}")
                continue

            observations_by_class = load_ground_truth(gt_path)
            for class_name in target_classes:
                objects_by_id = observations_by_class.get(class_name, {})
                row = summarize_class_scene(
                    split=split,
                    scene=scene,
                    class_name=class_name,
                    objects_by_id=objects_by_id,
                    movement_threshold=args.movement_threshold,
                    step_threshold=args.step_threshold,
                )
                if row is not None:
                    rows.append(row)

    rows.sort(key=lambda row: (row["split"], row["scene"], row["class_label"]))
    write_csv(rows, args.output_csv)

    total_moving = sum(1 for row in rows if row["move"] == "TRUE")
    total_static = len(rows) - total_moving
    print(f"total_rows: {len(rows)}")
    print(f"total_moving_class_scene_rows: {total_moving}")
    print(f"total_static_class_scene_rows: {total_static}")
    print(f"output_csv: {args.output_csv}")


if __name__ == "__main__":
    main()
