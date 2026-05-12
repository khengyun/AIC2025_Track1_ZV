#!/usr/bin/env python3
import argparse
import gc
import json
import os
import threading
import time

import numpy as np

o3d = None
gui = None
rendering = None


CLASS_NAMES = {
    0: "Person",
    1: "Forklift",
    2: "NovaCarter",
    3: "Transporter",
    4: "FourierGR1T2",
    5: "AgilityDigit",
    6: "PalletTruck",
}

CLASS_TO_LABEL = {name: label for label, name in CLASS_NAMES.items()}

CLASS_COLORS = {
    "Person": [0.0, 1.0, 0.0],
    "Forklift": [1.0, 0.55, 0.0],
    "NovaCarter": [0.0, 0.35, 1.0],
    "Transporter": [0.65, 0.0, 1.0],
    "FourierGR1T2": [0.0, 1.0, 1.0],
    "AgilityDigit": [1.0, 0.0, 0.65],
    "PalletTruck": [1.0, 1.0, 0.0],
}


def load_open3d(include_gui=True):
    global o3d, gui, rendering
    if o3d is not None and (not include_gui or gui is not None):
        return
    try:
        import open3d as open3d_module
    except ImportError as e:
        raise RuntimeError("Open3D is required to run this viewer.") from e
    o3d = open3d_module
    if include_gui:
        try:
            from open3d.visualization import gui as gui_module
            from open3d.visualization import rendering as rendering_module
        except ImportError as e:
            raise RuntimeError("Open3D with GUI support is required to run this viewer.") from e
        gui = gui_module
        rendering = rendering_module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Open3D GUI viewer for tracking GT object IDs across point-cloud frames."
    )
    parser.add_argument("--processing-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene-name", default="Warehouse_000")
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=100)
    parser.add_argument("--mode", choices=["dynamic", "full", "static"], default="dynamic")
    parser.add_argument("--static-voxel-size", type=float, default=0.05)
    parser.add_argument("--distance-threshold", type=float, default=0.10)
    parser.add_argument("--point-budget", type=int, default=300000)
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--initial-show-gt", action="store_true", default=False)
    parser.add_argument("--initial-show-ids", action="store_true", default=False)
    parser.add_argument("--initial-show-static", action="store_true", default=False)
    parser.add_argument("--initial-show-static-in-box", action="store_true", default=False)
    parser.add_argument("--debug-memory", action="store_true")
    parser.add_argument("--dry-load", action="store_true")
    return parser.parse_args()


def voxel_mm(voxel_size):
    return int(round(voxel_size * 1000))


def threshold_mm(distance_threshold):
    return int(round(distance_threshold * 1000))


def list_scenes(processing_root):
    pcd_root = os.path.join(processing_root, "point_cloud")
    if not os.path.isdir(pcd_root):
        return []
    return [
        name
        for name in sorted(os.listdir(pcd_root))
        if os.path.isdir(os.path.join(pcd_root, name)) and name.startswith("Warehouse_")
    ]


def build_point_cloud_path(args, scene_name, frame_id, mode):
    if mode == "dynamic":
        return os.path.join(
            args.processing_root,
            "point_cloud_object",
            scene_name,
            f"{scene_name}_{frame_id:05d}_dynamic_thr{threshold_mm(args.distance_threshold):03d}.ply",
        )
    if mode == "full":
        return os.path.join(
            args.processing_root,
            "point_cloud",
            scene_name,
            "pcd",
            f"{scene_name}_{frame_id:05d}.ply",
        )
    return os.path.join(
        args.processing_root,
        "point_cloud_background",
        scene_name,
        f"{scene_name}_static_voxel{voxel_mm(args.static_voxel_size):03d}.ply",
    )


def build_gt_path(args, scene_name):
    return os.path.join(args.data_root, args.split, scene_name, "ground_truth.json")


def memory_rss_mb():
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def log_memory(enabled, label):
    if not enabled:
        return
    rss = memory_rss_mb()
    if rss is None:
        print(f"[MEM] {label}: psutil unavailable")
    else:
        print(f"[MEM] {label}: RSS={rss:.1f} MB")


def get_rotation(rotation):
    return o3d.geometry.get_rotation_matrix_from_xyz(tuple(np.asarray(rotation, dtype=np.float64).tolist()))


def make_box(label, object_id, center, extent, rotation):
    class_name = CLASS_NAMES.get(label, f"unknown_{label}")
    return {
        "label": int(label),
        "class_name": class_name,
        "object_id": object_id,
        "center": np.asarray(center, dtype=np.float64),
        "extent": np.asarray(extent, dtype=np.float64),
        "rotation": np.asarray(rotation, dtype=np.float64),
    }


def load_gt_boxes(gt_path, frame_id):
    if not os.path.exists(gt_path):
        return []
    with open(gt_path, "r") as f:
        data = json.load(f)
    objects = data.get(str(frame_id), [])
    if not isinstance(objects, list):
        return []

    boxes = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        class_name = obj.get("object type")
        if class_name not in CLASS_TO_LABEL:
            continue
        try:
            boxes.append(
                make_box(
                    CLASS_TO_LABEL[class_name],
                    obj.get("object id", "unknown"),
                    obj["3d location"],
                    obj["3d bounding box scale"],
                    obj["3d bounding box rotation"],
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return boxes


def object_id_matches(value, wanted):
    if wanted is None or wanted == "":
        return False
    try:
        return int(value) == int(wanted)
    except (TypeError, ValueError):
        return str(value) == str(wanted)


def class_color(class_name):
    return CLASS_COLORS.get(class_name, [1.0, 1.0, 1.0])


def focused_color(class_name):
    color = np.asarray(class_color(class_name), dtype=np.float64)
    return np.clip(color * 1.35 + 0.15, 0.0, 1.0).tolist()


def make_material(shader="defaultUnlit", color=None, point_size=None, line_width=None):
    material = rendering.MaterialRecord()
    material.shader = shader
    if color is not None:
        rgba = np.asarray(color, dtype=np.float64)
        if len(rgba) == 3:
            rgba = np.concatenate([rgba, np.asarray([1.0])])
        material.base_color = np.clip(rgba, 0.0, 1.0).tolist()
    if point_size is not None and hasattr(material, "point_size"):
        material.point_size = float(point_size)
    if line_width is not None and hasattr(material, "line_width"):
        material.line_width = float(line_width)
    return material


def create_box_lineset(box, color):
    obb = o3d.geometry.OrientedBoundingBox(box["center"], get_rotation(box["rotation"]), box["extent"])
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lineset.paint_uniform_color(color)
    return lineset


def label_position(box):
    return box["center"] + np.asarray([0.0, 0.0, box["extent"][2] / 2.0 + 0.5], dtype=np.float64)


def sample_point_cloud(pcd, point_budget):
    point_count = len(pcd.points)
    print(f"points before sampling: {point_count}")
    if point_budget <= 0 or point_count <= point_budget:
        print(f"points after sampling: {point_count}")
        return pcd

    indices = np.random.default_rng(0).choice(point_count, size=point_budget, replace=False)
    sampled = o3d.geometry.PointCloud()
    sampled.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[indices])
    if pcd.has_colors():
        sampled.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[indices])
    if pcd.has_normals():
        sampled.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[indices])
    print(f"points after sampling: {len(sampled.points)}")
    return sampled


def ensure_point_colors(pcd, color):
    if not pcd.has_colors():
        pcd.paint_uniform_color(color)


def bounds_for_scene(pcd, boxes):
    bounds = []
    if len(pcd.points) > 0:
        points = np.asarray(pcd.points)
        bounds.extend([np.min(points, axis=0), np.max(points, axis=0)])
    for box in boxes:
        obb = o3d.geometry.OrientedBoundingBox(box["center"], get_rotation(box["rotation"]), box["extent"])
        corners = np.asarray(obb.get_box_points())
        bounds.extend([np.min(corners, axis=0), np.max(corners, axis=0)])

    if not bounds:
        return o3d.geometry.AxisAlignedBoundingBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

    stacked = np.vstack(bounds)
    min_bound = np.min(stacked, axis=0)
    max_bound = np.max(stacked, axis=0)
    if np.linalg.norm(max_bound - min_bound) < 1e-6:
        min_bound -= 1.0
        max_bound += 1.0
    return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)


def static_points_inside_boxes(static_pcd, boxes):
    if len(static_pcd.points) == 0 or not boxes:
        return o3d.geometry.PointCloud()
    matched = np.zeros(len(static_pcd.points), dtype=bool)
    for box in boxes:
        obb = o3d.geometry.OrientedBoundingBox(box["center"], get_rotation(box["rotation"]), box["extent"])
        indices = np.asarray(obb.get_point_indices_within_bounding_box(static_pcd.points), dtype=np.int64)
        if len(indices) > 0:
            matched[indices] = True
    points = np.asarray(static_pcd.points)[matched]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(points) > 0:
        pcd.paint_uniform_color([1.0, 0.0, 0.0])
    return pcd


def get_theme_font_size(window):
    return window.theme.font_size if hasattr(window.theme, "font_size") else 16


class ObjectTrackingViewer:
    def __init__(self, args):
        self.args = args
        self.scene_name = args.scene_name
        self.frame_id = int(args.frame_id)
        self.mode = args.mode
        if args.safe_mode:
            self.mode = "dynamic"
        self.frame_stride = int(args.frame_stride)
        self.point_budget = int(args.point_budget)
        self.playing = False
        self.play_thread = None
        self.stop_event = threading.Event()
        self.labels = []
        self.last_camera_bounds = None
        self.last_focus_center = None

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("Object Tracking Viewer", 1500, 900)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.0, 0.0, 0.0, 1.0])
        self.panel = gui.Vert(8, gui.Margins(12, 12, 12, 12))

        self._build_ui()
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        if hasattr(self.scene_widget, "enable_scene_caching"):
            self.scene_widget.enable_scene_caching(True)
        if hasattr(self.scene_widget, "set_view_controls"):
            self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _build_ui(self):
        self.scene_combo = gui.Combobox()
        scenes = list_scenes(self.args.processing_root)
        if self.scene_name not in scenes:
            scenes.insert(0, self.scene_name)
        for scene in scenes:
            self.scene_combo.add_item(scene)
        self.scene_combo.selected_index = max(scenes.index(self.scene_name), 0)
        self.scene_combo.set_on_selection_changed(self._on_scene_changed)

        self.frame_input = gui.TextEdit()
        self.frame_input.text_value = str(self.frame_id)

        self.object_input = gui.TextEdit()
        self.object_input.text_value = ""

        self.mode_combo = gui.Combobox()
        modes = ["dynamic", "full", "static"]
        for mode in modes:
            self.mode_combo.add_item(mode)
        self.mode_combo.selected_index = modes.index(self.mode)
        self.mode_combo.set_on_selection_changed(self._on_mode_changed)

        self.show_gt_checkbox = gui.Checkbox("Show GT boxes")
        self.show_gt_checkbox.checked = bool(self.args.initial_show_gt)
        self.show_id_checkbox = gui.Checkbox("Show IDs")
        self.show_id_checkbox.checked = bool(self.args.initial_show_ids)
        self.show_static_checkbox = gui.Checkbox("Show static")
        self.show_static_checkbox.checked = bool(self.args.initial_show_static)
        self.show_static_in_box_checkbox = gui.Checkbox("Show static inside GT boxes")
        self.show_static_in_box_checkbox.checked = bool(self.args.initial_show_static_in_box)
        self.only_focused_checkbox = gui.Checkbox("Only focused object")
        self.only_focused_checkbox.checked = False
        self.paint_dynamic_checkbox = gui.Checkbox("Paint dynamic")
        self.paint_dynamic_checkbox.checked = True

        for checkbox in [
            self.show_gt_checkbox,
            self.show_id_checkbox,
            self.show_static_checkbox,
            self.show_static_in_box_checkbox,
            self.only_focused_checkbox,
            self.paint_dynamic_checkbox,
        ]:
            checkbox.set_on_checked(lambda checked: self.load_frame(follow_focus=False))

        self.load_button = gui.Button("Load frame")
        self.load_button.set_on_clicked(lambda: self.load_frame_from_ui(follow_focus=False))
        self.focus_button = gui.Button("Focus object")
        self.focus_button.set_on_clicked(self.focus_object)
        self.prev_button = gui.Button("Previous frame")
        self.prev_button.set_on_clicked(lambda: self.step_frame(-self.frame_stride))
        self.next_button = gui.Button("Next frame")
        self.next_button.set_on_clicked(lambda: self.step_frame(self.frame_stride))
        self.play_button = gui.Button("Play/Pause")
        self.play_button.set_on_clicked(self.toggle_playback)

        self.interval_slider = gui.Slider(gui.Slider.INT)
        self.interval_slider.set_limits(100, 3000)
        self.interval_slider.int_value = 500

        self.status_label = gui.Label("")
        self.focus_label = gui.Label("")

        self.panel.add_child(gui.Label("Scene"))
        self.panel.add_child(self.scene_combo)
        self.panel.add_child(gui.Label("Frame ID"))
        self.panel.add_child(self.frame_input)
        self.panel.add_child(gui.Label("Object ID"))
        self.panel.add_child(self.object_input)
        self.panel.add_child(gui.Label("Mode"))
        self.panel.add_child(self.mode_combo)
        self.panel.add_child(self.show_gt_checkbox)
        self.panel.add_child(self.show_id_checkbox)
        self.panel.add_child(self.show_static_checkbox)
        self.panel.add_child(self.show_static_in_box_checkbox)
        self.panel.add_child(self.only_focused_checkbox)
        self.panel.add_child(self.paint_dynamic_checkbox)
        self.panel.add_child(self.load_button)
        self.panel.add_child(self.focus_button)
        self.panel.add_child(self.prev_button)
        self.panel.add_child(self.next_button)
        self.panel.add_child(self.play_button)
        self.panel.add_child(gui.Label("Playback interval ms"))
        self.panel.add_child(self.interval_slider)
        self.panel.add_child(gui.Label("Status"))
        self.panel.add_child(self.status_label)
        self.panel.add_child(gui.Label("Focused object"))
        self.panel.add_child(self.focus_label)

    def _on_layout(self, context):
        del context
        rect = self.window.content_rect
        em = get_theme_font_size(self.window)
        panel_width = int(24 * em)
        self.scene_widget.frame = gui.Rect(rect.x, rect.y, max(rect.width - panel_width, 1), rect.height)
        self.panel.frame = gui.Rect(rect.x + rect.width - panel_width, rect.y, panel_width, rect.height)

    def _on_close(self):
        self.playing = False
        self.stop_event.set()
        return True

    def _on_scene_changed(self, text, index):
        del index
        self.scene_name = text
        self.load_frame(follow_focus=False)

    def _on_mode_changed(self, text, index):
        del index
        self.mode = text
        self.load_frame(follow_focus=False)

    def run(self):
        def delayed_initial_load():
            time.sleep(0.1)
            self.app.post_to_main_thread(self.window, lambda: self.load_frame(follow_focus=False))

        threading.Thread(target=delayed_initial_load, daemon=True).start()
        self.app.run()

    def parse_frame_input(self):
        try:
            self.frame_id = max(0, int(self.frame_input.text_value.strip()))
        except ValueError:
            self.set_status(f"Invalid frame id: {self.frame_input.text_value}")
        return self.frame_id

    def load_frame_from_ui(self, follow_focus):
        self.parse_frame_input()
        self.mode = self.mode_combo.selected_text
        self.scene_name = self.scene_combo.selected_text
        self.load_frame(follow_focus=follow_focus)

    def step_frame(self, delta):
        self.frame_id = max(0, int(self.frame_id) + int(delta))
        self.frame_input.text_value = str(self.frame_id)
        self.load_frame(follow_focus=True)

    def current_object_id(self):
        return self.object_input.text_value.strip()

    def find_focused_box(self, boxes):
        object_id = self.current_object_id()
        if not object_id:
            return None
        for box in boxes:
            if object_id_matches(box["object_id"], object_id):
                return box
        return None

    def clear_labels(self):
        for label in self.labels:
            try:
                self.scene_widget.remove_3d_label(label)
            except (AttributeError, RuntimeError):
                pass
        self.labels = []

    def set_status(self, text):
        self.status_label.text = text
        print(text)

    def set_focus_status(self, box, found):
        if found and box is not None:
            center = np.asarray(box["center"], dtype=np.float64)
            text = (
                f"found\n"
                f"class={box['class_name']}\n"
                f"object_id={box['object_id']}\n"
                f"center=[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]"
            )
        elif self.current_object_id():
            text = f"missing\nobject_id={self.current_object_id()}"
        else:
            text = "none"
        self.focus_label.text = text

    def load_point_cloud(self, mode=None, paint_dynamic=True):
        mode = self.mode if mode is None else mode
        path = build_point_cloud_path(self.args, self.scene_name, self.frame_id, mode)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        log_memory(self.args.debug_memory, f"before load {mode}")
        pcd = o3d.io.read_point_cloud(path)
        loaded_count = len(pcd.points)
        if len(pcd.points) == 0:
            raise RuntimeError(f"empty point cloud: {path}")
        sampled = sample_point_cloud(pcd, self.point_budget)
        del pcd
        gc.collect()
        log_memory(self.args.debug_memory, f"after load {mode}")
        if mode == "dynamic" and paint_dynamic and self.paint_dynamic_checkbox.checked:
            sampled.paint_uniform_color([0.0, 1.0, 0.0])
        elif mode == "static":
            sampled.paint_uniform_color([0.55, 0.55, 0.55])
        else:
            ensure_point_colors(sampled, [0.8, 0.8, 0.8])
        print(
            f"[LOAD] mode={mode} points_before_sampling={loaded_count} "
            f"points_after_sampling={len(sampled.points)} path={path}"
        )
        return sampled, path

    def load_frame(self, follow_focus):
        try:
            pcd, pcd_path = self.load_point_cloud()
            gt_path = build_gt_path(self.args, self.scene_name)
            boxes = []
            needs_gt = (
                self.show_gt_checkbox.checked
                or self.show_id_checkbox.checked
                or self.show_static_in_box_checkbox.checked
                or bool(self.current_object_id())
            )
            if needs_gt:
                boxes = load_gt_boxes(gt_path, self.frame_id)
            focused_box = self.find_focused_box(boxes)
            focused_found = focused_box is not None
            self.render_scene(pcd, boxes, focused_box, follow_focus)

            point_count = len(pcd.points)
            del pcd
            gc.collect()
            log_memory(self.args.debug_memory, "after add_geometry")

            status = (
                f"scene={self.scene_name}\n"
                f"frame_id={self.frame_id}\n"
                f"mode={self.mode}\n"
                f"point_count={point_count}\n"
                f"GT boxes={len(boxes)}\n"
                f"focused={'found' if focused_found else 'missing' if self.current_object_id() else 'none'}"
            )
            self.set_status(status)
            self.set_focus_status(focused_box, focused_found)
            print(
                f"[FRAME] scene={self.scene_name} frame_id={self.frame_id} mode={self.mode} "
                f"point_count={point_count} gt_boxes={len(boxes)} "
                f"focused={'found' if focused_found else 'missing' if self.current_object_id() else 'none'} "
                f"path={pcd_path}"
            )
            if self.current_object_id() and not focused_found:
                print(
                    f"[WARN] object_id={self.current_object_id()} missing in "
                    f"{self.scene_name}/{self.frame_id:05d}; keeping previous camera target."
                )
        except Exception as e:
            self.set_status(
                f"scene={self.scene_name}\n"
                f"frame_id={self.frame_id}\n"
                f"mode={self.mode}\n"
                f"error={e}"
            )
            self.set_focus_status(None, False)
            print(f"[WARN] failed to load frame: {e}")

    def render_scene(self, pcd, boxes, focused_box, follow_focus):
        self.scene_widget.scene.clear_geometry()
        self.clear_labels()

        point_material = make_material("defaultUnlit", point_size=2.0)
        self.scene_widget.scene.add_geometry("point_cloud", pcd, point_material)
        log_memory(self.args.debug_memory, "after add dynamic geometry")

        draw_boxes = []
        static_layer = None
        if self.show_static_checkbox.checked:
            try:
                static_layer, static_path = self.load_point_cloud(mode="static", paint_dynamic=False)
                self.scene_widget.scene.add_geometry(
                    "static_background",
                    static_layer,
                    make_material("defaultUnlit", point_size=1.0),
                )
                print(f"[LAYER] static points={len(static_layer.points)} path={static_path}")
                log_memory(self.args.debug_memory, "after add static geometry")
            except Exception as e:
                print(f"[WARN] failed to load static layer: {e}")

        if self.show_gt_checkbox.checked or self.show_id_checkbox.checked:
            for index, box in enumerate(boxes):
                is_focus = focused_box is not None and object_id_matches(box["object_id"], focused_box["object_id"])
                if self.only_focused_checkbox.checked and not is_focus:
                    continue
                color = focused_color(box["class_name"]) if is_focus else class_color(box["class_name"])
                if self.show_gt_checkbox.checked:
                    line_width = 5.0 if is_focus else 2.0
                    lineset = create_box_lineset(box, color)
                    self.scene_widget.scene.add_geometry(
                        f"gt_box_{index}",
                        lineset,
                        make_material("unlitLine", color=color, line_width=line_width),
                    )
                    draw_boxes.append(box)

                should_show_id = self.show_id_checkbox.checked or is_focus
                if should_show_id:
                    label = self.scene_widget.add_3d_label(
                        label_position(box).astype(np.float32),
                        str(box["object_id"]),
                    )
                    if label is not None:
                        label.color = gui.Color(1.0, 1.0, 0.0, 1.0)
                        label.scale = 1.0 if not is_focus else 1.25
                        self.labels.append(label)

        if self.show_static_in_box_checkbox.checked:
            if static_layer is None:
                try:
                    static_layer, static_path = self.load_point_cloud(mode="static", paint_dynamic=False)
                    print(f"[LAYER] static-for-box points={len(static_layer.points)} path={static_path}")
                except Exception as e:
                    print(f"[WARN] failed to load static-in-box source: {e}")
                    static_layer = None
            if static_layer is not None:
                static_box_pcd = static_points_inside_boxes(static_layer, boxes)
                self.scene_widget.scene.add_geometry(
                    "static_inside_gt_boxes",
                    static_box_pcd,
                    make_material("defaultUnlit", point_size=5.0),
                )
                print(f"[LAYER] static_inside_gt_boxes points={len(static_box_pcd.points)}")
                log_memory(self.args.debug_memory, "after add static-in-box geometry")
                del static_box_pcd

        bounds = bounds_for_scene(pcd, draw_boxes)
        if not draw_boxes and static_layer is not None:
            bounds = bounds_for_scene(static_layer, [])
        if static_layer is not None:
            del static_layer
            gc.collect()
        if focused_box is not None and follow_focus:
            self.last_focus_center = np.asarray(focused_box["center"], dtype=np.float64)
            focus_extent = max(float(np.linalg.norm(focused_box["extent"])), 3.0)
            focus_bounds = o3d.geometry.AxisAlignedBoundingBox(
                self.last_focus_center - focus_extent,
                self.last_focus_center + focus_extent,
            )
            self.scene_widget.setup_camera(60.0, focus_bounds, self.last_focus_center)
            self.last_camera_bounds = focus_bounds
        elif self.last_camera_bounds is None:
            self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
            self.last_camera_bounds = bounds
        elif focused_box is not None:
            self.last_focus_center = np.asarray(focused_box["center"], dtype=np.float64)

    def focus_object(self):
        self.parse_frame_input()
        self.load_frame(follow_focus=True)

    def toggle_playback(self):
        self.playing = not self.playing
        if self.playing:
            self.stop_event.clear()
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()
            self.set_status(f"{self.status_label.text}\nplayback=on")
        else:
            self.stop_event.set()
            self.set_status(f"{self.status_label.text}\nplayback=off")

    def _play_loop(self):
        while not self.stop_event.is_set() and self.playing:
            interval = max(0.1, float(self.interval_slider.int_value) / 1000.0)
            time.sleep(interval)
            if self.stop_event.is_set() or not self.playing:
                break

            def advance():
                if not self.playing:
                    return
                self.frame_id = max(0, int(self.frame_id) + int(self.frame_stride))
                self.frame_input.text_value = str(self.frame_id)
                self.load_frame(follow_focus=True)

            self.app.post_to_main_thread(self.window, advance)


def dry_load(args):
    mode = "dynamic" if args.safe_mode else args.mode
    scene_name = args.scene_name
    frame_id = int(args.frame_id)
    pcd_path = build_point_cloud_path(args, scene_name, frame_id, mode)
    gt_path = build_gt_path(args, scene_name)

    print("[DRY-LOAD]")
    print(f"scene={scene_name}")
    print(f"frame_id={frame_id}")
    print(f"mode={mode}")
    print(f"point_cloud_path={pcd_path}")
    print(f"gt_path={gt_path}")

    log_memory(args.debug_memory, "before dry point load")
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(pcd_path)
    pcd = o3d.io.read_point_cloud(pcd_path)
    before_count = len(pcd.points)
    if before_count == 0:
        raise RuntimeError(f"empty point cloud: {pcd_path}")
    sampled = sample_point_cloud(pcd, args.point_budget)
    after_count = len(sampled.points)
    del pcd
    del sampled
    gc.collect()
    log_memory(args.debug_memory, "after dry point load")

    boxes = load_gt_boxes(gt_path, frame_id)
    print(f"points_before_sampling={before_count}")
    print(f"points_after_sampling={after_count}")
    print(f"gt_boxes={len(boxes)}")
    log_memory(args.debug_memory, "after dry gt load")


def main():
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    if args.static_voxel_size <= 0:
        raise ValueError("--static-voxel-size must be positive.")
    if args.distance_threshold < 0:
        raise ValueError("--distance-threshold must be non-negative.")
    if args.point_budget <= 0:
        raise ValueError("--point-budget must be positive.")

    if args.dry_load:
        load_open3d(include_gui=False)
        dry_load(args)
        return

    load_open3d(include_gui=True)
    viewer = ObjectTrackingViewer(args)
    viewer.run()


if __name__ == "__main__":
    main()
