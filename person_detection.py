import argparse
import cv2
import json
import csv
import os
import sqlite3
from datetime import datetime
import numpy as np
try:
    import imageio
except Exception:
    imageio = None
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="yoloworld.pt")
    p.add_argument("--source", type=str, default="dataset")
    p.add_argument("--dataset_dir", type=str, default="dataset")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--frame_w", type=int, default=640)
    p.add_argument("--frame_h", type=int, default=640)
    p.add_argument("--output_video", type=str, default="")
    p.add_argument("--log", type=str, default="")
    p.add_argument("--show", action="store_true")
    p.add_argument("--zones", type=str, default="")
    p.add_argument("--line_x", type=int, default=-1)
    p.add_argument("--line_ratio", type=float, default=0.55)
    p.add_argument("--employee_side", type=str, default="right")
    p.add_argument("--db_path", type=str, default="counter.db")
    p.add_argument("--frames_dir", type=str, default="data/frames")
    p.add_argument("--calibration_dir", type=str, default="data/calibration")
    p.add_argument("--force_imageio", action="store_true")
    p.add_argument("--frame_stride", type=int, default=2)
    p.add_argument("--hand_threshold", type=float, default=0.4)
    p.add_argument("--world_model", type=str, default="yoloworld.pt")
    p.add_argument("--hand_model", type=str, default="hand_yolov8.pt")
    return p.parse_args()

def open_source(s):
    if s.isdigit():
        return cv2.VideoCapture(int(s))
    return cv2.VideoCapture(s)

def resolve_source(source, dataset_dir):
    if source.lower() == "dataset" or source.strip() == "":
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        d = Path(dataset_dir)
        if d.exists():
            files = sorted([str(p) for p in d.iterdir() if p.suffix.lower() in exts])
            if files:
                return files[0]
        return "0"
    return source

def create_writer(path, fps, size):
    if not path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps if fps > 0 else 30.0, size)

def setup_logger(path):
    if not path:
        return None, None
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(["frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence"])
        return f, w
    else:
        f = open(path, "w", encoding="utf-8")
        return f, None

def ensure_dirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_zones(zones_path, frame_w, default_ratio, explicit_line_x, employee_side):
    if zones_path and Path(zones_path).exists():
        with open(zones_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "p1" in data and "p2" in data:
            p1 = tuple(map(int, data.get("p1")))
            p2 = tuple(map(int, data.get("p2")))
            emp_side = str(data.get("employee_side", employee_side))
            return (p1, p2), emp_side
        line_x = int(data.get("line_x", int(frame_w * default_ratio)))
        emp_side = str(data.get("employee_side", employee_side))
        return line_x, emp_side
    if explicit_line_x and explicit_line_x > 0:
        return int(explicit_line_x), employee_side
    return int(frame_w * default_ratio), employee_side

def write_zones(zones_path, line_x=None, employee_side="right", p1=None, p2=None):
    if not zones_path:
        return
    if p1 is not None and p2 is not None:
        data = {"p1": [int(p1[0]), int(p1[1])], "p2": [int(p2[0]), int(p2[1])], "employee_side": str(employee_side)}
    else:
        data = {"line_x": int(line_x), "employee_side": str(employee_side)}
    with open(zones_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_track_id INTEGER,
            employee_track_id INTEGER,
            timestamp_start TEXT,
            timestamp_end TEXT,
            cnic_detected INTEGER,
            classification TEXT,
            cnic_image_path TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER,
            event_type TEXT,
            image_path TEXT,
            zero_shot_result TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    return conn

def start_interaction(conn, customer_track_id, employee_track_id):
    ts = datetime.now().isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO interactions (customer_track_id, employee_track_id, timestamp_start, cnic_detected, classification) VALUES (?, ?, ?, ?, ?)",
        (customer_track_id, employee_track_id, ts, 0, "")
    )
    conn.commit()
    return cur.lastrowid

def end_interaction(conn, interaction_id, cnic_detected, classification, cnic_image_path):
    ts = datetime.now().isoformat()
    cur = conn.cursor()
    cur.execute(
        "UPDATE interactions SET timestamp_end=?, cnic_detected=?, classification=?, cnic_image_path=? WHERE interaction_id=?",
        (ts, 1 if cnic_detected else 0, classification, cnic_image_path or "", interaction_id)
    )
    conn.commit()

def insert_event(conn, interaction_id, image_path, zero_shot_result, event_type="HAND_CROSSING"):
    ts = datetime.now().isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events (interaction_id, event_type, image_path, zero_shot_result, timestamp) VALUES (?, ?, ?, ?, ?)",
        (interaction_id, event_type, image_path, zero_shot_result or "", ts)
    )
    conn.commit()

def save_frame(frame, frames_dir, interaction_id):
    ensure_dirs(frames_dir)
    sub = Path(frames_dir) / f"interaction_{interaction_id}"
    sub.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = str(sub / f"frame_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path

def save_sample_frame(cap, calibration_dir):
    ret, frame = cap.read()
    if not ret:
        return ""
    ensure_dirs(calibration_dir)
    path = str(Path(calibration_dir) / "sample_frame.jpg")
    cv2.imwrite(path, frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return path

def save_sample_frame_reader(reader, calibration_dir):
    try:
        frame = reader.get_data(0)
    except Exception:
        return ""
    ensure_dirs(calibration_dir)
    path = str(Path(calibration_dir) / "sample_frame.jpg")
    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return path

def open_source_reader(path):
    if imageio is None:
        return None
    try:
        return imageio.get_reader(path)
    except Exception:
        return None

def point_side(p, p1, p2):
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

def on_employee_side(cx, cy, p1, p2, employee_side, line_x_current):
    if p1 is not None and p2 is not None:
        x1, y1 = p1
        x2, y2 = p2
        if abs(x2 - x1) < 1e-6:
            if employee_side == "right":
                return cx > x1
            else:
                return cx < x1
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        x_on_line = (cy - b) / m
        if employee_side == "right":
            return cx > x_on_line
        else:
            return cx < x_on_line
    else:
        if employee_side == "right":
            return cx > line_x_current
        else:
            return cx < line_x_current

def point_in_rect(px, py, x1, y1, x2, y2):
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)

def segments_intersect(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    if (o1 == 0 and min(p1[0], p2[0]) <= q1[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= q1[1] <= max(p1[1], p2[1])):
        return True
    if (o2 == 0 and min(p1[0], p2[0]) <= q2[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= q2[1] <= max(p1[1], p2[1])):
        return True
    if (o3 == 0 and min(q1[0], q2[0]) <= p1[0] <= max(q1[0], q2[0]) and min(q1[1], q2[1]) <= p1[1] <= max(q1[1], q2[1])):
        return True
    if (o4 == 0 and min(q1[0], q2[0]) <= p2[0] <= max(q1[0], q2[0]) and min(q1[1], q2[1]) <= p2[1] <= max(q1[1], q2[1])):
        return True
    return (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0)

def rect_touches_line(x1, y1, x2, y2, p1, p2, line_x_current):
    if p1 is not None and p2 is not None:
        if point_in_rect(p1[0], p1[1], x1, y1, x2, y2) or point_in_rect(p2[0], p2[1], x1, y1, x2, y2):
            return True
        if segments_intersect(p1, p2, (x1, y1), (x2, y1)):
            return True
        if segments_intersect(p1, p2, (x2, y1), (x2, y2)):
            return True
        if segments_intersect(p1, p2, (x2, y2), (x1, y2)):
            return True
        if segments_intersect(p1, p2, (x1, y2), (x1, y1)):
            return True
        return False
    else:
        return x1 <= line_x_current <= x2

def get_class_name(names, cls_idx):
    try:
        if isinstance(names, dict):
            return str(names.get(int(cls_idx), ""))
        if isinstance(names, list):
            return str(names[int(cls_idx)])
    except Exception:
        pass
    return ""

def crop_and_save(frame, bbox, frames_dir, interaction_id, prefix="hand"):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    ensure_dirs(frames_dir)
    sub = Path(frames_dir) / f"interaction_{interaction_id}"
    sub.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = str(sub / f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, crop)
    return path

def detect_zero_shot(model, image, threshold):
    try:
        model.set_classes(["identity card", "id card", "national id card", "plastic card", "cnic"])
    except Exception:
        pass
    res = model.predict(image, conf=threshold, verbose=False)
    r = res[0]
    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and boxes.xyxy is not None else np.empty((0, 4))
    clss = boxes.cls.cpu().numpy() if boxes is not None and boxes.cls is not None else np.empty((0,))
    names = r.names if hasattr(r, "names") and r.names is not None else {}
    found = False
    for i in range(len(xyxy)):
        name = names.get(int(clss[i]), "") if isinstance(names, dict) else ""
        if name in {"identity card", "id card", "national id card", "plastic card", "cnic"}:
            found = True
            break
    return "cnic" if found else "no_cnic"

def log_detection(handle, writer, frame_idx, bbox, conf):
    if handle is None:
        return
    if writer is not None:
        writer.writerow([frame_idx, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), float(conf)])
    else:
        obj = {"frame": int(frame_idx), "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])], "confidence": float(conf)}
        handle.write(json.dumps(obj) + "\n")

def main():
    args = parse_args()
    world_path = Path(args.world_model)
    if not world_path.exists():
        world = YOLO("yolov8s-world.pt")
    else:
        world = YOLO(str(world_path))
    try:
        world.set_classes(["person", "identity card", "id card", "national id card", "plastic card", "cnic"])
    except Exception:
        pass
    hand_path = None
    if args.hand_model:
        hp = Path(args.hand_model)
        if hp.exists():
            hand_path = hp
    if hand_path is None and args.model:
        mp = Path(args.model)
        if mp.exists():
            hand_path = mp
    hand = None
    if hand_path is not None:
        hand = YOLO(str(hand_path))
    src = resolve_source(args.source, args.dataset_dir)
    print(f"Input source: {src}")
    cap = open_source(src)
    use_reader = False
    reader = None
    if (args.force_imageio and not src.isdigit()) or (not cap.isOpened() and (not src.isdigit())):
        reader = open_source_reader(src)
        use_reader = reader is not None
        if use_reader:
            meta = {}
            try:
                meta = reader.get_meta_data()
            except Exception:
                meta = {}
            fps = float(meta.get("fps", 30.0))
            print("Using imageio for video reading.")
        else:
            print("Failed to open video with both OpenCV and imageio. Check codec/format.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
    writer = create_writer(args.output_video, fps, (args.frame_w, args.frame_h))
    log_handle, log_writer = setup_logger(args.log)
    conn = init_db(args.db_path)
    ensure_dirs(args.frames_dir)
    if use_reader:
        save_sample_frame_reader(reader, args.calibration_dir)
    else:
        save_sample_frame(cap, args.calibration_dir)
    frame_idx = 0
    interaction_id = None
    cnic_detected_flag = False
    cnic_image_path = ""
    last_persons = []
    last_hands_roles = []
    last_visitor_box = None
    last_employee_box = None
    line_def, emp_side = load_zones(args.zones, args.frame_w, args.line_ratio, args.line_x, args.employee_side)
    p1 = None
    p2 = None
    line_x_current = line_def if isinstance(line_def, int) else args.frame_w // 2
    if isinstance(line_def, tuple):
        p1, p2 = line_def
    line_dirty = False
    if args.show:
        cv2.namedWindow("detections", cv2.WINDOW_NORMAL)
        def on_mouse(event, x, y, flags, param):
            nonlocal line_x_current, line_dirty, p1, p2
            if event == cv2.EVENT_LBUTTONDOWN:
                if p1 is None:
                    p1 = (int(x), int(y))
                    line_dirty = True
                elif p2 is None:
                    p2 = (int(x), int(y))
                    line_dirty = True
                else:
                    p1 = (int(x), int(y))
                    p2 = None
                    line_dirty = True
        cv2.setMouseCallback("detections", on_mouse)
    while True:
        if use_reader:
            try:
                frame = reader.get_next_data()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ok = True
            except Exception:
                ok = False
        else:
            ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = cv2.resize(frame, (args.frame_w, args.frame_h))
        if args.frame_stride > 1 and (frame_idx % args.frame_stride != 0):
            if p1 is not None and p2 is not None:
                cv2.line(frame, p1, p2, (0, 255, 255), 2)
            else:
                cv2.line(frame, (line_x_current, 0), (line_x_current, frame.shape[0]), (0, 255, 255), 2)
            for pb in last_persons:
                x1, y1, x2, y2, pconf = pb
                is_visitor = last_visitor_box is not None and (x1, y1, x2, y2, pconf) == last_visitor_box
                is_employee = last_employee_box is not None and (x1, y1, x2, y2, pconf) == last_employee_box
                color = (0, 255, 0) if is_visitor else ((0, 0, 255) if is_employee else (0, 255, 0))
                label_role = ("CUSTOMER" if cnic_detected_flag else "VISITOR") if is_visitor else ("EMPLOYEE" if is_employee else "PERSON")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label_role} {pconf:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            for hb in last_hands_roles:
                hx1, hy1, hx2, hy2, hconf, role = hb
                color = (255, 0, 0) if role == "visitor" else ((0, 0, 255) if role == "employee" else (255, 255, 0))
                label = "V_HAND" if role == "visitor" else ("E_HAND" if role == "employee" else "HAND")
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
                cv2.putText(frame, f"{label} {hconf:.2f}", (hx1, max(0, hy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            if args.show:
                msg = f"Line: {'segment' if p1 and p2 else f'x={line_x_current}'} | click set (2 points), 's' save, 'c' clear"
                cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("detections", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    if p1 is not None and p2 is not None:
                        write_zones(args.zones, employee_side=emp_side, p1=p1, p2=p2)
                    else:
                        write_zones(args.zones, line_x_current, emp_side)
                if key == ord("c"):
                    p1, p2 = None, None
            if writer is not None:
                writer.write(frame)
            continue
        world_results = world.predict(frame, conf=args.threshold, verbose=False)
        rw = world_results[0]
        wboxes = rw.boxes
        wxyxy = wboxes.xyxy.cpu().numpy() if wboxes is not None and wboxes.xyxy is not None else np.empty((0, 4))
        wconfs = wboxes.conf.cpu().numpy() if wboxes is not None and wboxes.conf is not None else np.empty((0,))
        wclss = wboxes.cls.cpu().numpy() if wboxes is not None and wboxes.cls is not None else np.empty((0,))
        wnames = rw.names if hasattr(rw, "names") and rw.names is not None else {}
        persons = []
        hands = []
        items = []
        for i in range(len(wxyxy)):
            name = get_class_name(wnames, wclss[i])
            x1, y1, x2, y2 = wxyxy[i].astype(int)
            if name == "person" and wconfs[i] >= args.threshold:
                persons.append((x1, y1, x2, y2, wconfs[i]))
            else:
                if wconfs[i] >= args.threshold:
                    items.append((name, x1, y1, x2, y2, wconfs[i]))
        if hand is not None:
            hres = hand.predict(frame, conf=args.hand_threshold, verbose=False)
            rh = hres[0]
            hboxes = rh.boxes
            hxyxy = hboxes.xyxy.cpu().numpy() if hboxes is not None and hboxes.xyxy is not None else np.empty((0, 4))
            hconfs = hboxes.conf.cpu().numpy() if hboxes is not None and hboxes.conf is not None else np.empty((0,))
            for i in range(len(hxyxy)):
                x1, y1, x2, y2 = hxyxy[i].astype(int)
                hands.append((x1, y1, x2, y2, hconfs[i]))
        visitor_box = None
        employee_box = None
        hand_roles_collected = []
        if len(persons) >= 1:
            persons_sorted = sorted(persons, key=lambda b: (b[0] + b[2]) / 2)
            if len(persons_sorted) > 1:
                if emp_side == "right":
                    employee_box = persons_sorted[-1]
                    visitor_box = persons_sorted[0]
                else:
                    employee_box = persons_sorted[0]
                    visitor_box = persons_sorted[-1]
            else:
                x1, y1, x2, y2, _ = persons_sorted[0]
                cx_person = (x1 + x2) / 2
                if emp_side == "right":
                    if cx_person > line_x_current:
                        employee_box = persons_sorted[0]
                    else:
                        visitor_box = persons_sorted[0]
                else:
                    if cx_person < line_x_current:
                        employee_box = persons_sorted[0]
                    else:
                        visitor_box = persons_sorted[0]
            if interaction_id is None and employee_box is not None and visitor_box is not None:
                interaction_id = start_interaction(conn, 1, 2)
        if visitor_box is not None or employee_box is not None:
            if visitor_box is not None and employee_box is not None:
                print(f"Frame {frame_idx}: Visitor detected | Employee detected | CNIC: {'YES' if cnic_detected_flag else 'NO'}")
            elif visitor_box is not None:
                print(f"Frame {frame_idx}: Visitor detected | CNIC: {'YES' if cnic_detected_flag else 'NO'}")
            else:
                print(f"Frame {frame_idx}: Employee detected | CNIC: {'YES' if cnic_detected_flag else 'NO'}")
        last_persons = persons[:]
        last_hands_roles = hand_roles_collected
        last_visitor_box = visitor_box
        last_employee_box = employee_box
        saved_visitor_this_frame = False
        hand_roles_collected = []
        for hb in hands:
            hx1, hy1, hx2, hy2, hconf = hb
            cx = (hx1 + hx2) / 2
            cy = (hy1 + hy2) / 2
            belongs_to_visitor = False
            belongs_to_employee = False
            if visitor_box is not None or employee_box is not None:
                vx, vy = None, None
                ex, ey = None, None
                if visitor_box is not None:
                    vx = (visitor_box[0] + visitor_box[2]) / 2
                    vy = (visitor_box[1] + visitor_box[3]) / 2
                if employee_box is not None:
                    ex = (employee_box[0] + employee_box[2]) / 2
                    ey = (employee_box[1] + employee_box[3]) / 2
                if vx is not None and ex is not None:
                    dv = (cx - vx) ** 2 + (cy - vy) ** 2
                    de = (cx - ex) ** 2 + (cy - ey) ** 2
                    belongs_to_visitor = dv <= de
                    belongs_to_employee = de < dv
                elif vx is not None:
                    belongs_to_visitor = True
                    belongs_to_employee = False
                else:
                    belongs_to_visitor = False
                    belongs_to_employee = True
            else:
                if p1 is not None and p2 is not None:
                    s = point_side((cx, cy), p1, p2)
                    belongs_to_visitor = (s > 0) if emp_side == "right" else (s < 0)
                    belongs_to_employee = not belongs_to_visitor
                else:
                    belongs_to_visitor = (cx < line_x_current) if emp_side == "right" else (cx > line_x_current)
                    belongs_to_employee = not belongs_to_visitor
            touches = rect_touches_line(hx1, hy1, hx2, hy2, p1, p2, line_x_current)
            role = "visitor" if belongs_to_visitor else ("employee" if belongs_to_employee else "unknown")
            if belongs_to_visitor and touches and not saved_visitor_this_frame:
                if interaction_id is None:
                    interaction_id = start_interaction(conn, 1, 2)
                frame_path = save_frame(frame, args.frames_dir, interaction_id)
                hand_path = crop_and_save(frame, (hx1, hy1, hx2, hy2), args.frames_dir, interaction_id, prefix="hand")
                zs = detect_zero_shot(world, frame[hy1:hy2, hx1:hx2], args.threshold)
                insert_event(conn, interaction_id, hand_path, zs, event_type="HAND_CROSSING_VISITOR")
                if zs == "cnic":
                    cnic_detected_flag = True
                    cnic_image_path = hand_path
                    print(f"Frame {frame_idx}: CNIC detected in visitor hand")
                saved_visitor_this_frame = True
            if role == "visitor":
                color = (255, 0, 0)
                label = "V_HAND"
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
                cv2.putText(frame, f"{label} {hconf:.2f}", (hx1, max(0, hy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                hand_roles_collected.append((hx1, hy1, hx2, hy2, hconf, role))
        for pb in persons:
            x1, y1, x2, y2, pconf = pb
            is_visitor = visitor_box is not None and (x1, y1, x2, y2, pconf) == visitor_box
            is_employee = employee_box is not None and (x1, y1, x2, y2, pconf) == employee_box
            color = (0, 255, 0) if is_visitor else ((0, 0, 255) if is_employee else (0, 255, 0))
            label_role = ("CUSTOMER" if cnic_detected_flag else "VISITOR") if is_visitor else ("EMPLOYEE" if is_employee else "PERSON")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_role} {pconf:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if p1 is not None and p2 is not None:
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
        else:
            cv2.line(frame, (line_x_current, 0), (line_x_current, frame.shape[0]), (0, 255, 255), 2)
        if args.show:
            msg = f"Line: {'segment' if p1 and p2 else f'x={line_x_current}'} | click set (2 points), 's' save, 'c' clear"
            if line_dirty:
                msg += " *"
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        if writer is not None:
            writer.write(frame)
        if args.show:
            cv2.imshow("detections", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if p1 is not None and p2 is not None:
                    write_zones(args.zones, employee_side=emp_side, p1=p1, p2=p2)
                else:
                    write_zones(args.zones, line_x_current, emp_side)
                line_dirty = False
            if key == ord("c"):
                p1, p2 = None, None
                line_dirty = True
    if use_reader and reader is not None:
        try:
            reader.close()
        except Exception:
            pass
    else:
        cap.release()
    if writer is not None:
        writer.release()
    if log_handle is not None:
        log_handle.close()
    if interaction_id is not None:
        end_interaction(conn, interaction_id, cnic_detected_flag, "CLIENT" if cnic_detected_flag else "VISITOR", cnic_image_path)
    conn.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()