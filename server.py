from fastapi import FastAPI
from fastapi import BackgroundTasks
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import time
from pydantic import BaseModel
import uuid
import threading
from pathlib import Path
import cv2
import numpy as np
try:
    import imageio
except Exception:
    imageio = None
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import json
import person_detection as pd

app = FastAPI()

class ProcessRequest(BaseModel):
    source: str = "dataset"
    zones: str = "zones.json"
    employee_side: str = "right"
    frame_stride: int = 2
    threshold: float = 0.5
    hand_threshold: float = 0.4
    world_model: str = "yoloworld.pt"
    hand_model: str = "hand_yolov8.pt"
    db_path: str = "counter.db"
    frames_dir: str = "data/frames"
    calibration_dir: str = "data/calibration"
    force_imageio: bool = False

runs = {}

def open_reader(path):
    if imageio is None:
        return None
    try:
        return imageio.get_reader(path)
    except Exception:
        return None

def start_detection(req: ProcessRequest, run_id: str):
    src = pd.resolve_source(req.source, req.source if req.source != "dataset" else "dataset")
    cap = pd.open_source(src)
    use_reader = False
    reader = None
    if (req.force_imageio and not src.isdigit()) or (not cap.isOpened() and (not src.isdigit())):
        reader = open_reader(src)
        use_reader = reader is not None
        if not use_reader:
            runs[run_id] = {"status": "error", "message": "Failed to open source"}
            return
        try:
            meta = reader.get_meta_data()
            fps = float(meta.get("fps", 30.0))
        except Exception:
            fps = 30.0
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
    world_path = Path(req.world_model)
    if not world_path.exists():
        world = YOLO("yolov8s-world.pt")
    else:
        world = YOLO(str(world_path))
    try:
        world.set_classes(["person", "identity card", "id card", "national id card", "plastic card", "cnic"])
    except Exception:
        pass
    hand_path = Path(req.hand_model)
    hand = YOLO(str(hand_path)) if hand_path.exists() else None
    conn = pd.init_db(req.db_path)
    pd.ensure_dirs(req.frames_dir)
    if use_reader:
        pd.save_sample_frame_reader(reader, req.calibration_dir)
    else:
        pd.save_sample_frame(cap, req.calibration_dir)
    p1 = None
    p2 = None
    emp_zone_rect = None
    vis_zone_rect = None
    cust_zone_rect = None
    emp_side = req.employee_side
    line_x_current = int(640 * 0.55)
    frame_idx = 0
    interaction_id = None
    cnic_detected_flag = False
    cnic_image_path = ""
    last_persons = []
    last_hands_roles = []
    last_visitor_box = None
    last_employee_box = None
    runs[run_id] = {"status": "running", "frames": 0, "customers": 0, "last_frame": b"", "last_customer_frame": -9999, "customer_tracks": []}
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
        frame = cv2.resize(frame, (640, 640))
        emp_zone_rect = None
        vis_zone_rect = None
        p1 = None
        p2 = None
        zp = Path(req.zones)
        if zp.exists():
            try:
                with open(zp, "r", encoding="utf-8") as f:
                    zdata = json.load(f)
                emp_side = str(zdata.get("employee_side", emp_side))
                if "p1" in zdata and "p2" in zdata:
                    p1 = (int(zdata["p1"][0]), int(zdata["p1"][1]))
                    p2 = (int(zdata["p2"][0]), int(zdata["p2"][1]))
                else:
                    lx = zdata.get("line_x")
                    if lx is not None:
                        line_x_current = int(lx)
                ez = zdata.get("employee_zone")
                if ez and len(ez) == 4:
                    emp_zone_rect = (int(ez[0]), int(ez[1]), int(ez[2]), int(ez[3]))
                vz = zdata.get("visitor_zone")
                if vz and len(vz) == 4:
                    vis_zone_rect = (int(vz[0]), int(vz[1]), int(vz[2]), int(vz[3]))
            except Exception:
                pass
        if req.frame_stride > 1 and (frame_idx % req.frame_stride != 0):
            runs[run_id]["frames"] = frame_idx
            continue
        world_results = world.predict(frame, conf=req.threshold, verbose=False)
        rw = world_results[0]
        wboxes = rw.boxes
        wxyxy = wboxes.xyxy.cpu().numpy() if wboxes is not None and wboxes.xyxy is not None else np.empty((0, 4))
        wconfs = wboxes.conf.cpu().numpy() if wboxes is not None and wboxes.conf is not None else np.empty((0,))
        wclss = wboxes.cls.cpu().numpy() if wboxes is not None and wboxes.cls is not None else np.empty((0,))
        wnames = rw.names if hasattr(rw, "names") and rw.names is not None else {}
        persons = []
        hands = []
        for i in range(len(wxyxy)):
            name = pd.get_class_name(wnames, wclss[i])
            x1, y1, x2, y2 = wxyxy[i].astype(int)
            if name == "person" and wconfs[i] >= req.threshold:
                persons.append((x1, y1, x2, y2, wconfs[i]))
        if hand is not None:
            hres = hand.predict(frame, conf=req.hand_threshold, verbose=False)
            rh = hres[0]
            hboxes = rh.boxes
            hxyxy = hboxes.xyxy.cpu().numpy() if hboxes is not None and hboxes.xyxy is not None else np.empty((0, 4))
            hconfs = hboxes.conf.cpu().numpy() if hboxes is not None and hboxes.conf is not None else np.empty((0,))
            for i in range(len(hxyxy)):
                x1, y1, x2, y2 = hxyxy[i].astype(int)
                hands.append((x1, y1, x2, y2, hconfs[i]))
        visitor_boxes = []
        employee_boxes = []
        customer_boxes = []
        if len(persons) >= 1:
            persons_sorted = sorted(persons, key=lambda b: (b[0] + b[2]) / 2)
            for pb in persons_sorted:
                x1, y1, x2, y2, _ = pb
                cxp = (x1 + x2) / 2
                cyp = (y1 + y2) / 2
                in_emp = emp_zone_rect is not None and (cxp >= emp_zone_rect[0] and cxp <= emp_zone_rect[2] and cyp >= emp_zone_rect[1] and cyp <= emp_zone_rect[3])
                in_vis = vis_zone_rect is not None and (cxp >= vis_zone_rect[0] and cxp <= vis_zone_rect[2] and cyp >= vis_zone_rect[1] and cyp <= vis_zone_rect[3])
                if in_emp:
                    employee_boxes.append(pb)
                elif in_vis:
                    visitor_boxes.append(pb)
            tracks = runs[run_id].get("customer_tracks", [])
            new_tracks = []
            used_indices = set()
            for t in tracks:
                if len(t) >= 9:
                    tcx, tcy, tw, th, tlast, tx1, ty1, tx2, ty2 = t
                else:
                    tcx, tcy, tw, th, tlast = t
                    tx1 = int(tcx - tw * 0.5)
                    ty1 = int(tcy - th * 0.5)
                    tx2 = int(tcx + tw * 0.5)
                    ty2 = int(tcy + th * 0.5)
                best_i = -1
                best_iou = 0.0
                for i, vb in enumerate(visitor_boxes):
                    vx1, vy1, vx2, vy2, _ = vb
                    ix1 = max(tx1, vx1)
                    iy1 = max(ty1, vy1)
                    ix2 = min(tx2, vx2)
                    iy2 = min(ty2, vy2)
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    a = max(1, (tx2 - tx1) * (ty2 - ty1))
                    b = max(1, (vx2 - vx1) * (vy2 - vy1))
                    denom = a + b - inter + 1e-6
                    iou = inter / denom if denom > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_i = i
                matched = False
                if best_i >= 0 and best_iou >= 0.2:
                    vx1, vy1, vx2, vy2, _ = visitor_boxes[best_i]
                    vcx = (vx1 + vx2) / 2
                    vcy = (vy1 + vy2) / 2
                    inside_vis = vis_zone_rect is not None and (vcx >= vis_zone_rect[0] and vcx <= vis_zone_rect[2] and vcy >= vis_zone_rect[1] and vcy <= vis_zone_rect[3])
                    if inside_vis:
                        customer_boxes.append(visitor_boxes[best_i])
                        used_indices.add(best_i)
                        new_tracks.append([vcx, vcy, vx2 - vx1, vy2 - vy1, frame_idx, vx1, vy1, vx2, vy2])
                        matched = True
                if not matched:
                    if frame_idx - tlast <= 30:
                        new_tracks.append([tcx, tcy, tw, th, tlast, tx1, ty1, tx2, ty2])
            visitor_boxes = [vb for i, vb in enumerate(visitor_boxes) if i not in used_indices]
            runs[run_id]["customer_tracks"] = new_tracks
            if interaction_id is None and employee_boxes and visitor_boxes:
                interaction_id = pd.start_interaction(conn, 1, 2)
        for hb in hands:
            hx1, hy1, hx2, hy2, hconf = hb
            cx = (hx1 + hx2) / 2
            cy = (hy1 + hy2) / 2
            belongs_to_visitor = False
            belongs_to_employee = False
            if visitor_boxes or employee_boxes:
                dv = None
                de = None
                if visitor_boxes:
                    vmin = 1e18
                    for vb in visitor_boxes:
                        vx = (vb[0] + vb[2]) / 2
                        vy = (vb[1] + vb[3]) / 2
                        d = (cx - vx) ** 2 + (cy - vy) ** 2
                        if d < vmin:
                            vmin = d
                    dv = vmin
                if employee_boxes:
                    emin = 1e18
                    for eb in employee_boxes:
                        ex = (eb[0] + eb[2]) / 2
                        ey = (eb[1] + eb[3]) / 2
                        d = (cx - ex) ** 2 + (cy - ey) ** 2
                        if d < emin:
                            emin = d
                    de = emin
                if dv is not None and de is not None:
                    belongs_to_visitor = dv <= de
                    belongs_to_employee = de < dv
                elif dv is not None:
                    belongs_to_visitor = True
                elif de is not None:
                    belongs_to_employee = True
            else:
                belongs_to_visitor = False
                belongs_to_employee = False
            for t in runs[run_id].get("customer_tracks", []):
                tcx = t[0]
                tcy = t[1]
                tw = t[2]
                th = t[3]
                rad2 = ((tw + th) * 0.5) ** 2
                d2 = (cx - tcx) ** 2 + (cy - tcy) ** 2
                if d2 <= rad2:
                    belongs_to_visitor = False
                    belongs_to_employee = False
                    touches = False
                    break
            touches = pd.rect_touches_line(hx1, hy1, hx2, hy2, p1, p2, line_x_current)
            if belongs_to_visitor and touches:
                if interaction_id is None:
                    interaction_id = pd.start_interaction(conn, 1, 2)
                pd.save_frame(frame, req.frames_dir, interaction_id)
                hand_path = pd.crop_and_save(frame, (hx1, hy1, hx2, hy2), req.frames_dir, interaction_id, prefix="hand")
                crop = frame[hy1:hy2, hx1:hx2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                var = cv2.Laplacian(gray, cv2.CV_64F).var()
                not_empty = var > 40.0
                pd.insert_event(conn, interaction_id, hand_path, "non_empty" if not_empty else "empty", event_type="HAND_CROSSING_VISITOR")
                if not_empty and (frame_idx - runs[run_id]["last_customer_frame"] > 10):
                    nearest = None
                    bestd = 1e18
                    for vb in visitor_boxes:
                        vx = (vb[0] + vb[2]) / 2
                        vy = (vb[1] + vb[3]) / 2
                        d = (vx - cx) ** 2 + (vy - cy) ** 2
                        if d < bestd:
                            bestd = d
                            nearest = vb
                    if nearest is not None:
                        vx1, vy1, vx2, vy2, _ = nearest
                        vcx = (vx1 + vx2) / 2
                        vcy = (vy1 + vy2) / 2
                        exists = False
                        for t in runs[run_id].get("customer_tracks", []):
                            tcx = t[0]
                            tcy = t[1]
                            tw = t[2]
                            th = t[3]
                            if (vcx - tcx) ** 2 + (vcy - tcy) ** 2 <= ((tw + th) * 0.5) ** 2:
                                exists = True
                                break
                        if not exists:
                            runs[run_id]["customer_tracks"].append([vcx, vcy, vx2 - vx1, vy2 - vy1, frame_idx, vx1, vy1, vx2, vy2])
                            runs[run_id]["customers"] = runs[run_id].get("customers", 0) + 1
                            runs[run_id]["last_customer_frame"] = frame_idx
            if belongs_to_visitor:
                color = (255, 0, 0)
                label = "V_HAND"
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
                cv2.putText(frame, f"{label} {hconf:.2f}", (hx1, max(0, hy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        for pb in persons:
            x1, y1, x2, y2, pconf = pb
            is_customer = any((x1, y1, x2, y2, pconf) == cb for cb in customer_boxes)
            is_employee = any((x1, y1, x2, y2, pconf) == eb for eb in employee_boxes)
            is_visitor = any((x1, y1, x2, y2, pconf) == vb for vb in visitor_boxes)
            if not (is_customer or is_employee or is_visitor):
                continue
            color = (0, 255, 0) if is_visitor else ((0, 0, 255) if is_employee else (255, 255, 0))
            label_role = "VISITOR" if is_visitor else ("EMPLOYEE" if is_employee else "CUSTOMER")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_role} {pconf:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if emp_zone_rect is not None:
            cv2.rectangle(frame, (emp_zone_rect[0], emp_zone_rect[1]), (emp_zone_rect[2], emp_zone_rect[3]), (0,0,255), 2)
        if vis_zone_rect is not None:
            cv2.rectangle(frame, (vis_zone_rect[0], vis_zone_rect[1]), (vis_zone_rect[2], vis_zone_rect[3]), (0,255,0), 2)
        # draw customer tracks as small markers (optional)
        cv2.putText(frame, f"CUSTOMERS: {runs[run_id].get('customers',0)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
        if p1 is not None and p2 is not None:
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
        else:
            cv2.line(frame, (line_x_current, 0), (line_x_current, frame.shape[0]), (0, 255, 255), 2)
        ok2, buf = cv2.imencode('.jpg', frame)
        if ok2:
            runs[run_id]["last_frame"] = buf.tobytes()
        runs[run_id]["frames"] = frame_idx
    if use_reader and reader is not None:
        try:
            reader.close()
        except Exception:
            pass
    else:
        cap.release()
    if interaction_id is not None:
        pd.end_interaction(conn, interaction_id, False, "CLIENT" if runs[run_id].get("customers",0)>0 else "VISITOR", "")
    conn.close()
    runs[run_id]["status"] = "done"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process")
def process(req: ProcessRequest, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    runs[run_id] = {"status": "starting"}
    background_tasks.add_task(start_detection, req, run_id)
    return {"run_id": run_id}

@app.get("/process/{run_id}")
def process_status(run_id: str):
    info = runs.get(run_id)
    if info is None:
        return {"status": "unknown"}
    return {
        "status": info.get("status"),
        "frames": info.get("frames", 0),
        "customers": info.get("customers", 0)
    }

@app.get("/zones")
def get_zones(path: str = "zones.json"):
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"exists": True, "zones": data}

class ZonesRequest(BaseModel):
    employee_side: str = "right"
    line_x: int | None = None
    p1: tuple[int, int] | None = None
    p2: tuple[int, int] | None = None
    path: str = "zones.json"
    employee_zone: list[int] | None = None
    visitor_zone: list[int] | None = None

@app.post("/zones")
def set_zones(req: ZonesRequest):
    p = Path(req.path)
    data = {}
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    if req.p1 is not None and req.p2 is not None:
        data["p1"] = [int(req.p1[0]), int(req.p1[1])]
        data["p2"] = [int(req.p2[0]), int(req.p2[1])]
    if req.line_x is not None:
        data["line_x"] = int(req.line_x)
    data["employee_side"] = str(req.employee_side)
    if req.employee_zone is not None and len(req.employee_zone) == 4:
        ez = req.employee_zone
        data["employee_zone"] = [int(ez[0]), int(ez[1]), int(ez[2]), int(ez[3])]
    if req.visitor_zone is not None and len(req.visitor_zone) == 4:
        vz = req.visitor_zone
        data["visitor_zone"] = [int(vz[0]), int(vz[1]), int(vz[2]), int(vz[3])]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return {"ok": True, "zones": data}

class ZonesClearRequest(BaseModel):
    path: str = "zones.json"
    clear: str = "all"  # one of: all, line, emp, vis

@app.post("/zones/clear")
def clear_zones(req: ZonesClearRequest):
    p = Path(req.path)
    data = {}
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    c = req.clear.lower()
    if c in ("all", "line"):
        data.pop("p1", None)
        data.pop("p2", None)
        data.pop("line_x", None)
    if c in ("all", "emp"):
        data.pop("employee_zone", None)
    if c in ("all", "vis"):
        data.pop("visitor_zone", None)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return {"ok": True, "zones": data}

@app.get("/interactions")
def interactions(db_path: str = "counter.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT interaction_id, timestamp_start, timestamp_end, cnic_detected, classification, cnic_image_path FROM interactions ORDER BY interaction_id DESC")
    rows = cur.fetchall()
    conn.close()
    return {"count": len(rows), "rows": [
        {
            "interaction_id": r[0],
            "timestamp_start": r[1],
            "timestamp_end": r[2],
            "cnic_detected": bool(r[3]),
            "classification": r[4],
            "cnic_image_path": r[5]
        } for r in rows
    ]}

@app.get("/events")
def events(db_path: str = "counter.db", interaction_id: int | None = None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if interaction_id is None:
        cur.execute("SELECT event_id, interaction_id, event_type, image_path, zero_shot_result, timestamp FROM events ORDER BY event_id DESC")
    else:
        cur.execute("SELECT event_id, interaction_id, event_type, image_path, zero_shot_result, timestamp FROM events WHERE interaction_id=? ORDER BY event_id DESC", (interaction_id,))
    rows = cur.fetchall()
    conn.close()
    return {"count": len(rows), "rows": [
        {
            "event_id": r[0],
            "interaction_id": r[1],
            "event_type": r[2],
            "image_path": r[3],
            "zero_shot_result": r[4],
            "timestamp": r[5]
        } for r in rows
    ]}

def _mjpeg_gen(run_id: str):
    boundary = b"--frame"
    while True:
        info = runs.get(run_id)
        if info is None:
            break
        frame_bytes = info.get("last_frame", b"")
        if frame_bytes:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        if info.get("status") == "done":
            break
        time.sleep(0.05)

@app.get("/stream/{run_id}")
def stream(run_id: str):
    return StreamingResponse(_mjpeg_gen(run_id), media_type="multipart/x-mixed-replace; boundary=frame")

app.mount("/ui", StaticFiles(directory="web", html=True), name="web")

@app.get("/")
def root():
    return RedirectResponse(url="/ui")