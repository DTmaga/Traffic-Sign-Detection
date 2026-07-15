import cv2
import numpy as np

# ===== CONFIG =====
VIDEO_LIST = ["video1.mp4", "video2.mp4"]
ESC_KEY = ord('q')
FRAME_WIDTH, FRAME_HEIGHT = 640, 360
MIN_CONTOUR_AREA, MAX_CONTOUR_AREA = 100, 4000
COLOR_FILL_RATIO = 0.4

student_ids = [ 
    "Student ID: 523H0191",
    "            523H0166",
    "            524H0187"
]

# Color HSV
lower_red1, upper_red1 = np.array([0,60,40]), np.array([10,255,255])
lower_red2, upper_red2 = np.array([160,60,40]), np.array([180,255,255])
lower_yellow, upper_yellow = np.array([15,70,70]), np.array([35,255,255])
lower_blue, upper_blue = np.array([80,160,60]), np.array([125,255,255])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Tracking
ALPHA_BOX = 0.4
HIT_THRESHOLD, MISS_THRESHOLD = 3, 4
tracked_objects = {}
next_obj_id = 0
frame_counter = 0

# ===== HELPERS =====
def enhance_frame(frame):
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
    lab = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = cv2.add(hsv[:,:,1], 40)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    enhanced = cv2.GaussianBlur(enhanced, (5,5), 0)
    kernel_sharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return cv2.filter2D(enhanced, -1, kernel_sharp)

def get_color_masks(frame):
    MASK_BOTTOM_RATIO = 0.4
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    start_row = int(FRAME_HEIGHT * MASK_BOTTOM_RATIO)
    
    mask_r = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_r[start_row:, :] = 0

    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_y[start_row:, :] = 0

    mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_b[start_row:, :] = 0

    for m in [mask_r, mask_y, mask_b]:
        m[:] = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m[:] = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close)
    return mask_r, mask_y, mask_b

def calculate_iou(box1, box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    inter_x = max(0, min(x1+w1, x2+w2) - max(x1,x2))
    inter_y = max(0, min(y1+h1, y2+h2) - max(y1,y2))
    inter_area = inter_x*inter_y
    union_area = w1*h1 + w2*h2 - inter_area
    return inter_area/union_area if union_area>0 else 0

def detect_shapes(mask, color_label):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area<MIN_CONTOUR_AREA or area>MAX_CONTOUR_AREA: continue
        x,y,w,h = cv2.boundingRect(cnt)
        roi = mask[y:y+h, x:x+w]
        if roi.size==0 or cv2.countNonZero(roi)/roi.size<COLOR_FILL_RATIO: continue
        epsilon = 0.03*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        circularity = 4*np.pi*area/(cv2.arcLength(cnt,True)**2) if cv2.arcLength(cnt,True)>0 else 0
        aspect_ratio = w/h if h>0 else 0
        if len(approx)==4 and (aspect_ratio>3.0 or aspect_ratio<0.3): continue

        shape="Unknown"
        if len(approx)==3:
            pts = approx
            a = np.linalg.norm(pts[0][0]-pts[1][0])
            b = np.linalg.norm(pts[1][0]-pts[2][0])
            c = np.linalg.norm(pts[2][0]-pts[0][0])
            sides = sorted([a,b,c])
            ratio = sides[0]/sides[2]
            if ratio < 0.5: continue
            shape="Triangle"
        elif len(approx)==4 and area/(w*h)>0.7:
            shape="Rectangle"
        elif circularity>0.7:
            shape="Circle"

        sign_type="Khac"
        if color_label=="Red" and shape=="Circle": sign_type="Bien bao cam"
        elif color_label=="Yellow" and shape=="Triangle": sign_type="Bien bao nguy hiem"
        elif color_label=="Blue" and shape in ["Rectangle"]: sign_type="Bien bao chi dan"
        elif color_label=="Blue" and shape=="Circle": sign_type="Bien bao hieu lenh"

        if sign_type=="Khac": continue

        detected.append({"box":(x,y,w,h), "label":sign_type, "score":cv2.countNonZero(roi)/roi.size})
    return detected

def update_tracks(detected, tracked_objects, next_obj_id, frame_counter):
    matched_detections = set()
    matched_tracks = set()
    for i, d in enumerate(detected):
        best_iou, best_id = 0, None
        for track_id, track in tracked_objects.items():
            if track["label"] != d["label"]: continue
            iou = calculate_iou(d["box"], track["box"])
            if iou > best_iou and iou > 0.3:
                best_iou, best_id = iou, track_id
        if best_id is not None:
            x, y, w, h = d["box"]
            x_old, y_old, w_old, h_old = tracked_objects[best_id]["smoothed_box"]
            tracked_objects[best_id].update({
                "box": d["box"],
                "smoothed_box": (
                    int(ALPHA_BOX*x + (1-ALPHA_BOX)*x_old),
                    int(ALPHA_BOX*y + (1-ALPHA_BOX)*y_old),
                    int(ALPHA_BOX*w + (1-ALPHA_BOX)*w_old),
                    int(ALPHA_BOX*h + (1-ALPHA_BOX)*h_old)
                ),
                "score": d["score"],
                "hits": tracked_objects[best_id]["hits"] + 1,
                "misses": 0,
                "last_seen": frame_counter
            })
            matched_detections.add(i)
            matched_tracks.add(best_id)
    for track_id in list(tracked_objects.keys()):
        if track_id not in matched_tracks:
            tracked_objects[track_id]["misses"] = frame_counter - tracked_objects[track_id]["last_seen"]
            if tracked_objects[track_id]["misses"] > MISS_THRESHOLD:
                del tracked_objects[track_id]
    for i, d in enumerate(detected):
        if i not in matched_detections:
            tracked_objects[next_obj_id] = {
                "box": d["box"],
                "smoothed_box": d["box"],
                "label": d["label"],
                "score": d["score"],
                "hits": 1,
                "misses": 0,
                "last_seen": frame_counter
            }
            next_obj_id += 1
    return tracked_objects, next_obj_id

def get_confirmed(tracked_objects):
    return [ {"box":t["smoothed_box"], "label":t["label"], "score":t["score"]} 
             for t in tracked_objects.values() if t["hits"]>=HIT_THRESHOLD]

def draw_boxes(frame, detected_list):
    for d in detected_list:
        x,y,w,h=d["box"]; label=d["label"]
        color=(0,255,0)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

def draw_student_ids(frame, ids):
    y0, dy = 20, 20
    for i, text in enumerate(ids):
        y = y0 + i*dy
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ===== MAIN LOOP =====
for VIDEO_PATH in VIDEO_LIST:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): 
        print(f"Cannot open {VIDEO_PATH}")
        continue
    print(f"Processing {VIDEO_PATH}, press 'q' to quit")

    # --- Video Writer ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    out_filename = f"{VIDEO_PATH.split('.')[0]}_523H0191_523H0166_524H0187.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_filename, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret: break
        enhanced = enhance_frame(frame)
        mask_r, mask_y, mask_b = get_color_masks(enhanced)

        detected = detect_shapes(mask_r,"Red")
        detected += detect_shapes(mask_y,"Yellow")
        detected += detect_shapes(mask_b,"Blue")

        tracked_objects, next_obj_id = update_tracks(detected, tracked_objects, next_obj_id, frame_counter)
        confirmed = get_confirmed(tracked_objects)
        draw_boxes(enhanced, confirmed)
        draw_student_ids(enhanced, student_ids)

        cv2.imshow("Result", enhanced)
        out.write(enhanced)

        if cv2.waitKey(1) & 0xFF == ESC_KEY: break
        frame_counter +=1
    cap.release()
    out.release()

cv2.destroyAllWindows()
