from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import requests
import json
from io import BytesIO
from PIL import Image

app = Flask(__name__, static_folder='static')
CORS(app)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# ─── Helpers ────────────────────────────────────────────────────────────────

def b64_to_cv2(b64_str):
    """Decode base64 image to OpenCV BGR array."""
    data = base64.b64decode(b64_str)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def cv2_to_b64(img):
    """Encode OpenCV image to base64 PNG."""
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')

def pil_to_b64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def resize_keep_aspect(img, max_dim=2000):
    h, w = img.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

# ─── Step 1: Align print to AW using feature matching ───────────────────────

def align_images(aw_img, print_img):
    """
    Align print_img to aw_img using ORB feature matching + homography.
    Returns aligned print image (same size as aw_img).
    """
    aw_gray = cv2.cvtColor(aw_img, cv2.COLOR_BGR2GRAY)
    pr_gray = cv2.cvtColor(print_img, cv2.COLOR_BGR2GRAY)

    # Try SIFT first, fall back to ORB
    try:
        detector = cv2.SIFT_create(nfeatures=2000)
        norm = cv2.NORM_L2
    except Exception:
        detector = cv2.ORB_create(nfeatures=2000)
        norm = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(aw_gray, None)
    kp2, des2 = detector.detectAndCompute(pr_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        # Can't align — return resized print
        h, w = aw_img.shape[:2]
        return cv2.resize(print_img, (w, h))

    bf = cv2.BFMatcher(norm, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
        # Lowe's ratio test
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
    except Exception:
        good = []

    if len(good) < 8:
        h, w = aw_img.shape[:2]
        return cv2.resize(print_img, (w, h))

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        h, w = aw_img.shape[:2]
        return cv2.resize(print_img, (w, h))

    h, w = aw_img.shape[:2]
    aligned = cv2.warpPerspective(print_img, H, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    return aligned

# ─── Step 2: Pixel-level diff and defect detection ──────────────────────────

def detect_defects(aw_img, aligned_print, sensitivity=35):
    """
    Compare AW and aligned print pixel-by-pixel.
    Returns list of defect regions with bounding boxes and types.
    """
    # Convert to LAB color space (better perceptual diff)
    aw_lab  = cv2.cvtColor(aw_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    pr_lab  = cv2.cvtColor(aligned_print, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Weighted diff (L channel more important)
    diff_L = np.abs(aw_lab[:,:,0] - pr_lab[:,:,0]) * 1.5
    diff_A = np.abs(aw_lab[:,:,1] - pr_lab[:,:,1])
    diff_B = np.abs(aw_lab[:,:,2] - pr_lab[:,:,2])
    diff   = (diff_L + diff_A + diff_B) / 3.0

    # Threshold
    thresh = (diff > sensitivity).astype(np.uint8) * 255

    # Remove noise with morphological ops
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel_open)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = aw_img.shape[:2]
    total_area = h * w
    defects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:  # ignore tiny noise
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Classify defect type by shape
        defect_type = classify_defect(cnt, area, bw, bh, total_area, aw_img, aligned_print)
        if defect_type is None:
            continue

        # Compute mean diff intensity in region
        mask_roi = np.zeros(thresh.shape, np.uint8)
        cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
        mean_diff = float(cv2.mean(diff, mask=mask_roi)[0])

        # Severity
        if mean_diff > 80 or area > total_area * 0.005:
            severity = 'high'
        elif mean_diff > 50 or area > total_area * 0.001:
            severity = 'medium'
        else:
            severity = 'low'

        defects.append({
            'type': defect_type['type'],
            'label': defect_type['label'],
            'x': int(x),
            'y': int(y),
            'w': int(bw),
            'h': int(bh),
            'area': int(area),
            'severity': severity,
            'mean_diff': round(mean_diff, 1),
            'contour': cnt
        })

    return defects, thresh

def classify_defect(cnt, area, bw, bh, total_area, aw_img, pr_img):
    """Classify defect type based on shape, size and context."""
    # Aspect ratio
    aspect = bw / bh if bh > 0 else 1.0
    # Circularity
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    # Relative size
    rel_size = area / total_area

    # Hickey: small circular ring-like defect
    if circularity > 0.6 and area < total_area * 0.002 and bw < 60 and bh < 60:
        return {'type': 'hickey', 'label': 'Hickey'}

    # Spot / ink spot: small roughly circular blob
    if circularity > 0.5 and area < total_area * 0.005 and max(bw, bh) < 80:
        return {'type': 'spot', 'label': 'Spot / đốm mực'}

    # Streak / vệt: very elongated
    if aspect > 5 or aspect < 0.2:
        return {'type': 'streak', 'label': 'Vệt / sọc'}

    # Scratch / xước: thin elongated
    if (aspect > 3 or aspect < 0.33) and area < total_area * 0.01:
        return {'type': 'scratch', 'label': 'Xước / trầy'}

    # Missing ink / thiếu nét: large area lighter than AW
    x, y, w2, h2 = cv2.boundingRect(cnt)
    aw_roi  = aw_img[y:y+h2, x:x+w2]
    pr_roi  = pr_img[y:y+h2, x:x+w2]
    aw_mean = float(np.mean(cv2.cvtColor(aw_roi, cv2.COLOR_BGR2GRAY))) if aw_roi.size > 0 else 128
    pr_mean = float(np.mean(cv2.cvtColor(pr_roi, cv2.COLOR_BGR2GRAY))) if pr_roi.size > 0 else 128

    if pr_mean > aw_mean + 20:
        if rel_size > 0.002:
            return {'type': 'missing_ink', 'label': 'Thiếu nét / mất mực'}
        else:
            return {'type': 'mat_net', 'label': 'Mất nét chữ'}

    # Extra ink / thừa mực
    if pr_mean < aw_mean - 20:
        return {'type': 'extra_ink', 'label': 'Thừa mực / lem mực'}

    # Blur / nhòe
    aw_lap  = cv2.Laplacian(cv2.cvtColor(aw_roi,  cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() if aw_roi.size > 0 else 100
    pr_lap  = cv2.Laplacian(cv2.cvtColor(pr_roi,  cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() if pr_roi.size > 0 else 100
    if pr_lap < aw_lap * 0.5:
        return {'type': 'blur', 'label': 'Nhòe / mờ chữ'}

    # General anomaly
    if rel_size < 0.0001:
        return None  # too small, skip
    return {'type': 'anomaly', 'label': 'Bất thường'}

# ─── Step 3: Draw results on image ──────────────────────────────────────────

DEFECT_COLORS = {
    'hickey':       (0,   100, 255),
    'spot':         (0,   200, 255),
    'streak':       (255, 100,   0),
    'scratch':      (180,   0, 255),
    'missing_ink':  (0,   255, 150),
    'mat_net':      (0,   255,  80),
    'extra_ink':    (255,  50,  50),
    'blur':         (255, 200,   0),
    'anomaly':      (200, 200, 200),
}

def draw_defects(img, defects):
    out = img.copy()
    for i, d in enumerate(defects):
        color = DEFECT_COLORS.get(d['type'], (200, 200, 200))
        x, y, w, h = d['x'], d['y'], d['w'], d['h']

        # Filled semi-transparent overlay
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

        # Border
        thick = 3 if d['severity'] == 'high' else 2
        cv2.rectangle(out, (x, y), (x+w, y+h), color, thick)

        # Label badge
        label = f"#{i+1} {d['label']}"
        font_scale = 0.45
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        by = max(y - 4, th + 4)
        cv2.rectangle(out, (x, by - th - 4), (x + tw + 6, by + 2), color, -1)
        cv2.putText(out, label, (x + 3, by - 2), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    return out

# ─── Step 4: Gemini description ─────────────────────────────────────────────

def describe_defects_with_gemini(defects, aw_b64, result_b64):
    if not GEMINI_API_KEY or not defects:
        return None

    summary_list = "\n".join([
        f"- Lỗi #{i+1}: {d['label']} (mức độ: {d['severity']}, diện tích: {d['area']}px²)"
        for i, d in enumerate(defects)
    ])

    prompt = f"""Bạn là chuyên gia QC ngành in ấn bao bì.
Tôi đã dùng Computer Vision phát hiện được {len(defects)} bất thường khi so sánh Artwork gốc với tờ in thực tế.

Danh sách lỗi phát hiện:
{summary_list}

Hãy viết báo cáo QC ngắn gọn bằng tiếng Việt:
1. Đánh giá tổng thể tờ in
2. Lỗi nghiêm trọng nhất cần xử lý ngay
3. Khuyến nghị hành động (in lại / có thể chấp nhận / cần kiểm tra thêm)

Trả lời ngắn gọn, không quá 150 từ."""

    try:
        body = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": aw_b64}},
                {"inline_data": {"mime_type": "image/png",  "data": result_b64}},
                {"text": prompt}
            ]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400}
        }
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json=body, timeout=30
        )
        data = r.json()
        return data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
    except Exception as e:
        return f"(Không lấy được nhận xét AI: {e})"

# ─── Main API endpoint ───────────────────────────────────────────────────────

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400

    aw_b64    = data.get('awImage')
    print_b64 = data.get('printImage')
    sensitivity = int(data.get('sensitivity', 35))

    if not aw_b64 or not print_b64:
        return jsonify({'error': 'Thiếu ảnh'}), 400

    try:
        # Decode images
        aw_img    = b64_to_cv2(aw_b64)
        print_img = b64_to_cv2(print_b64)

        if aw_img is None or print_img is None:
            return jsonify({'error': 'Không đọc được ảnh'}), 400

        # Resize to max 2000px
        aw_img    = resize_keep_aspect(aw_img,    2000)
        print_img = resize_keep_aspect(print_img, 2000)

        # Step 1: Align
        aligned = align_images(aw_img, print_img)

        # Step 2: Detect defects
        defects, diff_map = detect_defects(aw_img, aligned, sensitivity)

        # Step 3: Draw results
        result_img = draw_defects(aligned, defects)

        # Step 4: Encode outputs
        result_b64  = cv2_to_b64(result_img)
        diff_b64    = cv2_to_b64(diff_map)
        aligned_b64 = cv2_to_b64(aligned)

        # Step 5: Gemini summary
        ai_summary = describe_defects_with_gemini(
            defects,
            pil_to_b64(Image.fromarray(cv2.cvtColor(aw_img, cv2.COLOR_BGR2RGB))),
            result_b64
        )

        # Format response (remove contour - not JSON serializable)
        defects_out = [{k: v for k, v in d.items() if k != 'contour'} for d in defects]

        verdict = 'PASS' if len(defects) == 0 else (
            'FAIL' if any(d['severity'] == 'high' for d in defects) else 'REVIEW'
        )

        return jsonify({
            'verdict': verdict,
            'defect_count': len(defects),
            'defects': defects_out,
            'result_image': result_b64,
            'diff_image': diff_b64,
            'aligned_image': aligned_b64,
            'ai_summary': ai_summary,
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f'QC Inspector running at http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
