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
import math

app = Flask(__name__, static_folder='static')
CORS(app)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# ─── Helpers ────────────────────────────────────────────────────────────────

def b64_to_cv2(b64_str):
    data = base64.b64decode(b64_str)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def cv2_to_b64(img):
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')

def pil_to_b64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format='JPEG', quality=92)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def img_to_b64_jpeg(img):
    """CV2 image to base64 JPEG for Gemini."""
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode('utf-8')

def resize_keep_aspect(img, max_dim=2400):
    h, w = img.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def estimate_dpi(img_shape, physical_width_mm=210):
    """Estimate DPI assuming image is A4 width by default."""
    w = img_shape[1]
    dpi = w / (physical_width_mm / 25.4)
    return max(dpi, 72)

def mm2_to_px2(mm2, dpi):
    """Convert mm² to pixel²."""
    px_per_mm = dpi / 25.4
    return mm2 * (px_per_mm ** 2)

# ─── Step 1: Align ───────────────────────────────────────────────────────────

def align_images(aw_img, print_img):
    aw_gray = cv2.cvtColor(aw_img, cv2.COLOR_BGR2GRAY)
    pr_gray = cv2.cvtColor(print_img, cv2.COLOR_BGR2GRAY)

    try:
        detector = cv2.SIFT_create(nfeatures=3000)
        norm = cv2.NORM_L2
    except Exception:
        detector = cv2.ORB_create(nfeatures=3000)
        norm = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(aw_gray, None)
    kp2, des2 = detector.detectAndCompute(pr_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        h, w = aw_img.shape[:2]
        return cv2.resize(print_img, (w, h))

    bf = cv2.BFMatcher(norm, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
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
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if H is None:
        h, w = aw_img.shape[:2]
        return cv2.resize(print_img, (w, h))

    h, w = aw_img.shape[:2]
    aligned = cv2.warpPerspective(print_img, H, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    return aligned

# ─── Step 2: Physical defect detection (OpenCV) ──────────────────────────────

def detect_physical_defects(aw_img, aligned_print, sensitivity, dpi):
    """
    Detect physical print defects: spot, dot, hickey, scratch, streak, blur, missing ink.
    Min area: 0.5mm² converted to pixels.
    """
    # Min area in pixels² (0.5mm²)
    min_area_px = mm2_to_px2(0.5, dpi)
    # For very thin scratches: min length in pixels (0.2mm width × 5mm length)
    min_scratch_length_px = (5.0 / 25.4) * dpi   # 5mm in pixels
    min_scratch_width_px  = (0.1 / 25.4) * dpi   # 0.1mm min width

    # ── Color diff in LAB ──
    aw_lab = cv2.cvtColor(aw_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    pr_lab = cv2.cvtColor(aligned_print, cv2.COLOR_BGR2LAB).astype(np.float32)

    diff_L = np.abs(aw_lab[:,:,0] - pr_lab[:,:,0]) * 1.5
    diff_A = np.abs(aw_lab[:,:,1] - pr_lab[:,:,1])
    diff_B = np.abs(aw_lab[:,:,2] - pr_lab[:,:,2])
    diff   = (diff_L + diff_A + diff_B) / 3.0

    # ── Threshold ──
    thresh = (diff > sensitivity).astype(np.uint8) * 255

    # ── Morphological: remove salt-pepper noise, keep thin lines ──
    # Open with tiny kernel to remove 1-2px noise
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open)

    # Close small gaps in scratches/streaks
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    thresh_h = cv2.morphologyEx(thresh_clean, cv2.MORPH_CLOSE, k_close)
    k_close2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    thresh_v = cv2.morphologyEx(thresh_clean, cv2.MORPH_CLOSE, k_close2)
    thresh_lines = cv2.bitwise_or(thresh_h, thresh_v)

    # Final: combine both
    thresh_final = cv2.bitwise_or(thresh_clean, thresh_lines)

    # ── Find contours ──
    contours, _ = cv2.findContours(thresh_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = aw_img.shape[:2]
    total_area = h * w
    defects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Check min area (0.5mm²) — BUT allow thin scratches even if area < 0.5mm²
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        max_dim_px = max(bw, bh)

        is_long_scratch = (aspect > 8 and max_dim_px > min_scratch_length_px and min(bw, bh) >= max(min_scratch_width_px, 1))
        if area < min_area_px and not is_long_scratch:
            continue

        defect_type = classify_physical_defect(cnt, area, bw, bh, total_area, aw_img, aligned_print, dpi)
        if defect_type is None:
            continue

        mask_roi = np.zeros(thresh_final.shape, np.uint8)
        cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
        mean_diff = float(cv2.mean(diff, mask=mask_roi)[0])

        # Severity based on area in mm² and diff intensity
        area_mm2 = area / mm2_to_px2(1.0, dpi)
        if mean_diff > 70 or area_mm2 > 5:
            severity = 'high'
        elif mean_diff > 45 or area_mm2 > 1:
            severity = 'medium'
        else:
            severity = 'low'

        defects.append({
            'type': defect_type['type'],
            'label': defect_type['label'],
            'x': int(x), 'y': int(y), 'w': int(bw), 'h': int(bh),
            'area': int(area),
            'area_mm2': round(area_mm2, 3),
            'severity': severity,
            'mean_diff': round(mean_diff, 1),
            'contour': cnt
        })

    return defects, thresh_final

def classify_physical_defect(cnt, area, bw, bh, total_area, aw_img, pr_img, dpi):
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    rel_size = area / total_area

    # Min scratch length: 3mm
    min_scratch_px = (3.0 / 25.4) * dpi

    # ── Scratch / Vệt xước: rất dài và mảnh ──
    if aspect > 8 and max(bw, bh) > min_scratch_px:
        return {'type': 'scratch', 'label': 'Xước / vệt dài'}

    # ── Streak / Sọc: dài vừa ──
    if aspect > 4 and max(bw, bh) > min_scratch_px * 0.5:
        return {'type': 'streak', 'label': 'Vệt / sọc'}

    # ── Hickey: vòng tròn nhỏ đặc trưng ──
    if circularity > 0.65 and bw < (3.0/25.4)*dpi and bh < (3.0/25.4)*dpi:
        return {'type': 'hickey', 'label': 'Hickey'}

    # ── Dot: chấm nhỏ tròn ──
    if circularity > 0.5 and bw < (2.0/25.4)*dpi and bh < (2.0/25.4)*dpi:
        return {'type': 'dot', 'label': 'Dot / chấm mực'}

    # ── Spot: blob tròn vừa ──
    if circularity > 0.4 and bw < (5.0/25.4)*dpi:
        return {'type': 'spot', 'label': 'Spot / đốm'}

    # ── So sánh sáng/tối để phát hiện mất mực hoặc thừa mực ──
    x, y, w2, h2 = cv2.boundingRect(cnt)
    aw_roi = aw_img[y:y+h2, x:x+w2]
    pr_roi = pr_img[y:y+h2, x:x+w2]
    if aw_roi.size == 0 or pr_roi.size == 0:
        return {'type': 'anomaly', 'label': 'Bất thường'}

    aw_mean = float(np.mean(cv2.cvtColor(aw_roi, cv2.COLOR_BGR2GRAY)))
    pr_mean = float(np.mean(cv2.cvtColor(pr_roi, cv2.COLOR_BGR2GRAY)))

    if pr_mean > aw_mean + 15:
        return {'type': 'missing_ink', 'label': 'Thiếu mực / mất nét'}
    if pr_mean < aw_mean - 15:
        return {'type': 'extra_ink', 'label': 'Thừa mực / lem'}

    # ── Blur: giảm độ sắc nét ──
    aw_lap = cv2.Laplacian(cv2.cvtColor(aw_roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    pr_lap = cv2.Laplacian(cv2.cvtColor(pr_roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if pr_lap < aw_lap * 0.45:
        return {'type': 'blur', 'label': 'Nhòe / mờ'}

    if rel_size < 0.00005:
        return None
    return {'type': 'anomaly', 'label': 'Bất thường'}

# ─── Step 3: Text/diacritic check via Gemini ─────────────────────────────────

def check_text_with_gemini(aw_img, aligned_print, physical_defects):
    """
    Use Gemini to check text content, diacritics, punctuation differences.
    Returns list of text defects with approximate locations.
    """
    if not GEMINI_API_KEY:
        return []

    aw_b64    = img_to_b64_jpeg(aw_img)
    print_b64 = img_to_b64_jpeg(aligned_print)
    h, w = aw_img.shape[:2]

    prompt = f"""Bạn là chuyên gia QC in ấn bao bì. So sánh 2 ảnh:
- Ảnh 1: Artwork gốc (AW) — CHUẨN
- Ảnh 2: Tờ in thực tế — cần kiểm tra

CHỈ kiểm tra NỘI DUNG VĂN BẢN (KHÔNG cần báo lỗi vật lý như vệt, đốm):
1. Dấu thanh tiếng Việt: sắc/huyền/hỏi/ngã/nặng có đúng không?
2. Dấu câu: chấm (.) phẩy (,) chấm than (!) — thừa/thiếu không?
3. Chữ cái sai, số sai, từ thiếu/thừa

Ảnh có kích thước {w}x{h} pixels.

Trả về JSON (CHỈ JSON):
{{"text_defects":[{{"type":"wrong_diacritic|missing_punct|extra_punct|wrong_char|missing_word|extra_word","label":"tên tiếng Việt","detail":"mô tả: chữ gì, sai thành gì","x":0,"y":0,"w":50,"h":30}}]}}
Nếu không có lỗi văn bản: {{"text_defects":[]}}"""

    try:
        body = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": aw_b64}},
                {"inline_data": {"mime_type": "image/jpeg", "data": print_b64}},
                {"text": prompt}
            ]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2000}
        }
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json=body, timeout=45
        )
        data = r.json()
        raw = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        cleaned = raw.replace('```json', '').replace('```', '').strip()
        result = json.loads(cleaned)
        return result.get('text_defects', [])
    except Exception as e:
        print(f"Gemini text check error: {e}")
        return []

# ─── Step 4: AI Summary ──────────────────────────────────────────────────────

def get_ai_summary(all_defects, aw_img):
    if not GEMINI_API_KEY or not all_defects:
        return None

    summary_list = "\n".join([
        f"- #{i+1}: {d.get('label','?')} | mức độ: {d.get('severity','?')} | {d.get('detail') or d.get('area_mm2','')}"
        for i, d in enumerate(all_defects)
    ])

    prompt = f"""Chuyên gia QC in ấn bao bì. Phát hiện {len(all_defects)} lỗi:
{summary_list}

Viết báo cáo QC tiếng Việt ngắn gọn (tối đa 120 từ):
1. Đánh giá tổng thể
2. Lỗi nghiêm trọng nhất
3. Khuyến nghị: in lại / chấp nhận / kiểm tra thêm"""

    try:
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300}
        }
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json=body, timeout=30
        )
        data = r.json()
        return data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
    except Exception as e:
        return f"(Lỗi AI: {e})"

# ─── Step 5: Draw all defects ────────────────────────────────────────────────

DEFECT_COLORS = {
    'scratch':      (255,  80, 200),
    'streak':       (255, 140,   0),
    'hickey':       (  0, 100, 255),
    'dot':          (  0, 220, 255),
    'spot':         (  0, 255, 180),
    'missing_ink':  ( 50, 255,  50),
    'extra_ink':    (255,  50,  50),
    'blur':         (255, 220,   0),
    'anomaly':      (180, 180, 180),
    # Text defects
    'wrong_diacritic': (255,   0, 100),
    'missing_punct':   (255,   0, 200),
    'extra_punct':     (200,   0, 255),
    'wrong_char':      (255,  50, 150),
    'missing_word':    (255, 100,   0),
    'extra_word':      (200, 100,  50),
}

def draw_all_defects(img, physical_defects, text_defects):
    out = img.copy()
    all_d = list(physical_defects) + [dict(d, **{'is_text': True}) for d in text_defects]

    for i, d in enumerate(all_d):
        color = DEFECT_COLORS.get(d.get('type', ''), (200, 200, 200))
        x, y = d.get('x', 0), d.get('y', 0)
        w, h = d.get('w', 20), d.get('h', 20)

        # Clamp to image bounds
        ih, iw = out.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, iw-x), min(h, ih-y)
        if w <= 0 or h <= 0:
            continue

        # Semi-transparent fill
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        alpha = 0.35 if d.get('is_text') else 0.25
        cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)

        # Border — thicker for text errors, dashed-style for physical
        thick = 3 if d.get('severity') == 'high' or d.get('is_text') else 2
        cv2.rectangle(out, (x, y), (x+w, y+h), color, thick)

        # Label
        label = f"#{i+1} {d.get('label', '?')}"
        if d.get('is_text'):
            label += ' [T]'
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        (tw, th2), _ = cv2.getTextSize(label, font, scale, 1)
        by = max(y - 3, th2 + 3)
        cv2.rectangle(out, (x, by - th2 - 3), (x + tw + 5, by + 1), color, -1)
        cv2.putText(out, label, (x+3, by-1), font, scale, (0, 0, 0), 1, cv2.LINE_AA)

    return out

# ─── Main API ────────────────────────────────────────────────────────────────

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400

    aw_b64    = data.get('awImage')
    print_b64 = data.get('printImage')
    sensitivity   = int(data.get('sensitivity', 30))
    check_text    = data.get('checkText', True)
    dpi_override  = data.get('dpi', None)

    if not aw_b64 or not print_b64:
        return jsonify({'error': 'Thiếu ảnh'}), 400

    try:
        aw_img    = b64_to_cv2(aw_b64)
        print_img = b64_to_cv2(print_b64)

        if aw_img is None or print_img is None:
            return jsonify({'error': 'Không đọc được ảnh'}), 400

        # Resize — keep high resolution for better detection
        aw_img,    scale_aw    = resize_keep_aspect(aw_img,    2400)
        print_img, scale_print = resize_keep_aspect(print_img, 2400)

        # Estimate DPI
        dpi = float(dpi_override) if dpi_override else estimate_dpi(aw_img.shape)

        # Align
        aligned = align_images(aw_img, print_img)

        # Physical defects (OpenCV)
        physical_defects, diff_map = detect_physical_defects(aw_img, aligned, sensitivity, dpi)

        # Text/diacritic check (Gemini) — run in parallel conceptually
        text_defects = []
        if check_text and GEMINI_API_KEY:
            text_defects = check_text_with_gemini(aw_img, aligned, physical_defects)

        # Draw
        result_img = draw_all_defects(aligned, physical_defects, text_defects)

        # Encode
        result_b64  = cv2_to_b64(result_img)
        diff_b64    = cv2_to_b64(diff_map)

        # Merge all defects for summary
        all_defects = physical_defects + [dict(d, severity='high') for d in text_defects]

        # AI summary
        ai_summary = get_ai_summary(all_defects, aw_img) if all_defects else None

        # Clean output (remove contour)
        physical_out = [{k: v for k, v in d.items() if k != 'contour'} for d in physical_defects]
        text_out     = [dict(d, is_text=True) for d in text_defects]

        verdict = 'PASS'
        if any(d.get('severity') == 'high' for d in all_defects) or len(text_defects) > 0:
            verdict = 'FAIL'
        elif len(all_defects) > 0:
            verdict = 'REVIEW'

        return jsonify({
            'verdict': verdict,
            'defect_count': len(all_defects),
            'physical_count': len(physical_defects),
            'text_count': len(text_defects),
            'defects': physical_out + text_out,
            'result_image': result_b64,
            'diff_image': diff_b64,
            'ai_summary': ai_summary,
            'dpi_used': round(dpi, 1),
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
    print(f'QC Inspector Pro running at http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
