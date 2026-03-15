from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import requests
import json
import math
from io import BytesIO
from PIL import Image, ExifTags

app = Flask(__name__, static_folder='static')
CORS(app)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# ── Tiêu chuẩn chất lượng in (mm) ────────────────────────────────────────────
QC_STANDARDS = {
    'dot':          {'warn': 0.3,  'fail': 0.5,  'unit': 'diameter_mm'},
    'spot':         {'warn': 0.3,  'fail': 0.5,  'unit': 'diameter_mm'},
    'hickey':       {'warn': 0.3,  'fail': 0.5,  'unit': 'diameter_mm'},
    'scratch':      {'warn': 3.0,  'fail': 5.0,  'unit': 'length_mm'},
    'streak':       {'warn': 3.0,  'fail': 5.0,  'unit': 'length_mm'},
    'missing_ink':  {'warn': 0.5,  'fail': 1.0,  'unit': 'area_mm2'},
    'extra_ink':    {'warn': 0.5,  'fail': 1.0,  'unit': 'area_mm2'},
    'blur':         {'warn': 1.0,  'fail': 2.0,  'unit': 'area_mm2'},
    'anomaly':      {'warn': 0.5,  'fail': 1.0,  'unit': 'area_mm2'},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def b64_to_cv2(b64_str):
    data = base64.b64decode(b64_str)
    arr  = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str)))

def cv2_to_b64(img):
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')

def img_to_b64_jpeg(img, quality=92):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')

def px_to_mm(px, dpi):
    return px / dpi * 25.4

def mm_to_px(mm, dpi):
    return mm / 25.4 * dpi

def area_px_to_mm2(area_px, dpi):
    px_per_mm = dpi / 25.4
    return area_px / (px_per_mm ** 2)

def diameter_from_area_mm(area_mm2):
    return 2 * math.sqrt(area_mm2 / math.pi)

def judge_defect(defect_type, measurement_mm):
    """
    Phán định PASS/WARN/FAIL theo tiêu chuẩn QC in ấn.
    measurement_mm là đường kính (mm) cho dot/spot/hickey,
    hoặc chiều dài (mm) cho scratch/streak,
    hoặc diện tích (mm²) cho các loại khác.
    """
    std = QC_STANDARDS.get(defect_type, QC_STANDARDS['anomaly'])
    if measurement_mm >= std['fail']:
        return 'FAIL', std['fail'], std['unit']
    elif measurement_mm >= std['warn']:
        return 'WARN', std['warn'], std['unit']
    else:
        return 'PASS', std['warn'], std['unit']

# ── Đọc DPI/kích thước từ AW ─────────────────────────────────────────────────

def extract_aw_info(b64_str, file_type='image'):
    """
    Trả về dict: { dpi, width_mm, height_mm, source }
    """
    result = {'dpi': None, 'width_mm': None, 'height_mm': None, 'source': 'unknown'}

    if file_type == 'pdf':
        result = _extract_from_pdf(b64_str)
    else:
        result = _extract_from_image(b64_str)

    # Fallback nếu không đọc được
    if not result['dpi']:
        result['dpi'] = 150.0
        result['source'] = 'fallback_150dpi'

    return result

def _extract_from_pdf(b64_str):
    """Đọc PDF, lấy kích thước trang thật và render thành ảnh."""
    try:
        import fitz  # PyMuPDF
        pdf_bytes = base64.b64decode(b64_str)
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        page = doc[0]

        # Kích thước trang thật (points, 1 point = 1/72 inch)
        rect = page.rect
        width_pt  = rect.width
        height_pt = rect.height
        width_mm  = width_pt  / 72 * 25.4
        height_mm = height_pt / 72 * 25.4

        # Render ở DPI cao để phát hiện lỗi nhỏ
        render_dpi = 300
        mat = fitz.Matrix(render_dpi / 72, render_dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_bytes = pix.tobytes('png')

        # Convert to CV2
        arr = np.frombuffer(img_bytes, np.uint8)
        img_cv2 = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Encode back to b64 for processing
        _, buf = cv2.imencode('.png', img_cv2)
        rendered_b64 = base64.b64encode(buf).decode('utf-8')

        doc.close()
        return {
            'dpi': float(render_dpi),
            'width_mm': round(width_mm, 2),
            'height_mm': round(height_mm, 2),
            'source': f'pdf_metadata ({width_mm:.1f}×{height_mm:.1f}mm @ {render_dpi}dpi)',
            'rendered_b64': rendered_b64,
            'img_cv2': img_cv2,
        }
    except Exception as e:
        return {'dpi': None, 'width_mm': None, 'height_mm': None, 'source': f'pdf_error:{e}'}

def _extract_from_image(b64_str):
    """Đọc PNG/JPG, lấy DPI từ EXIF/metadata."""
    try:
        pil_img = b64_to_pil(b64_str)
        w_px, h_px = pil_img.size

        dpi_x, dpi_y = None, None

        # 1. Thử đọc DPI từ info (PNG dpi tag, JPEG JFIF)
        if 'dpi' in pil_img.info:
            dpi_x, dpi_y = pil_img.info['dpi']
        elif 'jfif_density' in pil_img.info:
            unit = pil_img.info.get('jfif_densityunit', 1)
            dx, dy = pil_img.info['jfif_density']
            if unit == 1:  # dots per inch
                dpi_x, dpi_y = dx, dy
            elif unit == 2:  # dots per cm
                dpi_x, dpi_y = dx * 2.54, dy * 2.54

        # 2. Thử EXIF
        if not dpi_x:
            try:
                exif = pil_img._getexif()
                if exif:
                    for tag_id, val in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, '')
                        if tag == 'XResolution':
                            dpi_x = float(val[0]) / float(val[1]) if isinstance(val, tuple) else float(val)
                        elif tag == 'YResolution':
                            dpi_y = float(val[0]) / float(val[1]) if isinstance(val, tuple) else float(val)
            except Exception:
                pass

        if dpi_x and dpi_x > 10:
            dpi = float(dpi_x)
            width_mm  = w_px / dpi * 25.4
            height_mm = h_px / dpi * 25.4
            source = f'image_metadata ({width_mm:.1f}×{height_mm:.1f}mm @ {dpi:.0f}dpi)'
        else:
            # Không có DPI — trả về None để frontend hỏi user
            dpi, width_mm, height_mm = None, None, None
            source = 'no_dpi_metadata'

        # Convert to CV2
        img_cv2 = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)

        return {
            'dpi': dpi,
            'width_mm': round(width_mm, 2) if width_mm else None,
            'height_mm': round(height_mm, 2) if height_mm else None,
            'source': source,
            'img_cv2': img_cv2,
        }
    except Exception as e:
        return {'dpi': None, 'width_mm': None, 'height_mm': None, 'source': f'image_error:{e}'}

# ── Align ─────────────────────────────────────────────────────────────────────

def align_images(aw_img, print_img):
    aw_gray = cv2.cvtColor(aw_img, cv2.COLOR_BGR2GRAY)
    pr_gray = cv2.cvtColor(print_img, cv2.COLOR_BGR2GRAY)

    try:
        detector = cv2.SIFT_create(nfeatures=4000)
        norm = cv2.NORM_L2
    except Exception:
        detector = cv2.ORB_create(nfeatures=4000)
        norm = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(aw_gray, None)
    kp2, des2 = detector.detectAndCompute(pr_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        h, w = aw_img.shape[:2]
        return cv2.resize(print_img, (w, h)), 1.0

    bf = cv2.BFMatcher(norm, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    except Exception:
        good = []

    if len(good) < 8:
        h, w = aw_img.shape[:2]
        return cv2.resize(print_img, (w, h)), 1.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if H is None:
        h, w = aw_img.shape[:2]
        return cv2.resize(print_img, (w, h)), 1.0

    # Tính scale factor từ homography
    scale_x = math.sqrt(H[0,0]**2 + H[1,0]**2)
    scale_y = math.sqrt(H[0,1]**2 + H[1,1]**2)
    scale_factor = (scale_x + scale_y) / 2.0

    h, w = aw_img.shape[:2]
    aligned = cv2.warpPerspective(print_img, H, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    return aligned, scale_factor

# ── Detect defects ────────────────────────────────────────────────────────────

def detect_defects(aw_img, aligned_print, sensitivity, dpi_aw, scale_factor):
    """
    dpi_aw: DPI thật của AW gốc
    scale_factor: tỉ lệ print/AW từ homography
    → dpi_effective: DPI hiệu dụng để tính kích thước thật trên tờ in
    """
    dpi_effective = dpi_aw / scale_factor if scale_factor > 0 else dpi_aw

    # LAB diff
    aw_lab = cv2.cvtColor(aw_img,        cv2.COLOR_BGR2LAB).astype(np.float32)
    pr_lab = cv2.cvtColor(aligned_print, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff   = (np.abs(aw_lab[:,:,0] - pr_lab[:,:,0]) * 1.5 +
              np.abs(aw_lab[:,:,1] - pr_lab[:,:,1]) +
              np.abs(aw_lab[:,:,2] - pr_lab[:,:,2])) / 3.0

    thresh = (diff > sensitivity).astype(np.uint8) * 255

    # Morphological ops — giữ vệt mảnh + lọc noise
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k2)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (7,2))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (2,7))
    th = cv2.bitwise_or(
        cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kh),
        cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kv)
    )
    thresh_final = cv2.bitwise_or(thresh, th)

    contours, _ = cv2.findContours(thresh_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Min detectable: dot đường kính 0.2mm
    min_diameter_mm = 0.2
    min_area_px = math.pi * (mm_to_px(min_diameter_mm/2, dpi_aw))**2

    defects = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        max_dim_px = max(bw, bh)
        min_dim_px = min(bw, bh)
        aspect = max_dim_px / max(min_dim_px, 1)

        # Cho phép scratch mảnh dài dù area nhỏ
        is_scratch = aspect > 6 and max_dim_px > mm_to_px(2.0, dpi_aw)
        if area_px < min_area_px and not is_scratch:
            continue

        # Phân loại
        dtype = classify_defect(cnt, area_px, bw, bh, aw_img, aligned_print, dpi_aw)
        if dtype is None:
            continue

        # ── Tính kích thước thật ──
        area_mm2   = area_px_to_mm2(area_px, dpi_aw)
        diameter_mm = diameter_from_area_mm(area_mm2)
        length_mm  = px_to_mm(max_dim_px, dpi_aw)
        width_mm   = px_to_mm(min_dim_px, dpi_aw)

        # Chọn measurement phù hợp để phán định
        if dtype in ('dot', 'spot', 'hickey'):
            measurement = diameter_mm
            size_str = f'⌀{diameter_mm:.2f}mm'
        elif dtype in ('scratch', 'streak'):
            measurement = length_mm
            size_str = f'{length_mm:.2f}×{width_mm:.2f}mm'
        else:
            measurement = area_mm2
            size_str = f'{area_mm2:.3f}mm²'

        verdict, threshold, unit = judge_defect(dtype, measurement)
        if verdict == 'PASS':
            continue  # Quá nhỏ, bỏ qua

        # Severity từ verdict
        severity = 'high' if verdict == 'FAIL' else 'medium'

        # Mean diff
        mask_roi = np.zeros(thresh_final.shape, np.uint8)
        cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
        mean_diff = float(cv2.mean(diff, mask=mask_roi)[0])

        defects.append({
            'type': dtype,
            'label': DEFECT_LABELS.get(dtype, dtype),
            'x': int(x), 'y': int(y), 'w': int(bw), 'h': int(bh),
            'area_px': int(area_px),
            'area_mm2': round(area_mm2, 3),
            'diameter_mm': round(diameter_mm, 3),
            'length_mm': round(length_mm, 3),
            'width_mm': round(width_mm, 3),
            'size_str': size_str,
            'verdict': verdict,
            'threshold_mm': threshold,
            'severity': severity,
            'mean_diff': round(mean_diff, 1),
            'contour': cnt,
        })

    return defects, thresh_final, dpi_effective

def classify_defect(cnt, area_px, bw, bh, aw_img, pr_img, dpi):
    perimeter    = cv2.arcLength(cnt, True)
    circularity  = 4 * math.pi * area_px / (perimeter**2) if perimeter > 0 else 0
    aspect       = max(bw,bh) / max(min(bw,bh), 1)
    max_dim_px   = max(bw, bh)

    # Scratch: rất dài và mảnh
    if aspect > 8 and max_dim_px > mm_to_px(3.0, dpi):
        return 'scratch'
    # Streak: dài vừa
    if aspect > 4 and max_dim_px > mm_to_px(2.0, dpi):
        return 'streak'
    # Hickey: tròn nhỏ < 3mm
    if circularity > 0.65 and bw < mm_to_px(3.0, dpi):
        return 'hickey'
    # Dot: tròn nhỏ < 1.5mm
    if circularity > 0.5 and bw < mm_to_px(1.5, dpi):
        return 'dot'
    # Spot: blob vừa
    if circularity > 0.35 and bw < mm_to_px(6.0, dpi):
        return 'spot'

    x, y, w2, h2 = cv2.boundingRect(cnt)
    aw_roi = aw_img[y:y+h2, x:x+w2]
    pr_roi = pr_img[y:y+h2, x:x+w2]
    if aw_roi.size == 0 or pr_roi.size == 0:
        return 'anomaly'

    aw_mean = float(np.mean(cv2.cvtColor(aw_roi, cv2.COLOR_BGR2GRAY)))
    pr_mean = float(np.mean(cv2.cvtColor(pr_roi, cv2.COLOR_BGR2GRAY)))

    if pr_mean > aw_mean + 15: return 'missing_ink'
    if pr_mean < aw_mean - 15: return 'extra_ink'

    aw_lap = cv2.Laplacian(cv2.cvtColor(aw_roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    pr_lap = cv2.Laplacian(cv2.cvtColor(pr_roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if pr_lap < aw_lap * 0.4:
        return 'blur'

    return 'anomaly'

DEFECT_LABELS = {
    'dot':         'Dot / chấm mực',
    'spot':        'Spot / đốm',
    'hickey':      'Hickey',
    'scratch':     'Xước / vệt dài',
    'streak':      'Vệt / sọc',
    'missing_ink': 'Thiếu mực / mất nét',
    'extra_ink':   'Thừa mực / lem',
    'blur':        'Nhòe / mờ',
    'anomaly':     'Bất thường',
}

# ── Draw ──────────────────────────────────────────────────────────────────────

DEFECT_COLORS_CV = {
    'dot':         (255, 220,   0),
    'spot':        (  0, 200, 255),
    'hickey':      (  0, 100, 255),
    'scratch':     (255,  50, 200),
    'streak':      (255, 140,   0),
    'missing_ink': ( 50, 255,  50),
    'extra_ink':   (255,  50,  50),
    'blur':        (255, 255,   0),
    'anomaly':     (180, 180, 180),
    # text
    'wrong_diacritic': (255,   0, 120),
    'missing_punct':   (255,   0, 200),
    'wrong_char':      (200,   0, 255),
    'missing_word':    (255, 100,   0),
}

def draw_defects(img, physical, text_defects):
    out = img.copy()
    ih, iw = out.shape[:2]
    all_d = [('phys', d) for d in physical] + [('text', d) for d in text_defects]

    for i, (kind, d) in enumerate(all_d):
        color = DEFECT_COLORS_CV.get(d.get('type',''), (180,180,180))
        x = max(0, d.get('x',0)); y = max(0, d.get('y',0))
        w = min(d.get('w',20), iw-x); h = min(d.get('h',20), ih-y)
        if w <= 0 or h <= 0: continue

        overlay = out.copy()
        cv2.rectangle(overlay, (x,y), (x+w,y+h), color, -1)
        cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)

        thick = 3 if d.get('verdict') == 'FAIL' or kind == 'text' else 2
        cv2.rectangle(out, (x,y), (x+w,y+h), color, thick)

        # Label với kích thước thật
        size_info = d.get('size_str', '')
        verdict   = d.get('verdict', '')
        label = f"#{i+1} {d.get('label','?')}"
        if size_info: label += f" {size_info}"
        if verdict:   label += f" [{verdict}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        (tw, th2), _ = cv2.getTextSize(label, font, scale, 1)
        by = max(y - 3, th2 + 3)
        cv2.rectangle(out, (x, by-th2-3), (x+tw+5, by+1), color, -1)
        cv2.putText(out, label, (x+3, by-1), font, scale, (0,0,0), 1, cv2.LINE_AA)

    return out

# ── Gemini text check ─────────────────────────────────────────────────────────

def check_text_gemini(aw_img, aligned_print):
    if not GEMINI_API_KEY:
        return []
    h, w = aw_img.shape[:2]
    prompt = f"""So sánh Ảnh 1 (AW gốc chuẩn) và Ảnh 2 (tờ in, {w}×{h}px).
Chỉ kiểm tra NỘI DUNG VĂN BẢN: dấu thanh tiếng Việt, dấu câu, ký tự sai/thiếu/thừa.
JSON (chỉ JSON): {{"text_defects":[{{"type":"wrong_diacritic|missing_punct|extra_punct|wrong_char|missing_word","label":"tên VN","detail":"mô tả chi tiết","x":0,"y":0,"w":40,"h":25}}]}}"""
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[
                {"inline_data":{"mime_type":"image/jpeg","data":img_to_b64_jpeg(aw_img)}},
                {"inline_data":{"mime_type":"image/jpeg","data":img_to_b64_jpeg(aligned_print)}},
                {"text":prompt}
            ]}],"generationConfig":{"temperature":0.1,"maxOutputTokens":2000}},
            timeout=45
        )
        raw = r.json().get('candidates',[{}])[0].get('content',{}).get('parts',[{}])[0].get('text','')
        return json.loads(raw.replace('```json','').replace('```','').strip()).get('text_defects',[])
    except Exception as e:
        print(f"Gemini text error: {e}")
        return []

def get_ai_summary(all_defects):
    if not GEMINI_API_KEY or not all_defects:
        return None
    lines = "\n".join([f"- #{i+1}: {d.get('label','?')} | {d.get('size_str','?')} | {d.get('verdict','?')}"
                       for i, d in enumerate(all_defects)])
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[{"text":
                f"Chuyên gia QC in ấn. {len(all_defects)} lỗi:\n{lines}\n"
                f"Báo cáo tiếng Việt ≤120 từ: đánh giá tổng thể, lỗi nghiêm trọng nhất, khuyến nghị."
            }]}],"generationConfig":{"temperature":0.2,"maxOutputTokens":300}},
            timeout=30
        )
        return r.json().get('candidates',[{}])[0].get('content',{}).get('parts',[{}])[0].get('text','')
    except:
        return None

# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.route('/api/get_aw_info', methods=['POST','OPTIONS'])
def get_aw_info():
    """Đọc DPI/kích thước từ file AW trước khi phân tích."""
    if request.method == 'OPTIONS': return '',204
    data = request.get_json()
    b64  = data.get('fileData','')
    ftype = data.get('fileType','image')  # 'pdf' or 'image'
    info = extract_aw_info(b64, ftype)
    # Không trả về img_cv2 (binary)
    return jsonify({k:v for k,v in info.items() if k not in ('img_cv2','rendered_b64')})

@app.route('/api/analyze', methods=['POST','OPTIONS'])
def analyze():
    if request.method == 'OPTIONS': return '',204
    data = request.get_json()
    if not data: return jsonify({'error':'No data'}),400

    aw_b64      = data.get('awImage')
    print_b64   = data.get('printImage')
    aw_file_b64 = data.get('awFileData')   # file gốc (PDF/PNG full)
    aw_type     = data.get('awFileType','image')
    sensitivity = int(data.get('sensitivity', 28))
    check_text  = data.get('checkText', True)
    manual_dpi  = data.get('manualDpi', None)  # user nhập tay nếu không đọc được

    if not aw_b64 or not print_b64:
        return jsonify({'error':'Thiếu ảnh'}),400

    try:
        # ── Lấy DPI từ file AW gốc ──
        aw_info = {'dpi': None, 'width_mm': None, 'source': 'none'}
        aw_cv2_hires = None

        if aw_file_b64:
            aw_info = extract_aw_info(aw_file_b64, aw_type)
            aw_cv2_hires = aw_info.get('img_cv2')

        if manual_dpi:
            aw_info['dpi'] = float(manual_dpi)
            aw_info['source'] = f'manual_{manual_dpi}dpi'

        # Fallback nếu vẫn không có DPI
        dpi_aw = aw_info.get('dpi') or 150.0

        # ── Decode ảnh hiển thị (canvas) ──
        aw_canvas   = b64_to_cv2(aw_b64)
        print_canvas = b64_to_cv2(print_b64)
        if aw_canvas is None or print_canvas is None:
            return jsonify({'error':'Không đọc được ảnh'}),400

        # Dùng ảnh hi-res nếu có (từ PDF render)
        aw_img = aw_cv2_hires if aw_cv2_hires is not None else aw_canvas

        # Resize print để match AW
        h_aw, w_aw = aw_img.shape[:2]
        print_resized = cv2.resize(print_canvas,
                                    (w_aw, h_aw) if aw_cv2_hires is not None else (print_canvas.shape[1], print_canvas.shape[0]))

        # ── Align ──
        aligned, scale_factor = align_images(aw_img, print_resized)

        # ── Physical defects ──
        physical, diff_map, dpi_effective = detect_defects(
            aw_img, aligned, sensitivity, dpi_aw, scale_factor
        )

        # ── Text check ──
        text_defects = check_text_gemini(aw_img, aligned) if check_text and GEMINI_API_KEY else []

        # ── Draw ──
        result_img = draw_defects(aligned, physical, text_defects)

        # ── Encode ──
        result_b64 = cv2_to_b64(result_img)
        diff_b64   = cv2_to_b64(diff_map)

        # ── AI Summary ──
        all_d = physical + [dict(d, size_str=d.get('detail',''), verdict='FAIL') for d in text_defects]
        ai_summary = get_ai_summary(all_d) if all_d else None

        physical_out = [{k:v for k,v in d.items() if k != 'contour'} for d in physical]

        # ── Verdict tổng thể ──
        fail_count = sum(1 for d in physical if d['verdict']=='FAIL') + len(text_defects)
        warn_count = sum(1 for d in physical if d['verdict']=='WARN')
        if fail_count > 0:   verdict = 'FAIL'
        elif warn_count > 0: verdict = 'WARN'
        else:                verdict = 'PASS'

        return jsonify({
            'verdict':        verdict,
            'defect_count':   len(all_d),
            'fail_count':     fail_count,
            'warn_count':     warn_count,
            'physical_count': len(physical),
            'text_count':     len(text_defects),
            'defects':        physical_out + [dict(d, is_text=True) for d in text_defects],
            'result_image':   result_b64,
            'diff_image':     diff_b64,
            'ai_summary':     ai_summary,
            'dpi_aw':         round(dpi_aw, 1),
            'dpi_effective':  round(dpi_effective, 1),
            'scale_factor':   round(scale_factor, 3),
            'aw_size':        f"{aw_info.get('width_mm','?')}×{aw_info.get('height_mm','?')}mm",
            'dpi_source':     aw_info.get('source','?'),
        })

    except Exception as e:
        import traceback
        return jsonify({'error':str(e),'trace':traceback.format_exc()}),500

@app.route('/', defaults={'path':''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f'QC Inspector Pro running at http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
