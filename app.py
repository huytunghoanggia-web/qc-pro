"""
QC Print Defect Inspector — No AW Required
==========================================
Pipeline: RECEIVE → RESIZE → NORMALIZE → DETECT → RENDER → RETURN
Detects: spot, dot, hickey, blur, smear, scratch, dirt, noise, stain
Works with: smartphone photo, different lighting, angle, scale
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64, os, math, json, requests
from io import BytesIO
from PIL import Image, ExifTags

app = Flask(__name__, static_folder='static')
CORS(app)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# ── Config ────────────────────────────────────────────────────────────────────
MAX_SIZE        = 1200   # max dimension after resize (avoid timeout)
MIN_SPOT_AREA   = 30     # px² minimum spot area
MIN_SCRATCH_LEN = 40     # px minimum scratch length
BLUR_THRESHOLD  = 40.0   # Laplacian variance below = blurred image
TEXTURE_THRESH  = 20     # local texture diff threshold
NOISE_AREA_LIMIT= 500    # px² max acceptable noise area

DEFECT_COLORS = {
    'spot':    (0,  200, 255),
    'dot':     (0,  220, 255),
    'hickey':  (0,  100, 255),
    'blur':    (255,220, 0  ),
    'smear':   (255,140, 0  ),
    'scratch': (255, 50, 200),
    'dirt':    (180, 50, 255),
    'noise':   (180,180, 180),
    'stain':   (50,  50, 255),
}
DEFECT_LABELS = {
    'spot':'Spot/đốm','dot':'Dot/chấm','hickey':'Hickey',
    'blur':'Nhòe/mờ','smear':'Lem mực','scratch':'Xước/vệt',
    'dirt':'Bẩn/tạp chất','noise':'Nhiễu/bất thường','stain':'Vệt bẩn',
}

def b64_to_cv2(b64):
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def cv2_to_b64(img):
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode()

def j2b64(img, q=90):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()

# ── DPI (optional, for mm sizing) ────────────────────────────────────────────
def get_dpi(b64, ftype='image'):
    try:
        pil = Image.open(BytesIO(base64.b64decode(b64)))
        dpi = None
        if 'dpi' in pil.info: dpi = float(pil.info['dpi'][0])
        if not dpi:
            try:
                ex = pil._getexif()
                if ex:
                    for tid,v in ex.items():
                        if ExifTags.TAGS.get(tid)=='XResolution':
                            dpi = float(v[0]/v[1]) if isinstance(v,tuple) else float(v)
            except: pass
        return dpi if (dpi and dpi > 10) else None
    except: return None

def px2mm(px, dpi): return px/dpi*25.4 if dpi else None
def a2mm2(a, dpi):  return a/(dpi/25.4)**2 if dpi else None
def diam_mm(a,dpi): return 2*math.sqrt(max(a,0)/math.pi)/dpi*25.4 if dpi else None

# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE — No AW needed
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(img_bgr, sensitivity, dpi):
    """
    Full defect detection pipeline on single image.
    sensitivity: 10 (most sensitive) → 70 (only obvious defects)
    """
    result = {
        'defects': [],
        'spot_count': 0,
        'blur_detected': False,
        'scratch_detected': False,
        'noise_area_px': 0,
        'laplacian_var': 0.0,
        'pipeline_steps': [],
    }

    # ── Step 1: Resize ──
    img, scale = _step_resize(img_bgr)
    result['pipeline_steps'].append(f'resize: {img_bgr.shape[1]}x{img_bgr.shape[0]} → {img.shape[1]}x{img.shape[0]} (scale={scale:.3f})')

    # ── Step 2: Convert to gray ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 3: Normalize brightness ──
    gray_norm = _step_normalize(gray)
    result['pipeline_steps'].append('normalize brightness')

    # ── Step 4: Denoise ──
    gray_dn = cv2.GaussianBlur(gray_norm, (5,5), 0)
    result['pipeline_steps'].append('denoise GaussianBlur(5,5)')

    # ── Step 5: Detect blur / smudge ──
    lap_var = _step_detect_blur(gray_norm)
    result['laplacian_var'] = round(lap_var, 2)
    blur_thresh = BLUR_THRESHOLD * (1 + (sensitivity-30)/100.0)
    if lap_var < blur_thresh:
        result['blur_detected'] = True
        result['defects'].append({
            'type': 'blur',
            'label': 'Nhòe/mờ toàn vùng',
            'detail': f'Laplacian variance={lap_var:.1f} < {blur_thresh:.1f}',
            'x': 0, 'y': 0,
            'w': img.shape[1], 'h': img.shape[0],
            'verdict': 'FAIL',
            'severity': 'high',
            'size_str': 'toàn vùng',
        })
    result['pipeline_steps'].append(f'blur check: lap_var={lap_var:.1f} blur={result["blur_detected"]}')

    # ── Step 6: Detect spot / dot / dirt / hickey ──
    spot_defects = _step_detect_spots(gray_dn, img, sensitivity, dpi, scale)
    result['defects'].extend(spot_defects)
    result['spot_count'] = len(spot_defects)
    result['pipeline_steps'].append(f'spot detect: found {len(spot_defects)}')

    # ── Step 7: Detect scratch / line defect ──
    scratch_defects, scratch_found = _step_detect_scratches(gray_dn, img, sensitivity, dpi, scale)
    result['defects'].extend(scratch_defects)
    result['scratch_detected'] = scratch_found
    result['pipeline_steps'].append(f'scratch detect: found {len(scratch_defects)}')

    # ── Step 8: Detect smear / stain (large ink spread) ──
    smear_defects = _step_detect_smear(gray_dn, img, sensitivity, dpi, scale)
    result['defects'].extend(smear_defects)
    result['pipeline_steps'].append(f'smear detect: found {len(smear_defects)}')

    # ── Step 9: Detect abnormal texture / noise area ──
    noise_defects, noise_area = _step_detect_noise(gray_dn, img, sensitivity, dpi, scale)
    result['defects'].extend(noise_defects)
    result['noise_area_px'] = noise_area
    result['pipeline_steps'].append(f'noise detect: area={noise_area}px found={len(noise_defects)}')

    # ── Sort by severity ──
    result['defects'].sort(key=lambda d: d.get('severity','medium')=='high', reverse=True)

    return result, img


# ── Step 1: Resize ────────────────────────────────────────────────────────────
def _step_resize(img):
    h, w = img.shape[:2]
    scale = min(MAX_SIZE/max(h,w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale


# ── Step 3: Normalize ────────────────────────────────────────────────────────
def _step_normalize(gray):
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


# ── Step 5: Blur detection ───────────────────────────────────────────────────
def _step_detect_blur(gray):
    """Laplacian variance — low = blurry image."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Step 6: Spot / dot / dirt / hickey ──────────────────────────────────────
def _step_detect_spots(gray, img, sensitivity, dpi, img_scale):
    """
    Detect dark/light blobs on surface using adaptive threshold.
    Classifies by shape + size into: dot, spot, hickey, dirt.
    """
    defects = []
    h, w = gray.shape[:2]

    # Adaptive threshold: removes uneven background lighting
    block = max(11, (min(h,w)//20) | 1)   # must be odd
    th_adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, 8
    )

    # Also Otsu for global blobs
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine
    th = cv2.bitwise_or(th_adapt, th_otsu)

    # Remove tiny noise
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k2)

    # Adjust min area by sensitivity
    # sensitivity 10 → min area 8px  |  30 → 30px  |  70 → 100px
    min_area = MIN_SPOT_AREA * (0.3 + sensitivity/50.0)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area: continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        # Ignore very large contours (likely background boundary, not defect)
        if area > h*w*0.15: continue

        perim = cv2.arcLength(cnt, True)
        circ  = 4*math.pi*area/perim**2 if perim>0 else 0
        aspect= max(bw,bh)/max(min(bw,bh),1)

        # Classify
        if circ > 0.6 and max(bw,bh) < 15:
            dtype = 'dot'
        elif circ > 0.5 and max(bw,bh) < 30:
            dtype = 'spot'
        elif circ > 0.4:
            # Hickey = ring-like, often has bright center
            roi = gray[y:y+bh, x:x+bw]
            if roi.size > 0:
                center_bright = float(np.mean(roi[bh//3:2*bh//3, bw//3:2*bw//3])) if bh>6 and bw>6 else 0
                edge_dark     = float(np.mean(roi))
                dtype = 'hickey' if center_bright > edge_dark+15 else 'spot'
            else:
                dtype = 'spot'
        else:
            dtype = 'dirt'

        # Compute real size if DPI available
        real_scale = img_scale  # account for resize
        a_real     = area / (real_scale**2)
        diam       = 2*math.sqrt(a_real/math.pi)
        diam_mm_v  = diam_mm(a_real, dpi) if dpi else None
        size_str   = f'⌀{diam_mm_v:.2f}mm' if diam_mm_v else f'⌀{diam:.0f}px'

        # Verdict
        if diam_mm_v:
            v = 'FAIL' if diam_mm_v>=0.5 else ('WARN' if diam_mm_v>=0.3 else 'WARN')
        else:
            v = 'FAIL' if diam>=8 else 'WARN'

        defects.append({
            'type': dtype, 'label': DEFECT_LABELS[dtype],
            'x':int(x), 'y':int(y), 'w':int(bw), 'h':int(bh),
            'area_px': int(area),
            'diameter_px': round(diam,1),
            'diameter_mm': round(diam_mm_v,3) if diam_mm_v else None,
            'size_str': size_str,
            'verdict': v,
            'severity': 'high' if v=='FAIL' else 'medium',
        })

    return defects


# ── Step 7: Scratch / line defect ────────────────────────────────────────────
def _step_detect_scratches(gray, img, sensitivity, dpi, img_scale):
    """
    Detect scratches using Canny + HoughLinesP.
    Long thin lines = scratch.
    """
    defects = []
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 40, 120)

    # Adjust min line length by sensitivity
    min_len = max(20, MIN_SCRATCH_LEN * (0.4 + sensitivity/100.0))
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi/180,
        threshold=max(15, int(30*(sensitivity/30.0))),
        minLineLength=min_len,
        maxLineGap=8
    )

    scratch_found = False
    if lines is not None:
        for ln in lines:
            x1,y1,x2,y2 = ln[0]
            length = math.sqrt((x2-x1)**2+(y2-y1)**2)
            dx = abs(x2-x1); dy = abs(y2-y1)
            # True scratch: long AND thin (high aspect)
            if length < min_len: continue
            angle = math.degrees(math.atan2(dy,dx+1e-6))
            # Not horizontal/vertical grid lines (those are likely borders)
            if angle < 2 or angle > 88:
                continue  # skip near-perfect H/V lines (likely image border)

            scratch_found = True
            xm = min(x1,x2); ym = min(y1,y2)
            bw = max(abs(x2-x1),3); bh = max(abs(y2-y1),3)

            length_mm_v = px2mm(length/img_scale, dpi)
            size_str = f'{length_mm_v:.1f}mm' if length_mm_v else f'{length:.0f}px'
            v = 'FAIL' if (length_mm_v and length_mm_v>=5.0) or length>=60 else 'WARN'

            defects.append({
                'type':'scratch','label':'Xước/vệt dài',
                'x':int(xm),'y':int(ym),'w':int(bw),'h':int(bh),
                'length_px': round(length,1),
                'length_mm': round(length_mm_v,2) if length_mm_v else None,
                'size_str': size_str,
                'verdict': v,
                'severity':'high' if v=='FAIL' else 'medium',
            })

    return defects, scratch_found


# ── Step 8: Smear / stain (large ink area) ───────────────────────────────────
def _step_detect_smear(gray, img, sensitivity, dpi, img_scale):
    """
    Detect smear = large area of ink spread.
    Uses morphological gradient to find ink boundaries.
    """
    defects = []
    h, w = gray.shape[:2]

    # Morphological gradient: highlights ink edges and smears
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k7)

    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU)

    # Close to form blobs
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kc)

    # Min area for smear: larger than spot
    min_area = max(200, 600*(sensitivity/50.0))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area: continue
        if area > h*w*0.20: continue  # skip background

        x,y,bw,bh = cv2.boundingRect(cnt)
        perim = cv2.arcLength(cnt,True)
        circ  = 4*math.pi*area/perim**2 if perim>0 else 0

        # Smear = irregular shape (low circularity, elongated)
        if circ > 0.5: continue  # too circular = spot, handled above

        a_real = area/(img_scale**2)
        amm2   = a2mm2(a_real, dpi)
        size_str = f'{amm2:.2f}mm²' if amm2 else f'{a_real:.0f}px²'
        v = 'FAIL' if (amm2 and amm2>=1.0) or a_real>=400 else 'WARN'

        defects.append({
            'type':'smear','label':'Lem mực/vệt bẩn',
            'x':int(x),'y':int(y),'w':int(bw),'h':int(bh),
            'area_px':int(area),
            'area_mm2':round(amm2,3) if amm2 else None,
            'size_str':size_str,
            'verdict':v,
            'severity':'high' if v=='FAIL' else 'medium',
        })
    return defects


# ── Step 9: Abnormal texture / noise ─────────────────────────────────────────
def _step_detect_noise(gray, img, sensitivity, dpi, img_scale):
    """
    Detect abnormal texture = difference between local and smooth surface.
    High-frequency noise that doesn't belong to normal print texture.
    """
    defects = []
    h, w = gray.shape[:2]

    # Expected smooth surface = heavily blurred version
    smooth = cv2.GaussianBlur(gray, (15,15), 0)
    diff   = cv2.absdiff(gray, smooth)

    # Adjust threshold by sensitivity
    tex_thresh = max(8, int(TEXTURE_THRESH * (1 + (sensitivity-30)/100.0)))
    _, th = cv2.threshold(diff, tex_thresh, 255, cv2.THRESH_BINARY)

    # Remove fine print texture (expected)
    # Use opening with medium kernel to keep only large anomalies
    km = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  km)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, km)

    total_noise = int(np.sum(th>0))

    # Find individual noise regions
    min_noise_area = max(50, NOISE_AREA_LIMIT*0.1*(sensitivity/30.0))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_noise_area: continue
        if area > h*w*0.15: continue

        x,y,bw,bh = cv2.boundingRect(cnt)
        a_real = area/(img_scale**2)
        amm2   = a2mm2(a_real, dpi)
        size_str = f'{amm2:.3f}mm²' if amm2 else f'{a_real:.0f}px²'
        v = 'FAIL' if a_real > 200 else 'WARN'

        defects.append({
            'type':'noise','label':'Bất thường/nhiễu',
            'x':int(x),'y':int(y),'w':int(bw),'h':int(bh),
            'area_px':int(area),
            'area_mm2':round(amm2,3) if amm2 else None,
            'size_str':size_str,
            'verdict':v,
            'severity':'high' if v=='FAIL' else 'medium',
        })

    return defects, total_noise


# ── Render outputs ────────────────────────────────────────────────────────────
def render_overlay(img, defects):
    """Color overlay — no text labels."""
    out = img.copy()
    ih, iw = out.shape[:2]
    for d in defects:
        c = DEFECT_COLORS.get(d['type'], (180,180,180))
        x=max(0,d['x']); y=max(0,d['y'])
        w=min(d['w'],iw-x); h=min(d['h'],ih-y)
        if w<=0 or h<=0: continue
        if d['type'] == 'blur':
            # Full-image overlay for blur
            ov = out.copy()
            cv2.rectangle(ov,(0,0),(iw,ih),c,-1)
            cv2.addWeighted(ov,0.15,out,0.85,0,out)
            cv2.rectangle(out,(2,2),(iw-2,ih-2),c,3)
            continue
        ov = out.copy()
        cv2.rectangle(ov,(x,y),(x+w,y+h),c,-1)
        alpha = 0.55 if d.get('verdict')=='FAIL' else 0.38
        cv2.addWeighted(ov,alpha,out,1-alpha,0,out)
        thick = 3 if d.get('verdict')=='FAIL' else 2
        cv2.rectangle(out,(x,y),(x+w,y+h),c,thick)
    return out


def render_dotmap(shape, defects):
    """
    Black background. White = defect location.
    QC sees white spots → knows where to re-inspect.
    """
    ih, iw = shape[:2]
    out = np.zeros((ih,iw,3), dtype=np.uint8)
    for d in defects:
        if d['type'] == 'blur': continue  # blur is global, skip on dotmap
        x=max(0,d['x']); y=max(0,d['y'])
        w=min(d['w'],iw-x); h=min(d['h'],ih-y)
        if w>0 and h>0:
            out[y:y+h, x:x+w] = 255
    return out


# ── Gemini enhancement (optional) ─────────────────────────────────────────────
def gemini_enhance(img, defects):
    """Use Gemini to add context to detected defects."""
    if not GEMINI_API_KEY or not defects: return None
    lines = "\n".join([f"- {d['label']} tại ({d['x']},{d['y']}) size={d['size_str']}"
                       for d in defects[:10]])
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[
                {"inline_data":{"mime_type":"image/jpeg","data":j2b64(img)}},
                {"text": f"Ảnh tờ in có các lỗi sau:\n{lines}\n"
                         "Đánh giá chất lượng tờ in, lỗi nghiêm trọng nhất, khuyến nghị. Tiếng Việt ≤80 từ."}
            ]}],"generationConfig":{"temperature":0.1,"maxOutputTokens":200}},
            timeout=30)
        return (r.json().get('candidates',[{}])[0]
                .get('content',{}).get('parts',[{}])[0].get('text',''))
    except: return None


# ── API ───────────────────────────────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST','OPTIONS'])
def analyze():
    if request.method=='OPTIONS': return '',204
    data = request.get_json()
    if not data: return jsonify({'error':'No data'}),400

    img_b64    = data.get('printImage') or data.get('awImage')
    sensitivity= int(data.get('sensitivity',30))
    man_dpi    = data.get('manualDpi',None)
    img_b64_raw= data.get('awFileData') or data.get('imageFileData')
    ftype      = data.get('awFileType','image')

    if not img_b64: return jsonify({'error':'Thiếu ảnh'}),400

    try:
        # Load image
        img = b64_to_cv2(img_b64)
        if img is None: return jsonify({'error':'Không đọc được ảnh'}),400

        # DPI (optional)
        dpi = float(man_dpi) if man_dpi else None
        if not dpi and img_b64_raw:
            dpi = get_dpi(img_b64_raw, ftype)
        print(f'Analyze: shape={img.shape} DPI={dpi} Sens={sensitivity}')

        # Run pipeline
        result, img_resized = run_pipeline(img, sensitivity, dpi)
        defects = result['defects']
        print(f'Found: {len(defects)} defects | blur={result["blur_detected"]} '
              f'spots={result["spot_count"]} scratches={result["scratch_detected"]}')

        # Render
        col_img = render_overlay(img_resized, defects)
        dot_img = render_dotmap(img_resized.shape, defects)

        # AI summary
        ai_sum = gemini_enhance(img_resized, defects) if defects else None

        # Result
        fail_c = sum(1 for d in defects if d.get('verdict')=='FAIL')
        warn_c = sum(1 for d in defects if d.get('verdict')=='WARN')
        overall = 'NG' if fail_c>0 or result['blur_detected'] else (
                  'WARN' if warn_c>0 or result['scratch_detected'] else 'OK')

        clean = [{k:v for k,v in d.items()} for d in defects]

        return jsonify({
            # Main result
            'result':        overall,           # "OK" / "WARN" / "NG"
            'verdict':       'FAIL' if overall=='NG' else ('WARN' if overall=='WARN' else 'PASS'),
            'defect_count':  len(defects),
            'fail_count':    fail_c,
            'warn_count':    warn_c,
            # Detail
            'spot_count':    result['spot_count'],
            'blur':          result['blur_detected'],
            'scratch':       result['scratch_detected'],
            'noise_area':    result['noise_area_px'],
            'laplacian_var': result['laplacian_var'],
            'defects':       clean,
            # Images
            'result_color':  cv2_to_b64(col_img),
            'result_dotmap': cv2_to_b64(dot_img),
            # Meta
            'ai_summary':    ai_sum,
            'dpi_used':      round(dpi,1) if dpi else None,
            'image_size':    f'{img_resized.shape[1]}x{img_resized.shape[0]}',
            'pipeline':      result['pipeline_steps'],
        })

    except Exception as e:
        import traceback
        return jsonify({'error':str(e),'trace':traceback.format_exc()}),500


# Keep DPI endpoint for compatibility
@app.route('/api/get_aw_info', methods=['POST','OPTIONS'])
def api_dpi():
    if request.method=='OPTIONS': return '',204
    d = request.get_json()
    b64 = d.get('fileData','')
    dpi = get_dpi(b64, d.get('fileType','image'))
    return jsonify({'dpi':dpi,'source':'exif' if dpi else 'not_found'})


@app.route('/', defaults={'path':''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder,path)):
        return send_from_directory(app.static_folder,path)
    return send_from_directory(app.static_folder,'index.html')

if __name__=='__main__':
    port=int(os.environ.get('PORT',8080))
    print('QC Defect Inspector — No AW required')
    print('Detects: spot/dot/hickey/blur/smear/scratch/dirt/noise')
    app.run(host='0.0.0.0',port=port,debug=False)
