"""
QC Structural Context Comparison
=================================
Không so pixel-by-pixel.
So sánh ngữ cảnh không gian (structural context) của từng patch:
  AW: patch tại (x,y) → context [left, right, top, bottom, center, texture, brightness]
  Print: tìm patch tương ứng → so sánh context vector
  Sai context → lỗi thật (không phải false positive do lệch ảnh)
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64, os, math, json, requests
from io import BytesIO
from PIL import Image, ExifTags
from skimage.metrics import structural_similarity as ssim_fn

app = Flask(__name__, static_folder='static')
CORS(app)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# ── Config ────────────────────────────────────────────────────────────────────
PATCH_SIZE    = 32    # px, kích thước mỗi patch để so context
PATCH_STRIDE  = 16    # px, bước nhảy (overlap 50%)
MAX_DIM       = 1200  # px, max dimension sau resize

DEFECT_COLORS_CV = {
    'missing':    (50,  255,  50),   # xanh lá — mất nội dung
    'extra':      (255,  50,  50),   # đỏ — thừa nội dung
    'deformed':   (255, 140,   0),   # cam — biến dạng hình dạng
    'blur':       (255, 220,   0),   # vàng — nhòe
    'spot':       (0,   200, 255),   # xanh dương — đốm/chấm lạ
    'scratch':    (255,  50, 200),   # hồng — vệt xước
    'color_shift':(100, 100, 255),   # tím — lệch màu
    'noise':      (180, 180, 180),   # xám — nhiễu
}
DEFECT_LABELS = {
    'missing':    'Mất nội dung',
    'extra':      'Thừa nội dung',
    'deformed':   'Biến dạng',
    'blur':       'Nhòe/mờ',
    'spot':       'Đốm/chấm lạ',
    'scratch':    'Xước/vệt',
    'color_shift':'Lệch màu',
    'noise':      'Nhiễu bất thường',
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

# ── DPI ───────────────────────────────────────────────────────────────────────
def get_dpi(b64, ftype='image'):
    if ftype == 'pdf': return _pdf_dpi(b64)
    return _img_dpi(b64)

def _pdf_dpi(b64):
    try:
        import fitz
        doc = fitz.open(stream=base64.b64decode(b64), filetype='pdf')
        p = doc[0]; r = p.rect
        wm = r.width/72*25.4; hm = r.height/72*25.4
        dpi = 300
        pix = p.get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72), colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.tobytes('png'), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        doc.close()
        return {'dpi':float(dpi),'width_mm':round(wm,2),'height_mm':round(hm,2),
                'source':f'PDF {wm:.1f}x{hm:.1f}mm@{dpi}dpi','img_cv2':img}
    except Exception as e:
        return {'dpi':None,'source':f'pdf_err:{e}','img_cv2':None}

def _img_dpi(b64):
    try:
        pil = Image.open(BytesIO(base64.b64decode(b64)))
        w,h = pil.size; dpi = None
        if 'dpi' in pil.info: dpi = float(pil.info['dpi'][0])
        if not dpi:
            try:
                ex = pil._getexif()
                if ex:
                    for tid,v in ex.items():
                        if ExifTags.TAGS.get(tid)=='XResolution':
                            dpi=float(v[0]/v[1]) if isinstance(v,tuple) else float(v)
            except: pass
        cv2img = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        if dpi and dpi > 10:
            wm=w/dpi*25.4; hm=h/dpi*25.4
            return {'dpi':dpi,'width_mm':round(wm,2),'height_mm':round(hm,2),
                    'source':f'EXIF {wm:.1f}x{hm:.1f}mm@{dpi:.0f}dpi','img_cv2':cv2img}
        return {'dpi':None,'source':'no_dpi','img_cv2':cv2img}
    except Exception as e:
        return {'dpi':None,'source':f'err:{e}','img_cv2':None}

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: REGISTER — align print to AW space
# ═══════════════════════════════════════════════════════════════════════════════
def register(aw, real):
    """Warp real → AW space using feature matching + homography."""
    h, w = aw.shape[:2]
    real_r = cv2.resize(real, (w,h), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    ag = cv2.cvtColor(aw,     cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(real_r, cv2.COLOR_BGR2GRAY)
    ag_e = clahe.apply(ag)
    rg_e = clahe.apply(rg)

    for name, det, norm in [
        ('SIFT', _make_sift(), cv2.NORM_L2),
        ('ORB',  _make_orb(),  cv2.NORM_HAMMING),
        ('AKAZE',_make_akaze(),cv2.NORM_HAMMING),
    ]:
        if det is None: continue
        result = _try_homo(aw, real_r, ag_e, rg_e, det, norm, w, h, name)
        if result is not None:
            warped, quality = result
            warped = _ecc_refine(ag, warped, w, h)
            return warped, name, quality

    return _ecc_only(ag, real_r, w, h), 'ECC', 0.3

def _make_sift():
    try: return cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.01)
    except: return None

def _make_orb():
    try: return cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=12)
    except: return None

def _make_akaze():
    try: return cv2.AKAZE_create()
    except: return None

def _try_homo(aw, real, ag_e, rg_e, det, norm, w, h, name):
    try:
        kp1,des1 = det.detectAndCompute(ag_e, None)
        kp2,des2 = det.detectAndCompute(rg_e, None)
        if des1 is None or des2 is None: return None
        if len(kp1)<8 or len(kp2)<8: return None

        bf  = cv2.BFMatcher(norm, crossCheck=False)
        raw = bf.knnMatch(des1, des2, k=2)
        good = [m for m,n in raw if m.distance < 0.72*n.distance] if raw and len(raw[0])==2 else []
        if len(good) < 8: return None

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 4.0, maxIters=3000, confidence=0.995)
        if H is None: return None

        inliers = int(mask.sum()) if mask is not None else 0
        ratio   = inliers/len(good)
        if ratio < 0.20 or inliers < 6: return None
        if not _valid_H(H, w, h): return None

        warped = cv2.warpPerspective(real, H, (w,h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        print(f'[{name}] kp={len(kp1)}/{len(kp2)} inliers={inliers}/{len(good)} ratio={ratio:.2f}')
        return warped, min(1.0, ratio*(inliers/30.0))
    except Exception as e:
        print(f'[{name}] error: {e}'); return None

def _valid_H(H, w, h):
    try:
        c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        wc = cv2.perspectiveTransform(c, H).reshape(4,2)
        if cv2.contourArea(wc)/(w*h) < 0.05: return False
        hull = cv2.convexHull(wc.astype(np.float32))
        return len(hull)==4
    except: return False

def _ecc_refine(ag, warped, w, h):
    rg = cv2.cvtColor(cv2.resize(warped,(w,h)), cv2.COLOR_BGR2GRAY)
    wm = np.eye(2,3,dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
    try:
        _, wm = cv2.findTransformECC(ag.astype(np.float32), rg.astype(np.float32),
                                      wm, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        return cv2.warpAffine(warped, wm, (w,h),
                               flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
    except: return warped

def _ecc_only(ag, real, w, h):
    rg = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY).astype(np.float32)
    wm = np.eye(2,3,dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 500, 1e-7)
    try:
        _, wm = cv2.findTransformECC(ag.astype(np.float32), rg, wm,
                                      cv2.MOTION_EUCLIDEAN, crit, None, 5)
        return cv2.warpAffine(real, wm, (w,h),
                               flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
    except: return real

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: EXTRACT CONTEXT VECTOR cho mỗi patch
#  Đây là trọng tâm: không so pixel, so "mô tả ngữ cảnh"
# ═══════════════════════════════════════════════════════════════════════════════

def extract_context(gray, x, y, patch_size):
    """
    Trích xuất vector ngữ cảnh của patch tại (x,y):
    - center: loại vùng (trắng / tối / texture / edge)
    - left/right/top/bottom: loại của 4 vùng lân cận
    - sharpness: độ sắc nét (Laplacian)
    - brightness: độ sáng trung bình
    - texture: entropy local
    - edge_density: mật độ cạnh nét
    - ink_ratio: tỷ lệ pixel tối (có mực)
    """
    H, W = gray.shape
    p = patch_size
    h2 = p // 2

    def get_patch(cx, cy):
        x1 = max(0, cx-h2); y1 = max(0, cy-h2)
        x2 = min(W, cx+h2); y2 = min(H, cy+h2)
        return gray[y1:y2, x1:x2]

    def describe_patch(patch):
        if patch.size == 0:
            return {'bright':128, 'ink':0.5, 'sharp':0, 'edge':0, 'type':2}
        mean  = float(np.mean(patch))
        std   = float(np.std(patch))
        sharp = float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.shape[0]>3 and patch.shape[1]>3 else 0
        edges = cv2.Canny(patch, 30, 100) if patch.shape[0]>3 and patch.shape[1]>3 else np.zeros_like(patch)
        edge_d= float(np.mean(edges>0))
        ink   = float(np.mean(patch < 128))  # ratio of dark pixels

        # Classify patch type (quantized to reduce noise sensitivity)
        if mean > 220:        ptype = 0   # near-white (background)
        elif mean > 180:      ptype = 1   # light gray
        elif ink > 0.4:       ptype = 3   # heavy ink
        elif edge_d > 0.15:   ptype = 4   # edge/text boundary
        elif std > 30:        ptype = 5   # texture
        else:                 ptype = 2   # mid-tone

        return {'bright':round(mean,1), 'ink':round(ink,3),
                'sharp':round(sharp,1), 'edge':round(edge_d,3), 'type':ptype}

    cx = x + h2; cy = y + h2
    ctx = {
        'center': describe_patch(get_patch(cx,     cy)),
        'left':   describe_patch(get_patch(cx-p,   cy)),
        'right':  describe_patch(get_patch(cx+p,   cy)),
        'top':    describe_patch(get_patch(cx,      cy-p)),
        'bottom': describe_patch(get_patch(cx,      cy+p)),
    }
    return ctx


def context_distance(ctx_aw, ctx_pr):
    """
    Tính khoảng cách giữa 2 context vector.
    Trả về score 0..1 (0 = identical, 1 = completely different).
    So LOẠI ngữ cảnh, không so giá trị pixel tuyệt đối.
    """
    total_weight = 0
    total_diff   = 0

    weights = {
        'center': 0.40,   # trọng tâm quan trọng nhất
        'left':   0.15,
        'right':  0.15,
        'top':    0.15,
        'bottom': 0.15,
    }

    for region, w in weights.items():
        a = ctx_aw.get(region, {})
        p = ctx_pr.get(region, {})

        # Type mismatch (quantized) = structural difference
        type_diff = 0 if a.get('type',0)==p.get('type',0) else 1

        # Ink presence difference (normalized)
        ink_diff = abs(a.get('ink',0) - p.get('ink',0))

        # Edge density difference
        edge_diff = abs(a.get('edge',0) - p.get('edge',0))

        # Brightness difference (normalized to 0..1, but less weight)
        bright_diff = abs(a.get('bright',128) - p.get('bright',128)) / 255.0

        # Sharpness ratio (blur detection)
        sa = max(a.get('sharp',0), 0.1)
        sp = max(p.get('sharp',0), 0.1)
        sharp_ratio = min(sa,sp)/max(sa,sp)
        sharp_diff  = 1.0 - sharp_ratio

        # Weighted sum for this region
        region_score = (
            type_diff   * 0.40 +
            ink_diff    * 0.30 +
            edge_diff   * 0.15 +
            bright_diff * 0.10 +
            sharp_diff  * 0.05
        )

        total_diff   += region_score * w
        total_weight += w

    return total_diff / total_weight if total_weight > 0 else 0


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: SCAN tất cả patches — so context AW vs Print
# ═══════════════════════════════════════════════════════════════════════════════

def scan_and_compare(aw_bgr, print_bgr, sensitivity):
    """
    Quét từng patch theo lưới, so context vector.
    Chỉ đánh dấu lỗi khi context THỰC SỰ khác biệt.
    sensitivity: 10→ nhạy, 70→ chỉ lỗi rõ
    Returns: heat_map, binary_mask, patch_results
    """
    h, w = aw_bgr.shape[:2]

    # Normalize brightness trước khi extract context
    aw_g  = cv2.cvtColor(aw_bgr,    cv2.COLOR_BGR2GRAY)
    pr_g  = cv2.cvtColor(print_bgr, cv2.COLOR_BGR2GRAY)

    # Equalize để bù sáng/tối do điều kiện chụp
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    aw_eq  = clahe.apply(aw_g)
    pr_eq  = clahe.apply(pr_g)

    # Blur nhẹ để giảm noise pixel nhưng giữ cấu trúc
    aw_s = cv2.GaussianBlur(aw_eq, (3,3), 0.5)
    pr_s = cv2.GaussianBlur(pr_eq, (3,3), 0.5)

    p  = PATCH_SIZE
    st = PATCH_STRIDE
    heat = np.zeros((h,w), dtype=np.float32)

    # Adaptive threshold: context distance threshold
    # sensitivity 10 → threshold 0.20 (catch small differences)
    # sensitivity 30 → threshold 0.35 (balanced)
    # sensitivity 70 → threshold 0.60 (only obvious defects)
    ctx_thresh = 0.15 + (sensitivity/100.0) * 0.55

    patch_diffs = []

    for y in range(0, h-p, st):
        for x in range(0, w-p, st):
            # Skip patches at image border (often alignment artifacts)
            if x < p//2 or y < p//2 or x+p+p//2 > w or y+p+p//2 > h:
                continue

            ctx_aw = extract_context(aw_s,  x, y, p)
            ctx_pr = extract_context(pr_s, x, y, p)

            dist = context_distance(ctx_aw, ctx_pr)
            if dist > 0:
                heat[y:y+p, x:x+p] = np.maximum(heat[y:y+p, x:x+p], dist)

            if dist > ctx_thresh:
                patch_diffs.append({
                    'x':x,'y':y,'w':p,'h':p,
                    'dist':dist,
                    'ctx_aw':ctx_aw,
                    'ctx_pr':ctx_pr,
                })

    # Smooth heat map
    heat = cv2.GaussianBlur(heat, (7,7), 0)
    heat = np.clip(heat, 0, 1).astype(np.float32)

    # Binary mask from heat
    binary = (heat > ctx_thresh * 0.8).astype(np.uint8) * 255
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  ko)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kc)

    return heat, binary, patch_diffs


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4: CLASSIFY defects từ context diff
# ═══════════════════════════════════════════════════════════════════════════════

def classify_context_defects(binary, heat, aw_bgr, print_bgr, patch_diffs, dpi):
    """
    Dựa vào context diff, phân loại lỗi cụ thể.
    Phân loại dựa trên: loại context thay đổi, hình dạng vùng diff.
    """
    contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defects = []
    H, W = aw_bgr.shape[:2]

    aw_g  = cv2.cvtColor(aw_bgr,    cv2.COLOR_BGR2GRAY)
    pr_g  = cv2.cvtColor(print_bgr, cv2.COLOR_BGR2GRAY)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20: continue
        if area > H*W*0.15: continue  # skip background

        x,y,bw,bh = cv2.boundingRect(cnt)
        cx,cy = x+bw//2, y+bh//2
        aspect = max(bw,bh)/max(min(bw,bh),1)

        # Get patches in this region
        region_patches = [p for p in patch_diffs
                          if x<=p['x']<=x+bw and y<=p['y']<=y+bh]

        # Classify based on context changes
        dtype = _classify_by_context(region_patches, aw_g, pr_g, x, y, bw, bh, aspect)

        # Size
        amm2 = area/(dpi/25.4)**2 if dpi else None
        sz   = f'{amm2:.3f}mm²' if amm2 else f'{area:.0f}px²'

        mk = np.zeros(binary.shape,np.uint8)
        cv2.drawContours(mk,[cnt],-1,255,-1)
        mean_heat = float(cv2.mean(heat,mask=mk)[0])

        # Verdict based on size + severity
        v = 'FAIL' if (amm2 and amm2>0.5) or area>100 else 'WARN'

        defects.append({
            'type':   dtype,
            'label':  DEFECT_LABELS.get(dtype, dtype),
            'x':int(x),'y':int(y),'w':int(bw),'h':int(bh),
            'area_mm2': round(amm2,3) if amm2 else None,
            'size_str': sz,
            'verdict':  v,
            'severity': 'high' if v=='FAIL' else 'medium',
            'heat_score': round(mean_heat,3),
            'context_diff': round(max((p['dist'] for p in region_patches),default=0),3),
        })

    defects.sort(key=lambda d:d['heat_score'],reverse=True)
    return defects


def _classify_by_context(patches, aw_g, pr_g, x, y, bw, bh, aspect):
    """Phân loại lỗi dựa trên loại context thay đổi."""
    if not patches:
        return 'noise'

    # Đếm các loại context thay đổi
    n_ink_increase = 0  # print có nhiều mực hơn AW
    n_ink_decrease = 0  # print thiếu mực so AW
    n_type_change  = 0  # loại vùng thay đổi (trắng→đen, text→blank...)
    n_sharp_drop   = 0  # độ sắc nét giảm → blur

    for p in patches:
        aw_c  = p['ctx_aw']['center']
        pr_c  = p['ctx_pr']['center']
        ink_d = pr_c.get('ink',0) - aw_c.get('ink',0)
        if ink_d >  0.20: n_ink_increase += 1
        if ink_d < -0.20: n_ink_decrease += 1
        if aw_c.get('type',-1) != pr_c.get('type',-1): n_type_change += 1
        sa = max(aw_c.get('sharp',1),1); sp = max(pr_c.get('sharp',1),1)
        if sp < sa*0.4: n_sharp_drop += 1

    total = max(len(patches), 1)

    # Blur: sharp drops predominate
    if n_sharp_drop/total > 0.4:
        return 'blur'

    # Scratch: long thin shape
    if aspect > 6:
        return 'scratch'

    # Missing ink: AW has ink but print doesn't
    if n_ink_decrease/total > 0.35:
        return 'missing'

    # Extra ink: print has ink not in AW (spot/dirt/smear)
    if n_ink_increase/total > 0.35:
        # Small circular → spot; large irregular → extra
        roi_aw = aw_g[y:y+bh, x:x+bw]
        if roi_aw.size > 0:
            aw_mean = float(np.mean(roi_aw))
            if aw_mean > 200 and max(bw,bh) < 40:
                return 'spot'
        return 'extra'

    # Type changed substantially
    if n_type_change/total > 0.50:
        return 'deformed'

    return 'noise'


# ═══════════════════════════════════════════════════════════════════════════════
#  RENDER
# ═══════════════════════════════════════════════════════════════════════════════

def render_overlay(img, defects):
    """Tô màu vùng lỗi — không nhãn chữ."""
    out = img.copy()
    ih,iw = out.shape[:2]
    for d in defects:
        c = DEFECT_COLORS_CV.get(d['type'],(180,180,180))
        x=max(0,d['x']); y=max(0,d['y'])
        w=min(d['w'],iw-x); h=min(d['h'],ih-y)
        if w<=0 or h<=0: continue
        ov=out.copy()
        cv2.rectangle(ov,(x,y),(x+w,y+h),c,-1)
        a=0.55 if d.get('verdict')=='FAIL' else 0.38
        cv2.addWeighted(ov,a,out,1-a,0,out)
        cv2.rectangle(out,(x,y),(x+w,y+h),c,3 if d.get('verdict')=='FAIL' else 2)
    return out


def render_dotmap(shape, defects, heat):
    """Nền đen, vị trí lỗi = trắng."""
    ih,iw = shape[:2]
    out = np.zeros((ih,iw,3),dtype=np.uint8)
    hh,hw = heat.shape[:2]
    rh=min(hh,ih); rw=min(hw,iw)
    if rh>0 and rw>0:
        hr = cv2.resize(heat[:rh,:rw],(rw,rh))
        _,mask = cv2.threshold((hr*255).astype(np.uint8),15,255,cv2.THRESH_BINARY)
        k5=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        out[:rh,:rw][cv2.dilate(mask,k5)>0] = 255
    for d in defects:
        x=max(0,d['x']); y=max(0,d['y'])
        w=min(d['w'],iw-x); h=min(d['h'],ih-y)
        if w>0 and h>0: out[y:y+h,x:x+w]=255
    return out


# ── Gemini text check ─────────────────────────────────────────────────────────
def text_check(aw, real):
    if not GEMINI_API_KEY: return []
    h,w = aw.shape[:2]
    prompt=(f"So sánh Ảnh 1 (AW gốc) và Ảnh 2 (tờ in, {w}x{h}px). "
            "Chỉ kiểm tra VĂN BẢN: dấu thanh tiếng Việt, dấu câu, ký tự sai/thiếu. "
            'JSON: {"text_defects":[{"type":"wrong_diacritic|missing_punct|wrong_char",'
            '"label":"tên lỗi","detail":"mô tả chi tiết","x":0,"y":0,"w":40,"h":25}]}')
    try:
        r=requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[
                {"inline_data":{"mime_type":"image/jpeg","data":j2b64(aw)}},
                {"inline_data":{"mime_type":"image/jpeg","data":j2b64(real)}},
                {"text":prompt}
            ]}],"generationConfig":{"temperature":0.05,"maxOutputTokens":1500}},
            timeout=45)
        raw=(r.json().get('candidates',[{}])[0]
             .get('content',{}).get('parts',[{}])[0].get('text',''))
        return json.loads(raw.replace('```json','').replace('```','').strip()).get('text_defects',[])
    except Exception as e:
        print(f'Gemini err:{e}'); return []

def ai_summary(defects):
    if not GEMINI_API_KEY or not defects: return None
    lines="\n".join([f"- #{i+1}: {d.get('label','?')} {d.get('size_str','')} [{d.get('verdict','?')}] ctx_diff={d.get('context_diff','?')}"
                     for i,d in enumerate(defects[:12])])
    try:
        r=requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[{"text":
                f"QC in ấn bao bì. {len(defects)} lỗi:\n{lines}\nBáo cáo tiếng Việt ≤80 từ."}]}],
                "generationConfig":{"temperature":0.2,"maxOutputTokens":200}},
            timeout=30)
        return (r.json().get('candidates',[{}])[0]
                .get('content',{}).get('parts',[{}])[0].get('text',''))
    except: return None

# ── API ───────────────────────────────────────────────────────────────────────
@app.route('/api/get_aw_info', methods=['POST','OPTIONS'])
def api_dpi():
    if request.method=='OPTIONS': return '',204
    d=request.get_json()
    info=get_dpi(d.get('fileData',''),d.get('fileType','image'))
    return jsonify({k:v for k,v in info.items() if k!='img_cv2'})


@app.route('/api/analyze', methods=['POST','OPTIONS'])
def analyze():
    if request.method=='OPTIONS': return '',204
    data=request.get_json()
    if not data: return jsonify({'error':'No data'}),400

    aw_b64    = data.get('awImage')
    pr_b64    = data.get('printImage')
    aw_fb64   = data.get('awFileData')
    aw_ftype  = data.get('awFileType','image')
    sens      = int(data.get('sensitivity',30))
    chk_txt   = data.get('checkText',True)
    man_dpi   = data.get('manualDpi',None)

    if not aw_b64 or not pr_b64:
        return jsonify({'error':'Thiếu ảnh'}),400

    try:
        # DPI
        aw_info={}
        if aw_fb64: aw_info=get_dpi(aw_fb64,aw_ftype)
        dpi=float(man_dpi) if man_dpi else (aw_info.get('dpi') or 150.0)

        # Load + resize
        aw = b64_to_cv2(aw_b64)
        pr = b64_to_cv2(pr_b64)
        if aw is None or pr is None:
            return jsonify({'error':'Không đọc được ảnh'}),400

        # Resize to max dim
        h,w=aw.shape[:2]
        sc=min(MAX_DIM/max(h,w),1.0)
        if sc<1.0:
            aw=cv2.resize(aw,(int(w*sc),int(h*sc)),interpolation=cv2.INTER_AREA)
            dpi*=sc
        print(f'AW={aw.shape} Real={pr.shape} DPI={dpi:.0f} Sens={sens}')

        # STEP 1: Register
        warped, reg_method, reg_q = register(aw, pr)
        print(f'Register: {reg_method} quality={reg_q:.2f}')

        # Brightness normalize
        aw_mean=float(np.mean(cv2.cvtColor(aw,cv2.COLOR_BGR2GRAY)))
        pr_mean=float(np.mean(cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)))
        scale=1.0; offset=aw_mean-pr_mean
        warped_n=np.clip(warped.astype(np.float32)+offset,0,255).astype(np.uint8)

        # STEP 2-3: Context scan
        heat, binary, patch_diffs = scan_and_compare(aw, warped_n, sens)
        n_diff=int(np.sum(binary>0))
        print(f'Context scan: {len(patch_diffs)} diff patches | binary={n_diff}px')

        # STEP 4: Classify defects
        defects = classify_context_defects(binary, heat, aw, warped_n, patch_diffs, dpi)
        print(f'Defects: {len(defects)}')

        # Text check
        txt_def=[]
        if chk_txt and GEMINI_API_KEY:
            txt_def=text_check(aw,warped_n)
            for td in txt_def:
                td['verdict']='FAIL'; td['size_str']=td.get('detail','')

        # Render
        col_img=render_overlay(warped_n, defects+[dict(d,type='spot') for d in txt_def])
        dot_img=render_dotmap(warped_n.shape, defects+txt_def, heat)

        all_d=defects+[dict(d,severity='high') for d in txt_def]
        summ=ai_summary(all_d) if all_d else None

        phys_out=[{k:v for k,v in d.items()} for d in defects]
        txt_out=[dict(d,is_text=True) for d in txt_def]

        fail_c=sum(1 for d in all_d if d.get('verdict')=='FAIL')
        warn_c=sum(1 for d in all_d if d.get('verdict')=='WARN')
        v='FAIL' if fail_c>0 else ('WARN' if warn_c>0 else 'PASS')

        return jsonify({
            'verdict':v,'defect_count':len(all_d),
            'fail_count':fail_c,'warn_count':warn_c,
            'physical_count':len(defects),'text_count':len(txt_def),
            'defects':phys_out+txt_out,
            'result_color':  cv2_to_b64(col_img),
            'result_dotmap': cv2_to_b64(dot_img),
            'ai_summary':    summ,
            'dpi_aw':        round(dpi,1),
            'aw_size':       f"{aw_info.get('width_mm','?')}x{aw_info.get('height_mm','?')}mm",
            'dpi_source':    aw_info.get('source','fallback'),
            'align_method':  reg_method,
            'reg_quality':   round(reg_q,3),
            'diff_patches':  len(patch_diffs),
        })

    except Exception as e:
        import traceback
        return jsonify({'error':str(e),'trace':traceback.format_exc()}),500


@app.route('/', defaults={'path':''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder,path)):
        return send_from_directory(app.static_folder,path)
    return send_from_directory(app.static_folder,'index.html')

if __name__=='__main__':
    port=int(os.environ.get('PORT',8080))
    print('QC Structural Context Inspector')
    app.run(host='0.0.0.0',port=port,debug=False)
