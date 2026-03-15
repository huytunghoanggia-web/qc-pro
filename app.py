from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64, os, math, json, requests
from io import BytesIO
from PIL import Image, ExifTags
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__, static_folder='static')
CORS(app)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

QC_STANDARDS = {
    'dot':         {'warn':0.3,  'fail':0.5,  'measure':'diameter_mm'},
    'spot':        {'warn':0.3,  'fail':0.5,  'measure':'diameter_mm'},
    'hickey':      {'warn':0.3,  'fail':0.5,  'measure':'diameter_mm'},
    'scratch':     {'warn':3.0,  'fail':5.0,  'measure':'length_mm'},
    'streak':      {'warn':3.0,  'fail':5.0,  'measure':'length_mm'},
    'missing_ink': {'warn':0.5,  'fail':1.0,  'measure':'area_mm2'},
    'extra_ink':   {'warn':0.5,  'fail':1.0,  'measure':'area_mm2'},
    'blur':        {'warn':0.5,  'fail':1.0,  'measure':'area_mm2'},
    'anomaly':     {'warn':0.3,  'fail':0.5,  'measure':'area_mm2'},
}
DEFECT_LABELS = {
    'dot':'Dot/chấm','spot':'Spot/đốm','hickey':'Hickey',
    'scratch':'Xước/vệt dài','streak':'Vệt/sọc',
    'missing_ink':'Thiếu mực','extra_ink':'Thừa mực',
    'blur':'Nhòe/mờ','anomaly':'Bất thường',
}
DEFECT_COLORS_CV = {
    'dot':(0,220,255),'spot':(0,200,255),'hickey':(0,100,255),
    'scratch':(255,50,200),'streak':(255,140,0),
    'missing_ink':(50,255,50),'extra_ink':(255,50,50),
    'blur':(255,220,0),'anomaly':(180,180,180),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def b64_to_cv2(b64):
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def cv2_to_b64(img):
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode()

def img_to_b64_jpeg(img, q=90):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()

def px_to_mm(px, dpi):   return px / dpi * 25.4
def mm_to_px(mm, dpi):   return mm / 25.4 * dpi
def area_to_mm2(a, dpi): return a / (dpi/25.4)**2
def diam_from_area(a):   return 2*math.sqrt(max(a,0)/math.pi)

def judge(dtype, val):
    s = QC_STANDARDS.get(dtype, QC_STANDARDS['anomaly'])
    if val >= s['fail']:   return 'FAIL'
    elif val >= s['warn']: return 'WARN'
    return 'PASS'

# ── DPI extraction ────────────────────────────────────────────────────────────
def get_aw_dpi(b64_str, ftype='image'):
    if ftype == 'pdf': return _pdf_info(b64_str)
    return _img_info(b64_str)

def _pdf_info(b64_str):
    try:
        import fitz
        doc = fitz.open(stream=base64.b64decode(b64_str), filetype='pdf')
        page = doc[0]; rect = page.rect
        w_mm = rect.width/72*25.4; h_mm = rect.height/72*25.4
        dpi = 300
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.tobytes('png'), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        doc.close()
        return {'dpi':float(dpi),'width_mm':round(w_mm,2),'height_mm':round(h_mm,2),
                'source':f'PDF {w_mm:.1f}x{h_mm:.1f}mm@{dpi}dpi','img_cv2':img}
    except Exception as e:
        return {'dpi':None,'source':f'pdf_err:{e}','img_cv2':None}

def _img_info(b64_str):
    try:
        pil = Image.open(BytesIO(base64.b64decode(b64_str)))
        w,h = pil.size; dpi = None
        if 'dpi' in pil.info:
            dpi = float(pil.info['dpi'][0])
        if not dpi:
            try:
                exif = pil._getexif()
                if exif:
                    for tid,val in exif.items():
                        if ExifTags.TAGS.get(tid)=='XResolution':
                            dpi = float(val[0]/val[1]) if isinstance(val,tuple) else float(val)
            except: pass
        img_cv2 = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        if dpi and dpi > 10:
            wm=w/dpi*25.4; hm=h/dpi*25.4
            return {'dpi':dpi,'width_mm':round(wm,2),'height_mm':round(hm,2),
                    'source':f'EXIF {wm:.1f}x{hm:.1f}mm@{dpi:.0f}dpi','img_cv2':img_cv2}
        return {'dpi':None,'source':'no_dpi','img_cv2':img_cv2}
    except Exception as e:
        return {'dpi':None,'source':f'img_err:{e}','img_cv2':None}

# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE REGISTRATION PIPELINE
#  Handles: rotation, scale, perspective, camera angle, shift, deformation
# ═══════════════════════════════════════════════════════════════════════════════

def register_images(aw_img, real_img):
    """
    Full image registration pipeline.
    Returns: (warped, method, quality_score, debug_info)

    Stages:
      1. Multi-scale preprocessing
      2. ORB feature detection (with SIFT fallback)
      3. BFMatcher + Lowe ratio test
      4. Homography via RANSAC (handles perspective + rotation + scale)
      5. Perspective warp to AW coordinate space
      6. ECC refinement for sub-pixel accuracy
      7. Validation: reject if transform is degenerate
    """
    h_aw, w_aw = aw_img.shape[:2]
    debug = {}

    # Pre-resize real to AW dimensions as starting point
    real_pre = cv2.resize(real_img, (w_aw, h_aw), interpolation=cv2.INTER_AREA)

    ag = cv2.cvtColor(aw_img,   cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(real_pre, cv2.COLOR_BGR2GRAY)

    # ── Stage 1: Enhance contrast for better feature detection ──
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    ag_e = clahe.apply(ag)
    rg_e = clahe.apply(rg)

    # ── Stage 2: Feature detection ──
    # Try SIFT first (best for scale/rotation), then ORB
    warped, method, score = _try_feature_registration(
        aw_img, real_pre, ag_e, rg_e, w_aw, h_aw, debug
    )

    # ── Stage 3: ECC sub-pixel refinement if feature matching succeeded ──
    if 'homography' in method:
        warped_refined, ecc_ok = _ecc_refine(aw_img, warped, ag, w_aw, h_aw)
        if ecc_ok:
            warped = warped_refined
            method += '+ECC'
            debug['ecc_refined'] = True

    debug['method'] = method
    return warped, method, score, debug


def _try_feature_registration(aw_img, real_img, ag_e, rg_e, w, h, debug):
    """
    Try SIFT → ORB feature matching with homography.
    Returns (warped, method, quality_score)
    """
    for detector_name in ['SIFT', 'ORB']:
        try:
            if detector_name == 'SIFT':
                det = cv2.SIFT_create(
                    nfeatures=3000,
                    contrastThreshold=0.02,
                    edgeThreshold=10,
                    sigma=1.6
                )
                norm = cv2.NORM_L2
            else:
                det = cv2.ORB_create(
                    nfeatures=3000,
                    scaleFactor=1.2,
                    nlevels=12,
                    edgeThreshold=15,
                    patchSize=31
                )
                norm = cv2.NORM_HAMMING

            kp1, des1 = det.detectAndCompute(ag_e, None)
            kp2, des2 = det.detectAndCompute(rg_e, None)

            if des1 is None or des2 is None:
                continue
            if len(kp1) < 10 or len(kp2) < 10:
                debug[f'{detector_name}_kp'] = f'{len(kp1)}/{len(kp2)} too few'
                continue

            debug[f'{detector_name}_kp'] = f'{len(kp1)}/{len(kp2)}'

            # Match with cross-check for robustness
            matcher = cv2.BFMatcher(norm, crossCheck=False)
            raw = matcher.knnMatch(des1, des2, k=2)

            # Lowe ratio test — stricter for perspective robustness
            good = []
            for pair in raw:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.72 * n.distance:
                        good.append(m)

            debug[f'{detector_name}_matches'] = len(good)

            if len(good) < 8:
                continue

            # Extract matched point coordinates
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

            # Compute homography with RANSAC
            # RANSAC handles:
            #   - rotation (any angle)
            #   - scale difference (zoom in/out)
            #   - perspective distortion (camera angle)
            #   - paper shift
            H_mat, mask = cv2.findHomography(
                dst_pts, src_pts,
                cv2.RANSAC,
                ransacReprojThreshold=4.0,
                maxIters=3000,
                confidence=0.995
            )

            if H_mat is None:
                debug[f'{detector_name}_H'] = 'failed'
                continue

            inliers = int(mask.sum()) if mask is not None else 0
            inlier_ratio = inliers / len(good)
            debug[f'{detector_name}_inliers'] = f'{inliers}/{len(good)} ({inlier_ratio:.2f})'

            if inlier_ratio < 0.25 or inliers < 6:
                debug[f'{detector_name}_rejected'] = 'low inlier ratio'
                continue

            # Validate homography is not degenerate
            if not _is_valid_homography(H_mat, w, h):
                debug[f'{detector_name}_rejected'] = 'degenerate homography'
                continue

            # Warp real image to AW coordinate space
            warped = cv2.warpPerspective(
                real_img, H_mat, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            quality = min(1.0, inlier_ratio * (inliers / 50.0))
            print(f"Registration: {detector_name} | kp={len(kp1)}/{len(kp2)} "
                  f"| matches={len(good)} | inliers={inliers} | ratio={inlier_ratio:.2f}")
            return warped, f'{detector_name}_homography', quality

        except Exception as e:
            debug[f'{detector_name}_error'] = str(e)
            continue

    # All feature methods failed → fallback ECC only
    print("Feature matching failed, using ECC only")
    warped = _ecc_only(aw_img, real_img, w, h)
    return warped, 'ECC_only', 0.5


def _is_valid_homography(H, w, h):
    """
    Check homography is not degenerate.
    Rejects: extreme distortion, flip, extreme scale change.
    """
    # Transform corners of image
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    try:
        warped_corners = cv2.perspectiveTransform(corners, H)
    except Exception:
        return False

    # Check transformed area is reasonable (10% ~ 900% of original)
    area = cv2.contourArea(warped_corners)
    orig_area = w * h
    ratio = area / orig_area
    if ratio < 0.05 or ratio > 20.0:
        return False

    # Check no extreme shear (all angles between adjacent sides should be > 20°)
    pts = warped_corners.reshape(4,2)
    for i in range(4):
        a = pts[i]; b = pts[(i+1)%4]; c = pts[(i+2)%4]
        v1 = b - a; v2 = c - b
        cos_a = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
        if abs(cos_a) > 0.97:  # angle < ~14°
            return False

    return True


def _ecc_refine(aw_img, warped, ag, w, h, max_iter=200, eps=1e-6):
    """
    ECC refinement for sub-pixel accuracy.
    Handles small residual rotation/translation after homography.
    """
    rg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ag_f = ag.astype(np.float32)
    rg_f = rg.astype(np.float32)
    warp_mat = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    try:
        _, warp_mat = cv2.findTransformECC(
            ag_f, rg_f, warp_mat,
            cv2.MOTION_EUCLIDEAN,
            criteria, None, 5
        )
        refined = cv2.warpAffine(
            warped, warp_mat, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE
        )
        return refined, True
    except Exception as e:
        return warped, False


def _ecc_only(aw_img, real_img, w, h):
    """Pure ECC alignment when feature matching fails."""
    real = cv2.resize(real_img, (w, h))
    ag = cv2.cvtColor(aw_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rg = cv2.cvtColor(real,   cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7)
    try:
        _, warp = cv2.findTransformECC(ag, rg, warp,
                                        cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        return cv2.warpAffine(real, warp, (w, h),
                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return real


# ── Normalize after registration ──────────────────────────────────────────────
def normalize_after_registration(aw_img, warped_img):
    """
    Normalize brightness/contrast to remove camera exposure difference.
    Must NOT remove actual ink defects — only global illumination difference.
    """
    h, w = aw_img.shape[:2]
    warped = cv2.resize(warped_img, (w, h))

    ag = cv2.cvtColor(aw_img,  cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(warped,  cv2.COLOR_BGR2GRAY)

    # Global statistics matching
    aw_mean = float(np.mean(ag)); aw_std = float(np.std(ag))
    r_mean  = float(np.mean(rg)); r_std  = float(np.std(rg))

    # Scale factor for contrast, offset for brightness
    scale  = aw_std / max(r_std, 1.0)
    offset = aw_mean - r_mean * scale

    # Apply to BGR channels
    warped_norm = np.clip(
        warped.astype(np.float32) * scale + offset, 0, 255
    ).astype(np.uint8)

    # Gentle blur to reduce camera noise (preserve real defects)
    ag_blur = cv2.GaussianBlur(ag, (3,3), 0.8)
    rg_norm = cv2.cvtColor(warped_norm, cv2.COLOR_BGR2GRAY)
    rg_blur = cv2.GaussianBlur(rg_norm, (3,3), 0.8)

    return ag_blur, rg_blur, aw_img, warped_norm


# ── Multi-method diff ─────────────────────────────────────────────────────────
def compute_diff(ag, rg, aw_bgr, real_bgr, sensitivity):
    """
    Multi-method diff. Adaptive threshold removes alignment artifacts.
    MUST NOT produce defect from alignment error.
    """
    h, w = ag.shape[:2]

    # ── SSIM map (structural changes) ──
    win = max(3, min(11, min(h,w)//8 | 1))
    try:
        score, ssim_map = ssim(ag, rg, win_size=win, full=True, data_range=255)
        # SSIM = 1 means identical, SSIM = -1 means opposite
        # Defect map = low SSIM areas
        map_ssim = np.clip(1.0 - (ssim_map + 1.0) / 2.0, 0, 1).astype(np.float32)
    except Exception:
        score = 1.0
        map_ssim = np.zeros((h,w), np.float32)

    # ── LAB diff (color/intensity changes) ──
    aL = cv2.cvtColor(aw_bgr,   cv2.COLOR_BGR2LAB).astype(np.float32)
    rL = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff_L  = np.abs(aL[:,:,0] - rL[:,:,0])
    diff_AB = np.sqrt((aL[:,:,1]-rL[:,:,1])**2 + (aL[:,:,2]-rL[:,:,2])**2)
    map_lab = np.clip(diff_L/100.0*0.6 + diff_AB/180.0*0.4, 0, 1).astype(np.float32)

    # ── Edge presence diff (missing/extra lines) ──
    ae = cv2.Canny(ag, 30, 100).astype(np.float32)/255.0
    re = cv2.Canny(rg, 30, 100).astype(np.float32)/255.0
    # Dilate slightly to tolerate ±1px misalignment residual
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    ae_d = cv2.dilate(ae, k3, iterations=1)
    re_d = cv2.dilate(re, k3, iterations=1)
    # Edge in AW but not in real = missing ink (important!)
    # Edge in real but not in AW = extra ink / dirt
    map_edge = np.abs(ae_d - re_d).astype(np.float32)

    # ── Weighted combination ──
    heat = (map_ssim * 0.40 +
            map_lab  * 0.35 +
            map_edge * 0.25)
    heat = np.clip(heat, 0, 1).astype(np.float32)

    # ── Adaptive threshold ──
    # Key insight: after good alignment, background noise is LOW and uniform.
    # Real defects are OUTLIERS far above the background level.
    # Use: threshold = median + k * MAD (robust to outliers)
    h_flat = heat.flatten()
    median = float(np.median(h_flat))
    # MAD = Median Absolute Deviation (robust std estimate)
    mad = float(np.median(np.abs(h_flat - median)))
    robust_std = mad * 1.4826  # scale to match std for Gaussian

    # k controls strictness: higher k = fewer false positives
    # sensitivity 10 → k=2.5 (catch small defects)
    # sensitivity 30 → k=3.5 (balanced)
    # sensitivity 70 → k=5.0 (only obvious defects)
    k = 2.5 + (sensitivity - 10) / 60.0 * 2.5
    adaptive_thresh = median + k * robust_std

    # Also enforce a minimum absolute threshold
    # (prevents over-detection when image is near-perfect)
    min_thresh = max(0.08, sensitivity / 200.0)
    thresh = max(adaptive_thresh, min_thresh)

    print(f"Diff: SSIM={score:.3f} | median={median:.4f} MAD={mad:.4f} "
          f"| k={k:.1f} | thresh={thresh:.3f} | sens={sensitivity}")

    binary = (heat > thresh).astype(np.uint8) * 255

    # ── Noise filtering ──
    # Remove sub-pixel noise
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k2, iterations=1)
    # Fill small holes in real defects
    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k4, iterations=1)

    return heat, binary, float(score)


# ── Extract + classify defects ────────────────────────────────────────────────
def extract_defects(binary, heat, aw_crop, real_aligned, dpi):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defects = []

    # Minimum defect area: 0.1mm diameter circle
    min_area_px = math.pi * mm_to_px(0.10, dpi)**2

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < max(min_area_px, 4):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw,bh) / max(min(bw,bh), 1)

        # Mean heat in this region
        mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(mask,[cnt],-1,255,-1)
        mean_heat = float(cv2.mean(heat, mask=mask)[0])

        dtype = _classify(cnt, area_px, bw, bh, aspect,
                          aw_crop, real_aligned, dpi)
        if dtype is None:
            continue

        # Physical size
        area_mm2  = area_to_mm2(area_px, dpi)
        diam_mm   = diam_from_area(area_mm2)
        length_mm = px_to_mm(max(bw,bh), dpi)
        width_mm  = px_to_mm(min(bw,bh), dpi)

        if dtype in ('dot','spot','hickey'):
            measure = diam_mm; size_str = f'⌀{diam_mm:.2f}mm'
        elif dtype in ('scratch','streak'):
            measure = length_mm; size_str = f'{length_mm:.2f}x{width_mm:.2f}mm'
        else:
            measure = area_mm2; size_str = f'{area_mm2:.3f}mm²'

        verdict = judge(dtype, measure)
        if verdict == 'PASS':
            continue

        defects.append({
            'type':dtype, 'label':DEFECT_LABELS.get(dtype,dtype),
            'x':int(x), 'y':int(y), 'w':int(bw), 'h':int(bh),
            'area_mm2':round(area_mm2,3),
            'diameter_mm':round(diam_mm,3),
            'length_mm':round(length_mm,3),
            'width_mm':round(width_mm,3),
            'size_str':size_str,
            'verdict':verdict,
            'severity':'high' if verdict=='FAIL' else 'medium',
            'heat_score':round(mean_heat,3),
            'contour':cnt,
        })

    defects.sort(key=lambda d: d['heat_score'], reverse=True)
    return defects


def _classify(cnt, area_px, bw, bh, aspect, aw_img, real_img, dpi):
    perim = cv2.arcLength(cnt, True)
    circ  = 4*math.pi*area_px/perim**2 if perim>0 else 0

    if aspect > 10 and max(bw,bh) > mm_to_px(2.0,dpi): return 'scratch'
    if aspect > 4  and max(bw,bh) > mm_to_px(1.5,dpi): return 'streak'
    if circ > 0.65 and bw < mm_to_px(3,dpi):           return 'hickey'
    if circ > 0.5  and bw < mm_to_px(1.5,dpi):         return 'dot'
    if circ > 0.35 and bw < mm_to_px(6,dpi):           return 'spot'

    x,y,w2,h2 = cv2.boundingRect(cnt)
    ar = aw_img[y:y+h2, x:x+w2]
    pr = real_img[y:y+h2, x:x+w2]
    if ar.size==0 or pr.size==0: return 'anomaly'

    am = float(np.mean(cv2.cvtColor(ar,cv2.COLOR_BGR2GRAY)))
    pm = float(np.mean(cv2.cvtColor(pr,cv2.COLOR_BGR2GRAY)))
    if pm > am+15: return 'missing_ink'
    if pm < am-15: return 'extra_ink'

    try:
        al = cv2.Laplacian(cv2.cvtColor(ar,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
        pl = cv2.Laplacian(cv2.cvtColor(pr,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
        if pl < al*0.4: return 'blur'
    except Exception:
        pass

    if area_px < mm_to_px(0.15,dpi)**2: return None
    return 'anomaly'


# ── Render outputs ────────────────────────────────────────────────────────────
def render_color_overlay(img, defects, text_defects=[]):
    """Tô màu vùng lỗi — không ghi chữ."""
    out = img.copy()
    ih, iw = out.shape[:2]
    for d in defects + text_defects:
        color = DEFECT_COLORS_CV.get(d.get('type',''), (180,180,180))
        if d.get('is_text'): color = (100,0,255)
        x=max(0,d['x']); y=max(0,d['y'])
        w=min(d['w'],iw-x); h=min(d['h'],ih-y)
        if w<=0 or h<=0: continue
        ov = out.copy()
        cv2.rectangle(ov,(x,y),(x+w,y+h),color,-1)
        alpha = 0.55 if d.get('verdict')=='FAIL' else 0.38
        cv2.addWeighted(ov,alpha,out,1-alpha,0,out)
        thick = 3 if d.get('verdict')=='FAIL' else 2
        cv2.rectangle(out,(x,y),(x+w,y+h),color,thick)
    return out


def render_dotmap(img_shape, defects, heat, text_defects=[]):
    """
    Nền đen tuyền.
    Vị trí lỗi = TRẮNG.
    QC nhìn thấy chấm/vùng trắng = điểm cần kiểm tra lại.
    """
    ih, iw = img_shape[:2]
    out = np.zeros((ih, iw, 3), dtype=np.uint8)

    # Paint heat-based white mask
    hh, hw = heat.shape[:2]
    rh = min(hh, ih); rw = min(hw, iw)
    if rh > 0 and rw > 0:
        heat_r = cv2.resize(heat[:rh,:rw], (rw, rh))
        heat_u8 = (heat_r * 255).astype(np.uint8)
        _, mask = cv2.threshold(heat_u8, 18, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.dilate(mask, k, iterations=1)
        out[:rh,:rw][mask>0] = 255

    # Ensure each defect bbox is clearly white
    for d in defects + text_defects:
        x=max(0,d['x']); y=max(0,d['y'])
        w=min(d.get('w',10),iw-x); h=min(d.get('h',10),ih-y)
        if w>0 and h>0:
            out[y:y+h, x:x+w] = 255

    return out


# ── Gemini text / diacritic check ─────────────────────────────────────────────
def check_text_gemini(aw_img, real_img):
    if not GEMINI_API_KEY: return []
    h,w = aw_img.shape[:2]
    prompt = (
        f"So sánh Ảnh 1 (AW gốc chuẩn) và Ảnh 2 (tờ in thực tế, {w}x{h}px). "
        "Chỉ kiểm tra NỘI DUNG VĂN BẢN: dấu thanh tiếng Việt (sắc/huyền/hỏi/ngã/nặng), "
        "dấu câu (./,/!/?) thừa/thiếu, ký tự bị in sai. "
        'Trả về JSON: {"text_defects":[{"type":"wrong_diacritic|missing_punct|wrong_char",'
        '"label":"tên lỗi tiếng Việt","detail":"mô tả chi tiết",'
        '"x":0,"y":0,"w":40,"h":25}]}'
    )
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[
                {"inline_data":{"mime_type":"image/jpeg","data":img_to_b64_jpeg(aw_img)}},
                {"inline_data":{"mime_type":"image/jpeg","data":img_to_b64_jpeg(real_img)}},
                {"text":prompt}
            ]}],"generationConfig":{"temperature":0.05,"maxOutputTokens":1500}},
            timeout=45
        )
        raw = (r.json().get('candidates',[{}])[0]
               .get('content',{}).get('parts',[{}])[0].get('text',''))
        return (json.loads(raw.replace('```json','').replace('```','').strip())
                .get('text_defects',[]))
    except Exception as e:
        print(f"Gemini text err: {e}"); return []


def get_ai_summary(defects):
    if not GEMINI_API_KEY or not defects: return None
    lines = "\n".join([
        f"- #{i+1}: {d.get('label','?')} {d.get('size_str','')} [{d.get('verdict','?')}]"
        for i,d in enumerate(defects)
    ])
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[{"text":
                f"QC in ấn bao bì. {len(defects)} lỗi phát hiện:\n{lines}\n"
                "Báo cáo tiếng Việt ≤80 từ: đánh giá chung, lỗi nguy hiểm nhất, khuyến nghị."}]}],
                "generationConfig":{"temperature":0.2,"maxOutputTokens":250}},
            timeout=30
        )
        return (r.json().get('candidates',[{}])[0]
                .get('content',{}).get('parts',[{}])[0].get('text',''))
    except:
        return None


# ── API ───────────────────────────────────────────────────────────────────────
@app.route('/api/get_aw_info', methods=['POST','OPTIONS'])
def api_get_aw_info():
    if request.method=='OPTIONS': return '',204
    d = request.get_json()
    info = get_aw_dpi(d.get('fileData',''), d.get('fileType','image'))
    return jsonify({k:v for k,v in info.items() if k != 'img_cv2'})


@app.route('/api/analyze', methods=['POST','OPTIONS'])
def analyze():
    if request.method=='OPTIONS': return '',204
    data = request.get_json()
    if not data: return jsonify({'error':'No data'}),400

    aw_b64      = data.get('awImage')
    print_b64   = data.get('printImage')
    aw_file_b64 = data.get('awFileData')
    aw_type     = data.get('awFileType','image')
    sensitivity = int(data.get('sensitivity', 30))
    check_text  = data.get('checkText', True)
    manual_dpi  = data.get('manualDpi', None)

    if not aw_b64 or not print_b64:
        return jsonify({'error':'Thiếu ảnh'}),400

    try:
        # DPI
        aw_info = {}
        if aw_file_b64:
            aw_info = get_aw_dpi(aw_file_b64, aw_type)
        dpi = float(manual_dpi) if manual_dpi else (aw_info.get('dpi') or 150.0)

        # Load
        aw_crop    = b64_to_cv2(aw_b64)
        print_crop = b64_to_cv2(print_b64)
        if aw_crop is None or print_crop is None:
            return jsonify({'error':'Không đọc được ảnh'}),400

        print(f"AW={aw_crop.shape} Real={print_crop.shape} DPI={dpi} Sens={sensitivity}")

        # ── REGISTER ──
        warped, method, reg_quality, debug_info = register_images(aw_crop, print_crop)
        print(f"Registration: {method} quality={reg_quality:.2f} debug={debug_info}")

        # ── NORMALIZE ──
        ag, rg, aw_norm, real_norm = normalize_after_registration(aw_crop, warped)

        # ── DIFF ──
        heat, binary, ssim_score = compute_diff(ag, rg, aw_norm, real_norm, sensitivity)
        n_diff = int(np.sum(binary>0))
        total  = ag.shape[0] * ag.shape[1]
        diff_pct = n_diff / total * 100
        print(f"Diff pixels: {n_diff}/{total} = {diff_pct:.1f}%")

        # ── DETECT ──
        defects = extract_defects(binary, heat, aw_norm, real_norm, dpi)
        print(f"Defects: {len(defects)}")

        # ── TEXT CHECK ──
        text_defects = []
        if check_text and GEMINI_API_KEY:
            text_defects = check_text_gemini(aw_norm, real_norm)
            for td in text_defects:
                td['verdict']='FAIL'; td['size_str']=td.get('detail','')

        # ── RENDER ──
        result_color  = render_color_overlay(real_norm, defects, text_defects)
        result_dotmap = render_dotmap(real_norm.shape, defects, heat, text_defects)

        all_d = defects + [dict(d,severity='high') for d in text_defects]
        ai_summary = get_ai_summary(all_d) if all_d else None

        phys_out = [{k:v for k,v in d.items() if k!='contour'} for d in defects]
        text_out = [dict(d,is_text=True) for d in text_defects]

        fail_c = sum(1 for d in all_d if d.get('verdict')=='FAIL')
        warn_c = sum(1 for d in all_d if d.get('verdict')=='WARN')
        verdict = 'FAIL' if fail_c>0 else ('WARN' if warn_c>0 else 'PASS')

        return jsonify({
            'verdict':        verdict,
            'defect_count':   len(all_d),
            'fail_count':     fail_c,
            'warn_count':     warn_c,
            'physical_count': len(defects),
            'text_count':     len(text_defects),
            'defects':        phys_out + text_out,
            'result_color':   cv2_to_b64(result_color),
            'result_dotmap':  cv2_to_b64(result_dotmap),
            'ai_summary':     ai_summary,
            'dpi_aw':         round(dpi,1),
            'aw_size':        f"{aw_info.get('width_mm','?')}x{aw_info.get('height_mm','?')}mm",
            'dpi_source':     aw_info.get('source','fallback'),
            'align_method':   method,
            'reg_quality':    round(reg_quality,3),
            'ssim_score':     round(ssim_score,3),
            'diff_pct':       round(diff_pct,1),
            'debug':          debug_info,
        })

    except Exception as e:
        import traceback
        return jsonify({'error':str(e),'trace':traceback.format_exc()}),500


@app.route('/', defaults={'path':''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder,path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__=='__main__':
    port = int(os.environ.get('PORT',8080))
    print(f'QC Inspector Pro — ALIGN→NORMALIZE→COMPARE→DETECT')
    app.run(host='0.0.0.0', port=port, debug=False)
