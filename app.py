"""
QC Print Inspector — Image Registration Pipeline
=================================================
Flow: LOAD → REGISTER → NORMALIZE → DIFF → DETECT → RENDER

Registration handles:
  - rotation (any angle)
  - scale difference
  - perspective distortion / camera angle
  - paper shift / small deformation

Key principle: MUST NOT produce defect from alignment error only.
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64, os, math, json, requests
from io import BytesIO
from PIL import Image, ExifTags
from skimage.metrics import structural_similarity as ssim_func

app = Flask(__name__, static_folder='static')
CORS(app)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# ── QC Standards ──────────────────────────────────────────────────────────────
QC = {
    'dot':         {'warn':0.3,  'fail':0.5},
    'spot':        {'warn':0.3,  'fail':0.5},
    'hickey':      {'warn':0.3,  'fail':0.5},
    'scratch':     {'warn':3.0,  'fail':5.0},
    'streak':      {'warn':3.0,  'fail':5.0},
    'missing_ink': {'warn':0.5,  'fail':1.0},
    'extra_ink':   {'warn':0.5,  'fail':1.0},
    'blur':        {'warn':0.5,  'fail':1.0},
    'anomaly':     {'warn':0.3,  'fail':0.5},
}
LABELS = {
    'dot':'Dot/chấm','spot':'Spot/đốm','hickey':'Hickey',
    'scratch':'Xước/vệt dài','streak':'Vệt/sọc',
    'missing_ink':'Thiếu mực','extra_ink':'Thừa mực',
    'blur':'Nhòe/mờ','anomaly':'Bất thường',
}
COLORS = {
    'dot':(0,220,255),'spot':(0,200,255),'hickey':(0,100,255),
    'scratch':(255,50,200),'streak':(255,140,0),
    'missing_ink':(50,255,50),'extra_ink':(255,50,50),
    'blur':(255,220,0),'anomaly':(180,180,180),
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

def px2mm(px,dpi):   return px/dpi*25.4
def mm2px(mm,dpi):   return mm/25.4*dpi
def a2mm2(a,dpi):    return a/(dpi/25.4)**2
def diam(a):         return 2*math.sqrt(max(a,0)/math.pi)

def verdict(dtype, val):
    s = QC.get(dtype, QC['anomaly'])
    return 'FAIL' if val>=s['fail'] else ('WARN' if val>=s['warn'] else 'PASS')

# ── DPI ───────────────────────────────────────────────────────────────────────
def get_dpi(b64, ftype='image'):
    if ftype=='pdf': return _pdf_dpi(b64)
    return _img_dpi(b64)

def _pdf_dpi(b64):
    try:
        import fitz
        doc = fitz.open(stream=base64.b64decode(b64), filetype='pdf')
        p=doc[0]; r=p.rect
        wm=r.width/72*25.4; hm=r.height/72*25.4
        dpi=300
        pix=p.get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72),colorspace=fitz.csRGB)
        arr=np.frombuffer(pix.tobytes('png'),np.uint8)
        img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
        doc.close()
        return {'dpi':float(dpi),'width_mm':round(wm,2),'height_mm':round(hm,2),
                'source':f'PDF {wm:.1f}x{hm:.1f}mm@{dpi}dpi','img_cv2':img}
    except Exception as e:
        return {'dpi':None,'source':f'pdf_err:{e}'}

def _img_dpi(b64):
    try:
        pil=Image.open(BytesIO(base64.b64decode(b64)))
        w,h=pil.size; dpi=None
        if 'dpi' in pil.info: dpi=float(pil.info['dpi'][0])
        if not dpi:
            try:
                ex=pil._getexif()
                if ex:
                    for tid,v in ex.items():
                        if ExifTags.TAGS.get(tid)=='XResolution':
                            dpi=float(v[0]/v[1]) if isinstance(v,tuple) else float(v)
            except: pass
        cv2img=cv2.cvtColor(np.array(pil.convert('RGB')),cv2.COLOR_RGB2BGR)
        if dpi and dpi>10:
            wm=w/dpi*25.4; hm=h/dpi*25.4
            return {'dpi':dpi,'width_mm':round(wm,2),'height_mm':round(hm,2),
                    'source':f'EXIF {wm:.1f}x{hm:.1f}mm@{dpi:.0f}dpi','img_cv2':cv2img}
        return {'dpi':None,'source':'no_dpi','img_cv2':cv2img}
    except Exception as e:
        return {'dpi':None,'source':f'err:{e}'}

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — IMAGE REGISTRATION
#  Goal: warp real photo → AW coordinate space
#  Handles: rotation, scale, perspective, camera angle, shift
# ═══════════════════════════════════════════════════════════════════════════════

def register(aw, real):
    """
    Full registration pipeline with fallback chain.

    Returns:
        warped      : real image warped to AW space (same size as AW)
        method      : which method succeeded
        quality     : 0..1 alignment quality score
        info        : debug dict
    """
    h, w = aw.shape[:2]
    info = {}

    # Pre-resize real to AW dimensions (coarse alignment)
    real_r = cv2.resize(real, (w, h), interpolation=cv2.INTER_AREA)

    # Enhance contrast before feature detection
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    ag = cv2.cvtColor(aw,     cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(real_r, cv2.COLOR_BGR2GRAY)
    ag_e = clahe.apply(ag)
    rg_e = clahe.apply(rg)

    # ── Method 1: SIFT homography (best for scale + perspective) ──
    result = _feature_homography(aw, real_r, ag_e, rg_e, w, h, 'SIFT', info)
    if result is not None:
        warped, H, q = result
        # ECC sub-pixel refinement on top of homography
        warped = _ecc_refine(aw, warped, ag, w, h)
        info['final'] = 'SIFT_homography+ECC'
        return warped, 'SIFT+ECC', q, info

    # ── Method 2: ORB homography (faster, works on low-texture) ──
    result = _feature_homography(aw, real_r, ag_e, rg_e, w, h, 'ORB', info)
    if result is not None:
        warped, H, q = result
        warped = _ecc_refine(aw, warped, ag, w, h)
        info['final'] = 'ORB_homography+ECC'
        return warped, 'ORB+ECC', q, info

    # ── Method 3: AKAZE (robust for blurry / low-contrast) ──
    result = _feature_homography(aw, real_r, ag_e, rg_e, w, h, 'AKAZE', info)
    if result is not None:
        warped, H, q = result
        warped = _ecc_refine(aw, warped, ag, w, h)
        info['final'] = 'AKAZE_homography+ECC'
        return warped, 'AKAZE+ECC', q, info

    # ── Method 4: Pure ECC (translation + rotation, no perspective) ──
    warped = _ecc_full(aw, real_r, ag, w, h)
    info['final'] = 'ECC_only'
    return warped, 'ECC', 0.4, info


def _feature_homography(aw, real, ag_e, rg_e, w, h, det_name, info):
    """
    Feature detection → matching → homography → warpPerspective.
    Returns (warped, H, quality) or None if failed.
    """
    try:
        # Select detector
        if det_name == 'SIFT':
            det  = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01, edgeThreshold=15)
            norm = cv2.NORM_L2
        elif det_name == 'ORB':
            det  = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=12, edgeThreshold=15)
            norm = cv2.NORM_HAMMING
        elif det_name == 'AKAZE':
            det  = cv2.AKAZE_create()
            norm = cv2.NORM_HAMMING
        else:
            return None

        # Detect keypoints
        kp1, des1 = det.detectAndCompute(ag_e, None)
        kp2, des2 = det.detectAndCompute(rg_e, None)

        info[f'{det_name}_kp'] = f'{len(kp1) if kp1 else 0}/{len(kp2) if kp2 else 0}'

        if des1 is None or des2 is None:
            return None
        if len(kp1) < 8 or len(kp2) < 8:
            return None

        # Match with Lowe ratio test
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        raw = matcher.knnMatch(des1, des2, k=2)
        good = [m for m,n in raw if m.distance < 0.72 * n.distance] if raw and len(raw[0])==2 else []

        info[f'{det_name}_matches'] = len(good)

        if len(good) < 8:
            return None

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # RANSAC homography
        # findHomography: maps dst (real) → src (AW)
        H, mask = cv2.findHomography(
            dst, src,
            cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=3000,
            confidence=0.995
        )

        if H is None:
            return None

        inliers      = int(mask.sum()) if mask is not None else 0
        inlier_ratio = inliers / len(good)
        info[f'{det_name}_inliers'] = f'{inliers}/{len(good)} ratio={inlier_ratio:.2f}'

        if inlier_ratio < 0.20 or inliers < 6:
            return None

        if not _valid_H(H, w, h):
            info[f'{det_name}_H'] = 'degenerate'
            return None

        # warpPerspective: transform real → AW coordinate space
        warped = cv2.warpPerspective(
            real, H, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        quality = min(1.0, inlier_ratio * min(inliers/30.0, 1.0))
        print(f'[{det_name}] kp={len(kp1)}/{len(kp2)} good={len(good)} '
              f'inliers={inliers} ratio={inlier_ratio:.2f} quality={quality:.2f}')
        return warped, H, quality

    except Exception as e:
        info[f'{det_name}_err'] = str(e)
        return None


def _valid_H(H, w, h):
    """Reject degenerate homography (extreme distortion / collapse / flip)."""
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    try:
        wc = cv2.perspectiveTransform(corners, H).reshape(4,2)
    except Exception:
        return False

    # Area ratio check
    area  = cv2.contourArea(wc)
    ratio = area / (w * h)
    if ratio < 0.05 or ratio > 25.0:
        return False

    # Convexity check (no self-intersecting quad)
    hull = cv2.convexHull(wc.astype(np.float32))
    if len(hull) != 4:
        return False

    # Minimum angle check (reject extreme shear)
    for i in range(4):
        a = wc[i]; b = wc[(i+1)%4]; c = wc[(i+2)%4]
        v1 = b-a; v2 = c-b
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            return False
        cos_a = np.dot(v1,v2)/(n1*n2)
        if abs(cos_a) > 0.96:   # angle < ~16° → degenerate
            return False
    return True


def _ecc_refine(aw, warped, ag, w, h):
    """Sub-pixel ECC refinement after homography."""
    rg = cv2.cvtColor(cv2.resize(warped,(w,h)), cv2.COLOR_BGR2GRAY)
    warp = np.eye(2,3,dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
    try:
        _, warp = cv2.findTransformECC(
            ag.astype(np.float32), rg.astype(np.float32),
            warp, cv2.MOTION_EUCLIDEAN, crit, None, 5
        )
        return cv2.warpAffine(warped, warp, (w,h),
                               flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return warped


def _ecc_full(aw, real, ag, w, h):
    """Full ECC alignment when feature matching fails completely."""
    rg = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2,3,dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 500, 1e-7)
    try:
        _, warp = cv2.findTransformECC(
            ag.astype(np.float32), rg, warp,
            cv2.MOTION_EUCLIDEAN, crit, None, 5
        )
        return cv2.warpAffine(real, warp, (w,h),
                               flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return real

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — NORMALIZE
#  Removes camera exposure difference, NOT real defects
# ═══════════════════════════════════════════════════════════════════════════════

def normalize(aw, warped):
    """
    Match global brightness/contrast of warped to AW.
    Use gentle Gaussian blur to suppress sensor noise.
    """
    h, w = aw.shape[:2]
    warped = cv2.resize(warped, (w,h))

    ag = cv2.cvtColor(aw,     cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    aw_mean, aw_std = float(np.mean(ag)), float(np.std(ag))
    r_mean,  r_std  = float(np.mean(rg)), float(np.std(rg))

    scale  = aw_std / max(r_std, 1.0)
    offset = aw_mean - r_mean * scale

    # Apply global correction to BGR
    warped_n = np.clip(warped.astype(np.float32)*scale+offset, 0, 255).astype(np.uint8)

    # Convert to gray + blur (noise reduction, NOT defect removal)
    ag_b = cv2.GaussianBlur(ag,                                    (3,3), 0.8)
    rg_n = cv2.cvtColor(warped_n, cv2.COLOR_BGR2GRAY)
    rg_b = cv2.GaussianBlur(rg_n,                                  (3,3), 0.8)

    return ag_b, rg_b, aw, warped_n

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — DIFF + THRESHOLD
#  SSIM + LAB + Edge combined. Adaptive threshold based on MAD.
# ═══════════════════════════════════════════════════════════════════════════════

def diff_and_threshold(ag, rg, aw_bgr, real_bgr, sensitivity):
    """
    Multi-method diff with adaptive threshold.

    Adaptive threshold using MAD (Median Absolute Deviation):
      thresh = median + k * MAD * 1.4826
    This AUTOMATICALLY adjusts based on alignment quality:
      - Good alignment → low noise → low MAD → catches small defects
      - Poor alignment → high noise → high MAD → ignores alignment artifacts
    """
    h, w = ag.shape[:2]

    # A. SSIM map
    win = max(3, min(11, min(h,w)//8 | 1))
    try:
        score, ssim_map = ssim_func(ag, rg, win_size=win, full=True, data_range=255)
        map_ssim = np.clip(1.0-(ssim_map+1.0)/2.0, 0, 1).astype(np.float32)
    except Exception:
        score=1.0; map_ssim=np.zeros((h,w),np.float32)

    # B. LAB perceptual diff
    aL = cv2.cvtColor(aw_bgr,   cv2.COLOR_BGR2LAB).astype(np.float32)
    rL = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dL  = np.abs(aL[:,:,0]-rL[:,:,0])
    dAB = np.sqrt((aL[:,:,1]-rL[:,:,1])**2+(aL[:,:,2]-rL[:,:,2])**2)
    map_lab = np.clip(dL/100.0*0.6+dAB/180.0*0.4, 0, 1).astype(np.float32)

    # C. Edge diff (detects missing/extra lines at ±1px tolerance)
    ae = cv2.Canny(ag,30,100).astype(np.float32)/255.0
    re = cv2.Canny(rg,30,100).astype(np.float32)/255.0
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    map_edge = np.abs(cv2.dilate(ae,k3)-cv2.dilate(re,k3)).astype(np.float32)

    # Combine
    heat = map_ssim*0.40 + map_lab*0.35 + map_edge*0.25
    heat = np.clip(heat,0,1).astype(np.float32)

    # Adaptive threshold with MAD
    flat   = heat.flatten()
    med    = float(np.median(flat))
    mad    = float(np.median(np.abs(flat-med)))
    rstd   = mad * 1.4826   # robust std estimate

    # k: 10→2.5  30→3.5  70→5.5
    k = 2.5 + (sensitivity-10)/60.0 * 3.0
    thresh = max(med + k*rstd, sensitivity/250.0, 0.06)

    print(f'SSIM={score:.3f} med={med:.4f} MAD={mad:.4f} k={k:.1f} thresh={thresh:.4f}')

    binary = (heat > thresh).astype(np.uint8)*255

    # Morphological noise removal
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  ko)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kc)

    return heat, binary, float(score)

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — DEFECT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect(binary, heat, aw_bgr, real_bgr, dpi):
    contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    min_area = math.pi * mm2px(0.10,dpi)**2  # min 0.1mm radius

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(min_area,3): continue

        x,y,bw,bh = cv2.boundingRect(cnt)
        aspect = max(bw,bh)/max(min(bw,bh),1)

        mk = np.zeros(binary.shape,np.uint8)
        cv2.drawContours(mk,[cnt],-1,255,-1)
        mh = float(cv2.mean(heat,mask=mk)[0])

        dtype = _classify(cnt,area,bw,bh,aspect,aw_bgr,real_bgr,dpi)
        if dtype is None: continue

        # Size
        amm2 = a2mm2(area,dpi); dm = diam(amm2)
        lm = px2mm(max(bw,bh),dpi); wm = px2mm(min(bw,bh),dpi)

        if dtype in ('dot','spot','hickey'):   meas=dm;   ss=f'⌀{dm:.2f}mm'
        elif dtype in ('scratch','streak'):    meas=lm;   ss=f'{lm:.2f}x{wm:.2f}mm'
        else:                                  meas=amm2; ss=f'{amm2:.3f}mm²'

        v = verdict(dtype, meas)
        if v=='PASS': continue

        out.append({'type':dtype,'label':LABELS.get(dtype,dtype),
                    'x':int(x),'y':int(y),'w':int(bw),'h':int(bh),
                    'area_mm2':round(amm2,3),'diameter_mm':round(dm,3),
                    'length_mm':round(lm,3),'width_mm':round(wm,3),
                    'size_str':ss,'verdict':v,
                    'severity':'high' if v=='FAIL' else 'medium',
                    'heat_score':round(mh,3),'contour':cnt})

    out.sort(key=lambda d:d['heat_score'],reverse=True)
    return out


def _classify(cnt,area,bw,bh,aspect,aw,real,dpi):
    perim = cv2.arcLength(cnt,True)
    circ  = 4*math.pi*area/perim**2 if perim>0 else 0

    if aspect>10 and max(bw,bh)>mm2px(2.0,dpi): return 'scratch'
    if aspect>4  and max(bw,bh)>mm2px(1.5,dpi): return 'streak'
    if circ>0.65 and bw<mm2px(3,dpi):            return 'hickey'
    if circ>0.5  and bw<mm2px(1.5,dpi):          return 'dot'
    if circ>0.35 and bw<mm2px(6,dpi):            return 'spot'

    x,y,w2,h2 = cv2.boundingRect(cnt)
    ar=aw[y:y+h2,x:x+w2]; pr=real[y:y+h2,x:x+w2]
    if ar.size==0 or pr.size==0: return 'anomaly'

    am=float(np.mean(cv2.cvtColor(ar,cv2.COLOR_BGR2GRAY)))
    pm=float(np.mean(cv2.cvtColor(pr,cv2.COLOR_BGR2GRAY)))
    if pm>am+15: return 'missing_ink'
    if pm<am-15: return 'extra_ink'

    try:
        al=cv2.Laplacian(cv2.cvtColor(ar,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
        pl=cv2.Laplacian(cv2.cvtColor(pr,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
        if pl<al*0.4: return 'blur'
    except Exception: pass

    return 'anomaly' if area>=mm2px(0.15,dpi)**2 else None

# ═══════════════════════════════════════════════════════════════════════════════
#  RENDER
# ═══════════════════════════════════════════════════════════════════════════════

def render_overlay(img, defects, text_def=[]):
    """Color overlay only — no text labels."""
    out = img.copy()
    ih,iw = out.shape[:2]
    for d in defects+text_def:
        c = COLORS.get(d.get('type',''), (180,180,180))
        if d.get('is_text'): c=(100,0,255)
        x=max(0,d['x']); y=max(0,d['y'])
        w=min(d['w'],iw-x); h=min(d['h'],ih-y)
        if w<=0 or h<=0: continue
        ov=out.copy()
        cv2.rectangle(ov,(x,y),(x+w,y+h),c,-1)
        a=0.55 if d.get('verdict')=='FAIL' else 0.38
        cv2.addWeighted(ov,a,out,1-a,0,out)
        cv2.rectangle(out,(x,y),(x+w,y+h),c,3 if d.get('verdict')=='FAIL' else 2)
    return out


def render_dotmap(shape, defects, heat, text_def=[]):
    """
    Pure black background.
    White = defect location.
    QC scans for white spots to know where to re-inspect.
    """
    ih,iw = shape[:2]
    out = np.zeros((ih,iw,3),dtype=np.uint8)

    # Heat → white mask
    hh,hw = heat.shape[:2]
    rh=min(hh,ih); rw=min(hw,iw)
    if rh>0 and rw>0:
        hr = cv2.resize(heat[:rh,:rw],(rw,rh))
        _,mask = cv2.threshold((hr*255).astype(np.uint8),15,255,cv2.THRESH_BINARY)
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.dilate(mask,k5)
        out[:rh,:rw][mask>0] = 255

    # Guarantee each defect bbox is white
    for d in defects+text_def:
        x=max(0,d['x']); y=max(0,d['y'])
        w=min(d.get('w',8),iw-x); h=min(d.get('h',8),ih-y)
        if w>0 and h>0:
            out[y:y+h,x:x+w] = 255

    return out

# ── Gemini text check ─────────────────────────────────────────────────────────
def text_check(aw, real):
    if not GEMINI_API_KEY: return []
    h,w = aw.shape[:2]
    prompt=(
        f"So sánh Ảnh 1 (AW gốc) và Ảnh 2 (tờ in, {w}x{h}px). "
        "Chỉ kiểm tra VĂN BẢN: dấu thanh tiếng Việt, dấu câu, ký tự sai/thiếu. "
        'JSON: {"text_defects":[{"type":"wrong_diacritic|missing_punct|wrong_char",'
        '"label":"tên lỗi","detail":"mô tả","x":0,"y":0,"w":40,"h":25}]}'
    )
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
        print(f'Gemini text err:{e}'); return []

def ai_summary(defects):
    if not GEMINI_API_KEY or not defects: return None
    lines="\n".join([f"- #{i+1}: {d.get('label','?')} {d.get('size_str','')} [{d.get('verdict','?')}]"
                     for i,d in enumerate(defects)])
    try:
        r=requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[{"text":
                f"QC in ấn. {len(defects)} lỗi:\n{lines}\n"
                "Báo cáo tiếng Việt ≤80 từ."}]}],
                "generationConfig":{"temperature":0.2,"maxOutputTokens":200}},
            timeout=30)
        return (r.json().get('candidates',[{}])[0]
                .get('content',{}).get('parts',[{}])[0].get('text',''))
    except: return None

# ── API endpoints ─────────────────────────────────────────────────────────────
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
        dpi = float(man_dpi) if man_dpi else (aw_info.get('dpi') or 150.0)

        # Load
        aw = b64_to_cv2(aw_b64)
        pr = b64_to_cv2(pr_b64)
        if aw is None or pr is None:
            return jsonify({'error':'Không đọc được ảnh'}),400
        print(f'Load: AW={aw.shape} Real={pr.shape} DPI={dpi} Sens={sens}')

        # ── REGISTER ──────────────────────────────────────────────────────────
        warped, method, quality, reg_info = register(aw, pr)
        print(f'Register: method={method} quality={quality:.2f} info={reg_info}')

        # ── NORMALIZE ─────────────────────────────────────────────────────────
        ag, rg, aw_n, real_n = normalize(aw, warped)

        # ── DIFF + THRESHOLD ──────────────────────────────────────────────────
        heat, binary, ssim_score = diff_and_threshold(ag, rg, aw_n, real_n, sens)
        n_diff = int(np.sum(binary>0))
        total  = ag.shape[0]*ag.shape[1]
        diff_pct = n_diff/total*100
        print(f'Diff: {n_diff}px/{total}px={diff_pct:.1f}% SSIM={ssim_score:.3f}')

        # ── DETECT ────────────────────────────────────────────────────────────
        defects = detect(binary, heat, aw_n, real_n, dpi)
        print(f'Defects: {len(defects)}')

        # ── TEXT CHECK ────────────────────────────────────────────────────────
        txt_def = []
        if chk_txt and GEMINI_API_KEY:
            txt_def = text_check(aw_n, real_n)
            for td in txt_def:
                td['verdict']='FAIL'; td['size_str']=td.get('detail','')

        # ── RENDER ────────────────────────────────────────────────────────────
        col_img = render_overlay(real_n, defects, txt_def)
        dot_img = render_dotmap(real_n.shape, defects, heat, txt_def)

        all_d = defects+[dict(d,severity='high') for d in txt_def]
        summ  = ai_summary(all_d) if all_d else None

        phys_out = [{k:v for k,v in d.items() if k!='contour'} for d in defects]
        txt_out  = [dict(d,is_text=True) for d in txt_def]

        fail_c = sum(1 for d in all_d if d.get('verdict')=='FAIL')
        warn_c = sum(1 for d in all_d if d.get('verdict')=='WARN')
        v = 'FAIL' if fail_c>0 else ('WARN' if warn_c>0 else 'PASS')

        return jsonify({
            'verdict':v, 'defect_count':len(all_d),
            'fail_count':fail_c, 'warn_count':warn_c,
            'physical_count':len(defects), 'text_count':len(txt_def),
            'defects':phys_out+txt_out,
            'result_color':  cv2_to_b64(col_img),
            'result_dotmap': cv2_to_b64(dot_img),
            'ai_summary':    summ,
            'dpi_aw':        round(dpi,1),
            'aw_size':       f"{aw_info.get('width_mm','?')}x{aw_info.get('height_mm','?')}mm",
            'dpi_source':    aw_info.get('source','fallback'),
            'align_method':  method,
            'reg_quality':   round(quality,3),
            'ssim_score':    round(ssim_score,3),
            'diff_pct':      round(diff_pct,1),
            'reg_debug':     reg_info,
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
    print('QC Inspector — REGISTER→NORMALIZE→DIFF→DETECT')
    app.run(host='0.0.0.0',port=port,debug=False)
