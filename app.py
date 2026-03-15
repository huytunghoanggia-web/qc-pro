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

# ── QC Standards (mm) ────────────────────────────────────────────────────────
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

def px_to_mm(px, dpi): return px / dpi * 25.4
def mm_to_px(mm, dpi): return mm / 25.4 * dpi
def area_to_mm2(area_px, dpi): return area_px / (dpi/25.4)**2
def diameter_from_area_mm(a): return 2*math.sqrt(max(a,0)/math.pi)

def judge(dtype, val_mm):
    s = QC_STANDARDS.get(dtype, QC_STANDARDS['anomaly'])
    if val_mm >= s['fail']:   return 'FAIL'
    elif val_mm >= s['warn']: return 'WARN'
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

# ── CORE: Align 2 vùng trước khi so sánh ────────────────────────────────────
def align_two_regions(aw_crop, print_crop):
    """
    Căn chỉnh print_crop vào aw_crop dùng ECC (Enhanced Correlation Coefficient).
    ECC tốt hơn feature matching cho 2 vùng nhỏ giống nhau.
    Trả về: (aligned_print, warp_matrix, success)
    """
    h, w = aw_crop.shape[:2]
    # Resize print về cùng size AW trước
    pc = cv2.resize(print_crop, (w, h))

    ag = cv2.cvtColor(aw_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pg = cv2.cvtColor(pc,      cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Thử ECC với translation (chỉ dịch chuyển x,y - phù hợp nhất khi 2 vùng gần giống)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, warp = cv2.findTransformECC(ag, pg, warp,
                                        cv2.MOTION_EUCLIDEAN, criteria,
                                        None, 5)
        aligned = cv2.warpAffine(pc, warp, (w, h),
                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
        return aligned, warp, True
    except Exception as e:
        # ECC thất bại - dùng ảnh resize thẳng
        return pc, warp, False

# ── CORE: 5-method comparison với normalization thông minh ──────────────────
def compare_regions(aw_crop, print_crop, sensitivity):
    """
    So sánh 2 vùng đã được align.
    sensitivity: 10-70, càng cao càng ít nhạy (chỉ bắt lỗi lớn).
    Trả về heat map [0,1] và binary mask.
    """
    h, w = aw_crop.shape[:2]
    pc = cv2.resize(print_crop, (w, h))

    # ── Normalize brightness/contrast để bù cho chụp ảnh ──
    # Dùng CLAHE để equalize histogram từng vùng nhỏ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ag_raw = cv2.cvtColor(aw_crop, cv2.COLOR_BGR2GRAY)
    pg_raw = cv2.cvtColor(pc,      cv2.COLOR_BGR2GRAY)

    # Global brightness correction
    aw_mean = float(np.mean(ag_raw))
    pr_mean = float(np.mean(pg_raw))
    brightness_diff = aw_mean - pr_mean
    pc_corrected = np.clip(pc.astype(np.float32) + brightness_diff, 0, 255).astype(np.uint8)

    ag = clahe.apply(ag_raw)
    pg = clahe.apply(cv2.cvtColor(pc_corrected, cv2.COLOR_BGR2GRAY))

    # ── 1. LAB diff sau brightness correction ──
    aL = cv2.cvtColor(aw_crop,        cv2.COLOR_BGR2LAB).astype(np.float32)
    pL = cv2.cvtColor(pc_corrected,   cv2.COLOR_BGR2LAB).astype(np.float32)
    # Chỉ dùng kênh L (lightness) và AB riêng
    diff_L  = np.abs(aL[:,:,0] - pL[:,:,0])
    diff_AB = np.sqrt((aL[:,:,1]-pL[:,:,1])**2 + (aL[:,:,2]-pL[:,:,2])**2)
    # Normalize: diff_L max ~100, diff_AB max ~180
    map_lab = np.clip(diff_L/80.0 * 0.6 + diff_AB/120.0 * 0.4, 0, 1).astype(np.float32)

    # ── 2. SSIM ──
    win = max(3, min(7, min(h,w)//4 | 1))
    try:
        _, ssim_map = ssim(ag, pg, win_size=win, full=True, data_range=255)
        map_ssim = np.clip((1.0 - ssim_map) / 2.0, 0, 1).astype(np.float32)
    except:
        map_ssim = np.zeros((h,w), np.float32)

    # ── 3. Edge diff ──
    ae = cv2.Canny(ag, 40, 120).astype(np.float32)/255.0
    pe = cv2.Canny(pg, 40, 120).astype(np.float32)/255.0
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    ae_d = cv2.dilate(ae, k3, iterations=1)
    pe_d = cv2.dilate(pe, k3, iterations=1)
    map_edge = np.abs(ae_d - pe_d).astype(np.float32)

    # ── 4. Gradient magnitude diff ──
    ag_f = ag.astype(np.float32)
    pg_f = pg.astype(np.float32)
    gx_a = cv2.Sobel(ag_f, cv2.CV_32F, 1, 0, ksize=3)
    gy_a = cv2.Sobel(ag_f, cv2.CV_32F, 0, 1, ksize=3)
    gx_p = cv2.Sobel(pg_f, cv2.CV_32F, 1, 0, ksize=3)
    gy_p = cv2.Sobel(pg_f, cv2.CV_32F, 0, 1, ksize=3)
    mag_a = np.sqrt(gx_a**2 + gy_a**2)
    mag_p = np.sqrt(gx_p**2 + gy_p**2)
    map_grad = np.abs(mag_a - mag_p) / (np.maximum(mag_a, mag_p) + 1e-6)
    map_grad = np.clip(map_grad, 0, 1).astype(np.float32)

    # ── 5. Local texture diff ──
    blur_a = cv2.GaussianBlur(ag_f, (5,5), 0)
    blur_p = cv2.GaussianBlur(pg_f, (5,5), 0)
    detail_a = np.abs(ag_f - blur_a)
    detail_p = np.abs(pg_f - blur_p)
    map_texture = np.abs(detail_a - detail_p) / 128.0
    map_texture = np.clip(map_texture, 0, 1).astype(np.float32)

    # ── Weighted combine ──
    heat = (map_lab     * 0.35 +
            map_ssim    * 0.25 +
            map_edge    * 0.20 +
            map_grad    * 0.12 +
            map_texture * 0.08)
    heat = np.clip(heat, 0, 1).astype(np.float32)

    # ── Adaptive threshold ──
    # sensitivity 28 (default) → thresh ~0.35
    # sensitivity 10 (max sensitive) → thresh ~0.15
    # sensitivity 70 (min sensitive) → thresh ~0.65
    thresh_val = sensitivity / 100.0

    # Thêm: loại bỏ "background noise" bằng cách tính median của heat
    # và chỉ giữ những điểm vượt quá median + k*std
    heat_flat = heat.flatten()
    med = float(np.median(heat_flat))
    std = float(np.std(heat_flat))
    # Adaptive: threshold phải ít nhất là median + 2*std
    adaptive_thresh = max(thresh_val, med + 2.0 * std)
    binary = (heat > adaptive_thresh).astype(np.uint8) * 255

    # Morphological cleanup
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k2)
    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k4)

    return heat, binary, pc_corrected

# ── Extract defects ───────────────────────────────────────────────────────────
def extract_defects(binary, heat, aw_crop, print_aligned, dpi):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = aw_crop.shape[:2]
    defects = []
    min_area_px = math.pi * (mm_to_px(0.15, dpi))**2

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < max(min_area_px, 3): continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw,bh) / max(min(bw,bh), 1)

        mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_heat = float(cv2.mean(heat, mask=mask)[0])

        dtype = _classify(cnt, area_px, bw, bh, aspect, aw_crop, print_aligned, dpi)
        if dtype is None: continue

        area_mm2  = area_to_mm2(area_px, dpi)
        diam_mm   = diameter_from_area_mm(area_mm2)
        length_mm = px_to_mm(max(bw,bh), dpi)
        width_mm  = px_to_mm(min(bw,bh), dpi)

        if dtype in ('dot','spot','hickey'):
            measure = diam_mm
            size_str = f'⌀{diam_mm:.2f}mm'
        elif dtype in ('scratch','streak'):
            measure = length_mm
            size_str = f'{length_mm:.2f}x{width_mm:.2f}mm'
        else:
            measure = area_mm2
            size_str = f'{area_mm2:.3f}mm²'

        verdict = judge(dtype, measure)
        if verdict == 'PASS': continue

        defects.append({
            'type':dtype, 'label':DEFECT_LABELS.get(dtype,dtype),
            'x':int(x), 'y':int(y), 'w':int(bw), 'h':int(bh),
            'area_mm2':round(area_mm2,3), 'diameter_mm':round(diam_mm,3),
            'length_mm':round(length_mm,3), 'width_mm':round(width_mm,3),
            'size_str':size_str, 'verdict':verdict,
            'severity':'high' if verdict=='FAIL' else 'medium',
            'heat_score':round(mean_heat,3), 'contour':cnt,
        })
    return defects

def _classify(cnt, area_px, bw, bh, aspect, aw_img, pr_img, dpi):
    perim = cv2.arcLength(cnt, True)
    circ  = 4*math.pi*area_px/perim**2 if perim>0 else 0
    if aspect > 10 and max(bw,bh) > mm_to_px(2.5,dpi): return 'scratch'
    if aspect > 4  and max(bw,bh) > mm_to_px(1.5,dpi): return 'streak'
    if circ > 0.65 and bw < mm_to_px(3,dpi):           return 'hickey'
    if circ > 0.5  and bw < mm_to_px(1.5,dpi):         return 'dot'
    if circ > 0.35 and bw < mm_to_px(6,dpi):           return 'spot'
    x,y,w2,h2 = cv2.boundingRect(cnt)
    ar = aw_img[y:y+h2, x:x+w2]; pr = pr_img[y:y+h2, x:x+w2]
    if ar.size==0 or pr.size==0: return 'anomaly'
    am = float(np.mean(cv2.cvtColor(ar,cv2.COLOR_BGR2GRAY)))
    pm = float(np.mean(cv2.cvtColor(pr,cv2.COLOR_BGR2GRAY)))
    if pm > am+15: return 'missing_ink'
    if pm < am-15: return 'extra_ink'
    al = cv2.Laplacian(cv2.cvtColor(ar,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
    pl = cv2.Laplacian(cv2.cvtColor(pr,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
    if pl < al*0.4: return 'blur'
    if area_px < mm_to_px(0.2,dpi)**2: return None
    return 'anomaly'

# ── Draw results ──────────────────────────────────────────────────────────────
def draw_defects(img, defects, text_defects=[]):
    """Chỉ tô màu vùng lỗi, không ghi chữ."""
    out = img.copy()
    ih, iw = out.shape[:2]
    all_d = [(d, False) for d in defects] + [(d, True) for d in text_defects]
    for i, (d, is_text) in enumerate(all_d):
        color = DEFECT_COLORS_CV.get(d.get('type',''), (180,180,180))
        x=max(0,d['x']); y=max(0,d['y'])
        w=min(d['w'],iw-x); h=min(d['h'],ih-y)
        if w<=0 or h<=0: continue
        # Tô màu bán trong suốt
        ov = out.copy()
        cv2.rectangle(ov,(x,y),(x+w,y+h),color,-1)
        alpha = 0.5 if d.get('verdict')=='FAIL' else 0.35
        cv2.addWeighted(ov,alpha,out,1-alpha,0,out)
        # Viền màu
        thick = 3 if d.get('verdict')=='FAIL' else 2
        cv2.rectangle(out,(x,y),(x+w,y+h),color,thick)
    return out

def make_dotmap(print_img, defects, heat, offset_x=0, offset_y=0):
    """
    Nền đen tuyền hoàn toàn.
    Vị trí lỗi = màu TRẮNG để QC xác định chính xác điểm cần kiểm tra lại.
    """
    ih, iw = print_img.shape[:2]
    # Bắt đầu với nền đen tuyền
    out = np.zeros((ih, iw, 3), dtype=np.uint8)

    hh, hw = heat.shape[:2]
    rh2 = min(hh, ih-offset_y)
    rw2 = min(hw, iw-offset_x)

    if rh2 > 0 and rw2 > 0:
        # Resize heat map về kích thước vùng
        heat_r = cv2.resize(heat[:rh2,:rw2].astype(np.float32), (rw2, rh2))
        # Tạo mask nhị phân từ heat
        heat_u8 = (heat_r * 255).astype(np.uint8)
        _, mask = cv2.threshold(heat_u8, 25, 255, cv2.THRESH_BINARY)
        # Dilate để vùng lỗi dễ thấy hơn
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.dilate(mask, k)
        # Tô TRẮNG vào vị trí lỗi
        out[offset_y:offset_y+rh2, offset_x:offset_x+rw2][mask>0] = (255,255,255)

    # Vẽ thêm hình chữ nhật trắng rõ ràng quanh từng defect bbox
    for d in defects:
        x=max(0,d['x']+offset_x); y=max(0,d['y']+offset_y)
        w=min(d['w'],iw-x); h=min(d['h'],ih-y)
        if w>0 and h>0:
            # Fill trắng toàn bộ bbox
            out[y:y+h, x:x+w] = 255
            # Viền đậm hơn
            cv2.rectangle(out,(x,y),(x+w,y+h),(255,255,255),2)

    return out

# ── Gemini text check ─────────────────────────────────────────────────────────
def check_text_gemini(aw_img, print_img):
    if not GEMINI_API_KEY: return []
    h,w = aw_img.shape[:2]
    prompt = f"""So sánh Ảnh 1 (AW gốc) và Ảnh 2 (tờ in, {w}x{h}px).
Chỉ kiểm tra VĂN BẢN: dấu thanh tiếng Việt, dấu câu, ký tự sai/thiếu/thừa.
JSON: {{"text_defects":[{{"type":"wrong_diacritic|missing_punct|wrong_char","label":"tên VN","detail":"mô tả","x":0,"y":0,"w":30,"h":20}}]}}"""
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[
                {"inline_data":{"mime_type":"image/jpeg","data":img_to_b64_jpeg(aw_img)}},
                {"inline_data":{"mime_type":"image/jpeg","data":img_to_b64_jpeg(print_img)}},
                {"text":prompt}
            ]}],"generationConfig":{"temperature":0.05,"maxOutputTokens":1500}},
            timeout=45)
        raw = r.json().get('candidates',[{}])[0].get('content',{}).get('parts',[{}])[0].get('text','')
        return json.loads(raw.replace('```json','').replace('```','').strip()).get('text_defects',[])
    except Exception as e:
        print(f"Gemini text err: {e}"); return []

def get_ai_summary(all_defects):
    if not GEMINI_API_KEY or not all_defects: return None
    lines = "\n".join([f"- #{i+1}: {d.get('label','?')} {d.get('size_str','')} [{d.get('verdict','?')}]"
                       for i,d in enumerate(all_defects)])
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[{"text":
                f"QC in ấn bao bì. {len(all_defects)} lỗi:\n{lines}\n"
                f"Báo cáo tiếng Việt tối đa 100 từ: tổng thể, lỗi nguy hiểm nhất, khuyến nghị."}]}],
                "generationConfig":{"temperature":0.2,"maxOutputTokens":300}},
            timeout=30)
        return r.json().get('candidates',[{}])[0].get('content',{}).get('parts',[{}])[0].get('text','')
    except: return None

# ── Main API ──────────────────────────────────────────────────────────────────
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

        # Decode 2 vùng đã crop từ frontend
        aw_crop    = b64_to_cv2(aw_b64)
        print_crop = b64_to_cv2(print_b64)
        if aw_crop is None or print_crop is None:
            return jsonify({'error':'Không đọc được ảnh'}),400

        print(f"AW crop: {aw_crop.shape}, Print crop: {print_crop.shape}, DPI: {dpi}, Sens: {sensitivity}")

        # ── BƯỚC 1: ECC Alignment ──
        print_aligned, warp, align_ok = align_two_regions(aw_crop, print_crop)
        print(f"Alignment: {'OK' if align_ok else 'fallback'}")

        # ── BƯỚC 2: So sánh 5 phương pháp với adaptive threshold ──
        heat, binary, print_corrected = compare_regions(aw_crop, print_aligned, sensitivity)

        n_diff = int(np.sum(binary > 0))
        h_aw, w_aw = aw_crop.shape[:2]
        pct = n_diff / (h_aw * w_aw) * 100
        print(f"Diff pixels: {n_diff} / {h_aw*w_aw} = {pct:.1f}%")

        # ── BƯỚC 3: Extract defects ──
        defects = extract_defects(binary, heat, aw_crop, print_corrected, dpi)
        print(f"Defects found: {len(defects)}")

        # ── BƯỚC 4: Text check (Gemini) ──
        text_defects = []
        if check_text and GEMINI_API_KEY:
            text_defects = check_text_gemini(aw_crop, print_corrected)
            for td in text_defects:
                td['verdict']='FAIL'; td['size_str']=td.get('detail','')

        # ── BƯỚC 5: Draw output images ──
        result_color  = draw_defects(print_corrected, defects, text_defects)
        result_dotmap = make_dotmap(print_corrected, defects + text_defects, heat)

        all_defects = defects + [dict(d,severity='high') for d in text_defects]
        ai_summary  = get_ai_summary(all_defects) if all_defects else None

        physical_out = [{k:v for k,v in d.items() if k!='contour'} for d in defects]
        text_out     = [dict(d, is_text=True) for d in text_defects]

        fail_c = sum(1 for d in all_defects if d.get('verdict')=='FAIL')
        warn_c = sum(1 for d in all_defects if d.get('verdict')=='WARN')
        verdict = 'FAIL' if fail_c>0 else ('WARN' if warn_c>0 else 'PASS')

        return jsonify({
            'verdict':verdict, 'defect_count':len(all_defects),
            'fail_count':fail_c, 'warn_count':warn_c,
            'physical_count':len(defects), 'text_count':len(text_defects),
            'defects': physical_out + text_out,
            'result_color':  cv2_to_b64(result_color),
            'result_dotmap': cv2_to_b64(result_dotmap),
            'ai_summary': ai_summary,
            'dpi_aw': round(dpi,1),
            'aw_size': f"{aw_info.get('width_mm','?')}x{aw_info.get('height_mm','?')}mm",
            'dpi_source': aw_info.get('source','fallback'),
            'align_ok': align_ok,
            'diff_pct': round(pct,1),
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
    print(f'QC Pro running at http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
