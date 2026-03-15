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
def diameter_from_area_mm(a): return 2*math.sqrt(a/math.pi)

def judge(dtype, val_mm):
    s = QC_STANDARDS.get(dtype, QC_STANDARDS['anomaly'])
    if val_mm >= s['fail']:   return 'FAIL'
    elif val_mm >= s['warn']: return 'WARN'
    return 'PASS'

# ── DPI extraction ────────────────────────────────────────────────────────────

def get_aw_dpi(b64_str, ftype='image'):
    if ftype == 'pdf':
        return _pdf_info(b64_str)
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
                'source':f'PDF {w_mm:.1f}×{h_mm:.1f}mm@{dpi}dpi','img_cv2':img}
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
            wm = w/dpi*25.4; hm = h/dpi*25.4
            return {'dpi':dpi,'width_mm':round(wm,2),'height_mm':round(hm,2),
                    'source':f'EXIF {wm:.1f}×{hm:.1f}mm@{dpi:.0f}dpi','img_cv2':img_cv2}
        return {'dpi':None,'source':'no_dpi','img_cv2':img_cv2}
    except Exception as e:
        return {'dpi':None,'source':f'img_err:{e}','img_cv2':None}

# ── Template matching: tìm tất cả vùng giống AW trên tờ in ──────────────────

def find_layout_regions(aw_region, print_img, threshold=0.55):
    """
    Tìm tất cả vùng trên print_img giống aw_region.
    Trả về list [(x,y,w,h,score), ...]
    """
    tpl_gray = cv2.cvtColor(aw_region, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(print_img, cv2.COLOR_BGR2GRAY)
    th, tw = tpl_gray.shape[:2]
    sh, sw = src_gray.shape[:2]

    if th > sh or tw > sw:
        return []

    # Multi-scale template matching
    best_scale = 1.0
    best_val   = -1
    for scale in np.linspace(0.7, 1.3, 25):
        rw = int(tw * scale); rh = int(th * scale)
        if rw < 10 or rh < 10 or rw > sw or rh > sh:
            continue
        resized = cv2.resize(tpl_gray, (rw, rh))
        res = cv2.matchTemplate(src_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, mv, _, _ = cv2.minMaxLoc(res)
        if mv > best_val:
            best_val = mv; best_scale = scale

    # Dùng scale tốt nhất để tìm tất cả vị trí
    rw = int(tw * best_scale); rh = int(th * best_scale)
    if rw < 10 or rh < 10 or rw > sw or rh > sh:
        return []
    resized_tpl = cv2.resize(tpl_gray, (rw, rh))
    result = cv2.matchTemplate(src_gray, resized_tpl, cv2.TM_CCOEFF_NORMED)

    # Non-max suppression để tìm nhiều vùng
    regions = []
    result_copy = result.copy()
    while True:
        _, max_val, _, max_loc = cv2.minMaxLoc(result_copy)
        if max_val < threshold: break
        x, y = max_loc
        regions.append((x, y, rw, rh, float(max_val)))
        # Suppress vùng lân cận
        x1 = max(0, x - rw//2); y1 = max(0, y - rh//2)
        x2 = min(result_copy.shape[1], x + rw//2 + rw)
        y2 = min(result_copy.shape[0], y + rh//2 + rh)
        result_copy[y1:y2, x1:x2] = -1
        if len(regions) > 50: break

    return regions

# ── Multi-method comparison cho 1 vùng ──────────────────────────────────────

def compare_region(aw_crop, print_crop, dpi, sensitivity):
    """
    So sánh 2 vùng bằng 5 phương pháp, trả về heat_map tổng hợp.
    """
    # Resize về cùng kích thước
    h, w = aw_crop.shape[:2]
    pc = cv2.resize(print_crop, (w, h))

    # ── 1. Pixel diff (LAB) ──
    aL = cv2.cvtColor(aw_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    pL = cv2.cvtColor(pc,      cv2.COLOR_BGR2LAB).astype(np.float32)
    diff_lab = (np.abs(aL[:,:,0]-pL[:,:,0])*1.5 +
                np.abs(aL[:,:,1]-pL[:,:,1]) +
                np.abs(aL[:,:,2]-pL[:,:,2])) / 3.0
    map_pixel = np.clip(diff_lab / 100.0, 0, 1)

    # ── 2. SSIM per-patch ──
    ag = cv2.cvtColor(aw_crop, cv2.COLOR_BGR2GRAY)
    pg = cv2.cvtColor(pc,      cv2.COLOR_BGR2GRAY)
    win = max(3, min(7, min(h,w)//4 | 1))  # odd window
    try:
        score, ssim_map = ssim(ag, pg, win_size=win, full=True, data_range=255)
        map_ssim = np.clip(1.0 - (ssim_map + 1) / 2.0, 0, 1)
    except:
        map_ssim = np.zeros((h,w), np.float32)

    # ── 3. Edge compare (Canny) ──
    ae = cv2.Canny(ag, 50, 150).astype(np.float32) / 255.0
    pe = cv2.Canny(pg, 50, 150).astype(np.float32) / 255.0
    # Dilate để cho phép slight misalign
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    ae_d = cv2.dilate(ae, k); pe_d = cv2.dilate(pe, k)
    map_edge = np.abs(ae_d - pe_d)

    # ── 4. Gradient magnitude diff ──
    ag_f = ag.astype(np.float32); pg_f = pg.astype(np.float32)
    gx_a = cv2.Sobel(ag_f,cv2.CV_32F,1,0,ksize=3)
    gy_a = cv2.Sobel(ag_f,cv2.CV_32F,0,1,ksize=3)
    gx_p = cv2.Sobel(pg_f,cv2.CV_32F,1,0,ksize=3)
    gy_p = cv2.Sobel(pg_f,cv2.CV_32F,0,1,ksize=3)
    mag_a = np.sqrt(gx_a**2+gy_a**2)
    mag_p = np.sqrt(gx_p**2+gy_p**2)
    map_grad = np.abs(mag_a-mag_p) / (np.maximum(mag_a,mag_p)+1e-6)
    map_grad = np.clip(map_grad, 0, 1).astype(np.float32)

    # ── 5. Local contrast diff ──
    blur_a = cv2.GaussianBlur(ag_f,(5,5),0)
    blur_p = cv2.GaussianBlur(pg_f,(5,5),0)
    detail_a = np.abs(ag_f - blur_a)
    detail_p = np.abs(pg_f - blur_p)
    map_detail = np.abs(detail_a - detail_p) / 128.0
    map_detail = np.clip(map_detail, 0, 1).astype(np.float32)

    # ── Tổng hợp weighted ──
    heat = (map_pixel * 0.30 +
            map_ssim  * 0.25 +
            map_edge  * 0.20 +
            map_grad  * 0.15 +
            map_detail* 0.10)
    heat = np.clip(heat, 0, 1).astype(np.float32)

    # Threshold adaptive
    thresh_val = sensitivity / 100.0
    binary = (heat > thresh_val).astype(np.uint8) * 255

    # Morphological cleanup
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k2)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kc)

    return heat, binary, pc

# ── Phát hiện và phân loại defects từ binary map ────────────────────────────

def extract_defects(binary, heat, aw_crop, print_crop, dpi, offset_x, offset_y):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = aw_crop.shape[:2]
    defects = []
    min_area_px = math.pi * (mm_to_px(0.15, dpi))**2  # min 0.15mm radius

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < max(min_area_px, 2): continue

        x,y,bw,bh = cv2.boundingRect(cnt)
        aspect = max(bw,bh)/max(min(bw,bh),1)

        # Lấy mean heat score
        mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(mask,[cnt],-1,255,-1)
        mean_heat = float(cv2.mean(heat, mask=mask)[0])

        # Classify
        dtype = _classify(cnt, area_px, bw, bh, aspect, aw_crop, print_crop, dpi)
        if dtype is None: continue

        # Tính kích thước thật
        area_mm2    = area_to_mm2(area_px, dpi)
        diam_mm     = diameter_from_area_mm(area_mm2)
        length_mm   = px_to_mm(max(bw,bh), dpi)
        width_mm    = px_to_mm(min(bw,bh), dpi)

        if dtype in ('dot','spot','hickey'):
            measure = diam_mm
            size_str = f'⌀{diam_mm:.2f}mm'
        elif dtype in ('scratch','streak'):
            measure = length_mm
            size_str = f'{length_mm:.2f}×{width_mm:.2f}mm'
        else:
            measure = area_mm2
            size_str = f'{area_mm2:.3f}mm²'

        verdict = judge(dtype, measure)
        if verdict == 'PASS': continue

        defects.append({
            'type': dtype,
            'label': DEFECT_LABELS.get(dtype, dtype),
            'x': int(x+offset_x), 'y': int(y+offset_y),
            'w': int(bw), 'h': int(bh),
            'area_mm2': round(area_mm2,3),
            'diameter_mm': round(diam_mm,3),
            'length_mm': round(length_mm,3),
            'width_mm': round(width_mm,3),
            'size_str': size_str,
            'verdict': verdict,
            'severity': 'high' if verdict=='FAIL' else 'medium',
            'heat_score': round(mean_heat,3),
            'contour': cnt,
        })
    return defects

def _classify(cnt, area_px, bw, bh, aspect, aw_crop, pr_crop, dpi):
    perim = cv2.arcLength(cnt,True)
    circ  = 4*math.pi*area_px/perim**2 if perim>0 else 0

    if aspect > 10 and max(bw,bh) > mm_to_px(2.5,dpi): return 'scratch'
    if aspect > 4  and max(bw,bh) > mm_to_px(1.5,dpi): return 'streak'
    if circ > 0.65 and bw < mm_to_px(3,dpi):           return 'hickey'
    if circ > 0.5  and bw < mm_to_px(1.5,dpi):         return 'dot'
    if circ > 0.35 and bw < mm_to_px(6,dpi):           return 'spot'

    x,y,w2,h2 = cv2.boundingRect(cnt)
    pc = cv2.resize(pr_crop, (aw_crop.shape[1], aw_crop.shape[0]))
    ar = aw_crop[y:y+h2, x:x+w2]; pr = pc[y:y+h2, x:x+w2]
    if ar.size == 0 or pr.size == 0: return 'anomaly'
    am = float(np.mean(cv2.cvtColor(ar,cv2.COLOR_BGR2GRAY)))
    pm = float(np.mean(cv2.cvtColor(pr,cv2.COLOR_BGR2GRAY)))
    if pm > am+15: return 'missing_ink'
    if pm < am-15: return 'extra_ink'

    al = cv2.Laplacian(cv2.cvtColor(ar,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
    pl = cv2.Laplacian(cv2.cvtColor(pr,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
    if pl < al*0.4: return 'blur'
    if area_px < mm_to_px(0.2,dpi)**2: return None
    return 'anomaly'

# ── Tạo 2 loại ảnh kết quả ──────────────────────────────────────────────────

def make_color_overlay(print_img, all_regions_defects):
    """Ảnh tờ in với vùng lỗi tô màu."""
    out = print_img.copy()
    for i, d in enumerate(all_regions_defects):
        color = DEFECT_COLORS_CV.get(d['type'], (180,180,180))
        x,y,w,h = d['x'],d['y'],d['w'],d['h']
        ih,iw = out.shape[:2]
        x=max(0,x); y=max(0,y); w=min(w,iw-x); h=min(h,ih-y)
        if w<=0 or h<=0: continue

        ov = out.copy()
        cv2.rectangle(ov,(x,y),(x+w,y+h),color,-1)
        alpha = 0.35 if d['verdict']=='FAIL' else 0.22
        cv2.addWeighted(ov,alpha,out,1-alpha,0,out)
        thick = 3 if d['verdict']=='FAIL' else 2
        cv2.rectangle(out,(x,y),(x+w,y+h),color,thick)

        label = f"#{i+1} {d['label']} {d['size_str']} [{d['verdict']}]"
        font = cv2.FONT_HERSHEY_SIMPLEX; sc = 0.38
        (tw,th),_ = cv2.getTextSize(label,font,sc,1)
        by = max(y-3, th+3)
        cv2.rectangle(out,(x,by-th-3),(x+tw+5,by+1),color,-1)
        cv2.putText(out,label,(x+3,by-1),font,sc,(0,0,0),1,cv2.LINE_AA)

    return out

def make_black_dot_map(print_img, all_regions_defects, heat_maps):
    """
    Nền đen tuyền. Vị trí lỗi = copy pixel thật từ ảnh chụp (màu gốc).
    QC nhìn ảnh nền đen, thấy màu = có lỗi ở đó.
    """
    # Nền đen hoàn toàn
    out = np.zeros_like(print_img)

    for region_heat, rx, ry, rw, rh in heat_maps:
        rh2 = min(rh, print_img.shape[0]-ry)
        rw2 = min(rw, print_img.shape[1]-rx)
        if rh2<=0 or rw2<=0: continue
        heat_r = cv2.resize(region_heat.astype(np.float32),(rw2,rh2))
        # Lấy binary mask từ heat
        _, mask = cv2.threshold((heat_r*255).astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
        # Dilate nhẹ để dễ thấy
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask = cv2.dilate(mask, k)
        # Copy pixel gốc từ print_img vào vùng mask
        roi_print = print_img[ry:ry+rh2, rx:rx+rw2]
        roi_out   = out[ry:ry+rh2, rx:rx+rw2]
        roi_out[mask>0] = roi_print[mask>0]

    # Thêm viền trắng mỏng quanh mỗi defect bbox để QC dễ định vị
    for d in all_regions_defects:
        x,y,w,h = d['x'],d['y'],d['w'],d['h']
        ih,iw = out.shape[:2]
        x=max(0,x);y=max(0,y);w=min(w,iw-x);h=min(h,ih-y)
        if w>0 and h>0:
            cv2.rectangle(out,(x,y),(x+w,y+h),(255,255,255),1)

    return out

# ── Gemini text check ─────────────────────────────────────────────────────────

def check_text_gemini(aw_img, print_img):
    if not GEMINI_API_KEY: return []
    h,w = aw_img.shape[:2]
    prompt = f"""So sánh Ảnh 1 (AW gốc) và Ảnh 2 (tờ in, {w}×{h}px).
Chỉ kiểm tra VĂN BẢN: dấu thanh (sắc/huyền/hỏi/ngã/nặng), dấu câu (./,/!/?) thừa/thiếu, ký tự sai.
JSON: {{"text_defects":[{{"type":"wrong_diacritic|missing_punct|wrong_char","label":"tên VN","detail":"mô tả chi tiết cụ thể","x":0,"y":0,"w":30,"h":20}}]}}"""
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[
                {"inline_data":{"mime_type":"image/jpeg","data":img_to_b64_jpeg(aw_img)}},
                {"inline_data":{"mime_type":"image/jpeg","data":img_to_b64_jpeg(print_img)}},
                {"text":prompt}
            ]}],"generationConfig":{"temperature":0.05,"maxOutputTokens":2000}},
            timeout=45)
        raw = r.json().get('candidates',[{}])[0].get('content',{}).get('parts',[{}])[0].get('text','')
        return json.loads(raw.replace('```json','').replace('```','').strip()).get('text_defects',[])
    except Exception as e:
        print(f"Gemini err: {e}"); return []

def get_ai_summary(all_defects):
    if not GEMINI_API_KEY or not all_defects: return None
    lines = "\n".join([f"- #{i+1}: {d.get('label','?')} {d.get('size_str','')} [{d.get('verdict','?')}]"
                       for i,d in enumerate(all_defects)])
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[{"text":
                f"QC in ấn bao bì. Phát hiện {len(all_defects)} lỗi:\n{lines}\n"
                f"Báo cáo tiếng Việt ≤100 từ: tổng thể, lỗi nguy hiểm nhất, khuyến nghị."}]}],
                "generationConfig":{"temperature":0.2,"maxOutputTokens":300}},
            timeout=30)
        return r.json().get('candidates',[{}])[0].get('content',{}).get('parts',[{}])[0].get('text','')
    except: return None

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.route('/api/get_aw_info', methods=['POST','OPTIONS'])
def api_get_aw_info():
    if request.method=='OPTIONS': return '',204
    d = request.get_json()
    info = get_aw_dpi(d.get('fileData',''), d.get('fileType','image'))
    return jsonify({k:v for k,v in info.items() if k not in ('img_cv2',)})

@app.route('/api/analyze', methods=['POST','OPTIONS'])
def analyze():
    if request.method=='OPTIONS': return '',204
    data = request.get_json()
    if not data: return jsonify({'error':'No data'}),400

    aw_b64       = data.get('awImage')       # vùng AW đã crop từ canvas
    print_b64    = data.get('printImage')    # ảnh tờ in đầy đủ
    aw_file_b64  = data.get('awFileData')    # file AW gốc (PDF/PNG) để lấy DPI
    aw_type      = data.get('awFileType','image')
    sensitivity  = int(data.get('sensitivity', 30))
    check_text   = data.get('checkText', True)
    manual_dpi   = data.get('manualDpi', None)

    if not aw_b64 or not print_b64:
        return jsonify({'error':'Thiếu ảnh'}),400

    try:
        # ── DPI ──
        aw_info = {}
        if aw_file_b64:
            aw_info = get_aw_dpi(aw_file_b64, aw_type)
        dpi = float(manual_dpi) if manual_dpi else (aw_info.get('dpi') or 150.0)

        # ── Decode ──
        aw_region  = b64_to_cv2(aw_b64)    # vùng mẫu từ AW
        print_full = b64_to_cv2(print_b64)  # tờ in đầy đủ
        if aw_region is None or print_full is None:
            return jsonify({'error':'Không đọc được ảnh'}),400

        mode = data.get('mode', 'template')  # 'direct' hoặc 'template'
        all_defects = []
        heat_maps   = []
        layout_count = 1

        if mode == 'direct':
            # ── Mode trực tiếp: so sánh đúng 2 vùng đã crop ──
            # print_b64 đã là vùng crop từ frontend
            print_crop = b64_to_cv2(print_b64)
            if print_crop is None:
                return jsonify({'error':'Không đọc được vùng tờ in'}),400

            # Resize về cùng kích thước
            h_aw,w_aw = aw_region.shape[:2]
            print_resized = cv2.resize(print_crop,(w_aw,h_aw))

            heat, binary, print_aligned = compare_region(aw_region, print_resized, dpi, sensitivity)
            heat_maps.append((heat, 0, 0, w_aw, h_aw))
            region_defects = extract_defects(binary, heat, aw_region, print_aligned, dpi, 0, 0)
            all_defects.extend(region_defects)

            # Text check trên 2 vùng
            text_defects = []
            if check_text and GEMINI_API_KEY:
                text_defects = check_text_gemini(aw_region, print_aligned)
                for td in text_defects:
                    td['verdict']='FAIL'; td['size_str']=td.get('detail','')

            # Dùng print_aligned làm ảnh base cho output
            print_full = print_aligned

        else:
            # ── Mode template: tìm tất cả vùng giống AW ──
            print_full = b64_to_cv2(print_b64)
            if print_full is None:
                return jsonify({'error':'Không đọc được ảnh tờ in'}),400

            regions = find_layout_regions(aw_region, print_full, threshold=0.50)
            if not regions:
                h,w = print_full.shape[:2]
                regions = [(0,0,w,h,1.0)]
            layout_count = len(regions)

            for (rx,ry,rw,rh,score) in regions:
                rh2=min(rh,print_full.shape[0]-ry); rw2=min(rw,print_full.shape[1]-rx)
                if rh2<10 or rw2<10: continue
                print_crop = print_full[ry:ry+rh2, rx:rx+rw2]
                aw_resized = cv2.resize(aw_region,(rw2,rh2))
                heat,binary,paligned = compare_region(aw_resized,print_crop,dpi,sensitivity)
                heat_maps.append((heat,rx,ry,rw2,rh2))
                all_defects.extend(extract_defects(binary,heat,aw_resized,paligned,dpi,rx,ry))

            text_defects = []
            if check_text and GEMINI_API_KEY and regions:
                rx,ry,rw,rh,_=regions[0]
                rh2=min(rh,print_full.shape[0]-ry); rw2=min(rw,print_full.shape[1]-rx)
                pc=print_full[ry:ry+rh2,rx:rx+rw2]; ar=cv2.resize(aw_region,(rw2,rh2))
                text_defects=check_text_gemini(ar,pc)
                for td in text_defects:
                    td['x']=td.get('x',0)+rx; td['y']=td.get('y',0)+ry
                    td['verdict']='FAIL'; td['size_str']=td.get('detail','')

        # ── Tạo 2 loại ảnh output ──
        combined = all_defects + text_defects
        result_color = make_color_overlay(print_full, combined)
        result_dotmap = make_black_dot_map(print_full, combined, heat_maps)

        # ── AI summary ──
        ai_summary = get_ai_summary(combined) if combined else None

        # ── Verdicts ──
        fail_c = sum(1 for d in combined if d.get('verdict')=='FAIL')
        warn_c = sum(1 for d in combined if d.get('verdict')=='WARN')
        verdict = 'FAIL' if fail_c>0 else ('WARN' if warn_c>0 else 'PASS')

        phys_out = [{k:v for k,v in d.items() if k!='contour'} for d in all_defects]
        text_out = [dict(d, is_text=True, severity='high') for d in text_defects]

        return jsonify({
            'verdict':        verdict,
            'defect_count':   len(combined),
            'fail_count':     fail_c,
            'warn_count':     warn_c,
            'physical_count': len(all_defects),
            'text_count':     len(text_defects),
            'layout_count':   layout_count,
            'defects':        phys_out + text_out,
            'result_color':   cv2_to_b64(result_color),
            'result_dotmap':  cv2_to_b64(result_dotmap),
            'dpi_aw':         round(dpi,1),
            'aw_size':        f"{aw_info.get('width_mm','?')}×{aw_info.get('height_mm','?')}mm",
            'dpi_source':     aw_info.get('source','fallback'),
            'ai_summary':     ai_summary,
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
