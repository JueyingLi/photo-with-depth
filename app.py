"""Depth Photo Studio —— 本地桌面 app 后端(FastAPI)。

跑法:
    python app.py           # 自动打开 http://localhost:8000

流程:上传照片 → 后台跑 pipeline.process_image → 轮询进度 → 完成后进编辑器。
所有处理在本机,产物存 outputs/cases/<id>/。
"""
import shutil
import threading
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from pipeline import process_image

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "outputs" / "cases"
CASES.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Depth Photo Studio")
jobs = {}   # job_id -> {status, step, total, msg, case}


def _run_job(job_id, upath, case_dir):
    def prog(i, total, msg):
        jobs[job_id].update(step=i, total=total, msg=msg)
    try:
        process_image(upath, case_dir, progress=prog)
        jobs[job_id].update(status="done", step=5)
    except Exception as e:
        import traceback; traceback.print_exc()
        jobs[job_id].update(status="error", msg=str(e))


@app.post("/api/process")
async def api_process(file: UploadFile = File(...)):
    cid = uuid.uuid4().hex[:8]
    case_dir = CASES / cid
    case_dir.mkdir(parents=True, exist_ok=True)
    upath = case_dir / ("upload" + (Path(file.filename).suffix or ".png"))
    with open(upath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    jobs[cid] = {"status": "running", "step": 0, "total": 5, "msg": "排队中", "case": cid}
    threading.Thread(target=_run_job, args=(cid, str(upath), str(case_dir)), daemon=True).start()
    return {"job": cid}


@app.get("/api/status/{job}")
def api_status(job):
    return jobs.get(job, {"status": "unknown"})


@app.post("/api/rebake")
async def api_rebake(case: str = Form(...), file: UploadFile = File(...)):
    """收到编辑后的 label 图 → 重算区域 → 重生成背景 + LDI 贴图。"""
    from regions import save_scene
    from step_3_build_regions import prepare_depth, describe_regions
    from build_background import build_background
    from build_sprites import build_sprites

    cdir = CASES / case
    if not (cdir / "scene.json").exists():
        return JSONResponse({"error": "unknown case"}, status_code=404)
    (cdir / "region_labels.png").write_bytes(await file.read())   # 覆盖为编辑后的
    label_map = np.asarray(Image.open(cdir / "region_labels.png").convert("L"))
    depth = np.asarray(Image.open(cdir / "depth_map.png").convert("L"), np.float32) / 255
    d, _ = prepare_depth(depth)
    regions = describe_regions(label_map, d)
    save_scene(label_map.astype(np.uint8), regions, cdir)
    build_background(cdir / "cropped_input.png", cdir / "scene.json",
                     cdir / "region_labels.png", cdir / "background.png")
    build_sprites(cdir / "cropped_input.png", cdir / "scene.json",
                  cdir / "region_labels.png", cdir / "sprites")
    return {"ok": True, "regions": len(regions)}


@app.get("/api/export/{case}")
def api_export(case: str):
    """把一个案例打包成单文件自包含 HTML(贴图 base64 内嵌 + 观看器),供下载/分享。"""
    import base64
    import json

    cdir = CASES / case
    scene_path = cdir / "scene.json"
    if not scene_path.exists():
        return JSONResponse({"error": "unknown case"}, status_code=404)
    scene = json.loads(scene_path.read_text())
    layers = sorted(scene["regions"], key=lambda r: r.get("layer_index", r["depth_mean"]))  # far→near
    n = scene.get("sprite_count", len(layers))
    uris = []
    for i in range(n):
        p = cdir / "sprites" / f"sprite_{i:02d}.png"
        uris.append("data:image/png;base64," + base64.b64encode(p.read_bytes()).decode())
    scene_obj = {"width": scene["width"], "height": scene["height"],
                 "layers": [{"id": l["id"], "depth_mean": l["depth_mean"]} for l in layers[:n]]}
    html = (ROOT / "viewer_template.html").read_text()
    html = html.replace("__TITLE__", f"Depth Photo — {case}")
    html = html.replace('/*__SCENE__*/ {"width":1,"height":1,"layers":[]}', "/*scene*/ " + json.dumps(scene_obj))
    html = html.replace("/*__SPRITES__*/ []", "/*sprites*/ " + json.dumps(uris))
    return Response(html, media_type="text/html",
                    headers={"Content-Disposition": f'attachment; filename="depth-{case}.html"'})


@app.get("/api/cases")
def api_cases():
    return [d.name for d in sorted(CASES.iterdir()) if (d / "scene.json").exists()]


_NOCACHE = {"Cache-Control": "no-store"}   # 前端 HTML 不缓存,改动即时生效


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "app.html", headers=_NOCACHE)


@app.get("/editor")
def editor():
    return FileResponse(ROOT / "index.html", headers=_NOCACHE)


app.mount("/outputs", StaticFiles(directory=str(ROOT / "outputs")), name="outputs")


if __name__ == "__main__":
    import webbrowser
    import uvicorn
    threading.Timer(1.3, lambda: webbrowser.open("http://localhost:8000")).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
