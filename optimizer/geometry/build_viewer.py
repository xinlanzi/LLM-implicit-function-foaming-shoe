# -*- coding: utf-8 -*-
"""
生成单文件离线 3D 查看页 `viewer/index.html`。

- three.js 与 OrbitControls 直接内嵌（不依赖 CDN，file:// 打开也能用）；
- 网格以 base64 编码的索引二进制内嵌（不是外部 STL，避免 file:// 的 CORS 限制）；
- 可切换 5 个结构、bulk/sheet、1×1×1/2×2×2、切面显示方式；
- 内嵌网格是低分辨率预览网格，完整精度请看同名 STL 文件。
"""

from __future__ import annotations

import base64
import json
import os
from typing import Dict, Tuple

from mesh_io import write_indexed_bin


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>LLM TPMS 组合结构 · 3D 抽查查看器</title>
<style>
  html, body { margin:0; padding:0; width:100%; height:100%; overflow:hidden;
               background:#0d1117; color:#e6edf3;
               font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; }
  #app { display:flex; width:100%; height:100%; }
  #side { width:300px; min-width:300px; height:100%; overflow-y:auto;
          background:#161b22; border-right:1px solid #30363d; padding:14px 14px 24px; box-sizing:border-box; }
  #stage { flex:1; position:relative; }
  canvas { display:block; }
  h1 { font-size:15px; margin:0 0 4px; color:#f0f6fc; }
  .sub { font-size:11px; color:#8b949e; line-height:1.5; margin-bottom:12px; }
  .group { margin:12px 0 6px; font-size:11px; color:#8b949e; letter-spacing:.04em; }
  button { background:#21262d; color:#c9d1d9; border:1px solid #30363d; border-radius:6px;
           padding:7px 8px; margin:3px 2px; cursor:pointer; font-size:12px; }
  button:hover { background:#30363d; }
  button.active { background:#1f6feb; border-color:#388bfd; color:#fff; }
  button.wide { width:100%; text-align:left; }
  #info { background:#0d1117; border:1px solid #30363d; border-radius:8px;
          padding:10px; font-size:11px; line-height:1.6; margin-top:12px; }
  #info .k { color:#8b949e; }
  #info .eq { color:#79c0ff; font-family:Consolas,monospace; font-size:10.5px; word-break:break-all; }
  #info .ok { color:#3fb950; } #info .warn { color:#d29922; }
  .hint { font-size:10.5px; color:#8b949e; line-height:1.6; margin-top:10px; }
  #badge { position:absolute; right:12px; top:12px; background:rgba(22,27,34,.86);
           border:1px solid #30363d; border-radius:6px; padding:6px 9px; font-size:11px; color:#8b949e; }
</style>
</head>
<body>
<div id="app">
  <div id="side">
    <h1>LLM TPMS 组合结构</h1>
    <div class="sub">高支撑 + 高弹性 · 5 个实际搜索方案<br>内嵌低分辨率预览网格；完整 STL 在 outputs/geometry/</div>
    <div class="group">结构</div>
    <div id="structBtns"></div>
    <div class="group">相 / 实体化</div>
    <div id="phaseBtns"></div>
    <div class="group">平铺</div>
    <div id="tileBtns"></div>
    <div class="group">切面封口</div>
    <div id="capBtns"></div>
    <div class="group">显示</div>
    <div id="miscBtns"></div>
    <div class="group">视角预设</div>
    <div id="viewBtns"></div>
    <div id="info"></div>
    <div class="hint">鼠标左键旋转，滚轮缩放，右键平移。<br>
      提示：选“隐藏切面”可看到盒子切开的内部通道；2×2×2 更能看出三向连通性。</div>
  </div>
  <div id="stage"><div id="badge">离线单文件 · three.js 内嵌</div></div>
</div>

<script>__THREE__</script>
<script>__ORBIT__</script>
<script>
const MESHES = __MESH_DATA__;
const SOLUTIONS = __SOLUTIONS__;

const state = { sid:"01", phase:"bulk", tile:1, capMode:"translucent",
                wireframe:false, current:null, currentKey:null, geom:null, capGeom:null };
const stage = document.getElementById("stage");

let renderer, scene, camera, controls, meshMain, meshCap, boxHelper, axesGroup;
let lastTime = 0;

function initThree() {
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(stage.clientWidth, stage.clientHeight);
  renderer.setClearColor(0x0d1117, 1);
  stage.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, stage.clientWidth/stage.clientHeight, 0.1, 2000);

  const hemi = new THREE.HemisphereLight(0xdfefff, 0x223344, 0.85);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 0.95);
  key.position.set(1.2, 1.6, 1.4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x88aaff, 0.35);
  fill.position.set(-1.4, -0.8, -1.0);
  scene.add(fill);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  window.addEventListener("resize", onResize);
  animate();
}

function onResize() {
  if (!renderer) return;
  const w = stage.clientWidth, h = stage.clientHeight;
  camera.aspect = w/h; camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

function b64ToBuffer(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i=0; i<bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function parseTPMS(b64) {
  const buf = b64ToBuffer(b64);
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0),dv.getUint8(1),dv.getUint8(2),dv.getUint8(3));
  if (magic !== "TPMS") throw new Error("bad mesh magic");
  const version = dv.getUint32(4, true);
  const nv = dv.getUint32(8, true);
  const nf = dv.getUint32(12, true);
  const posOff = 16;
  const idxOff = posOff + nv*3*4;
  const flagOff = idxOff + nf*3*4;
  const positions = new Float32Array(buf, posOff, nv*3);
  const indices = new Uint32Array(buf, idxOff, nf*3);
  const flags = new Uint8Array(buf, flagOff, nf);
  return {version, nv, nf, positions, indices, flags};
}

function buildGeometries(parsed) {
  // 分开“普通面 / 切面封口面”的索引，便于切换显示
  const solid = [], caps = [];
  for (let i=0; i<parsed.nf; i++) {
    const a = parsed.indices[i*3], b = parsed.indices[i*3+1], c = parsed.indices[i*3+2];
    if (parsed.flags[i]) caps.push(a,b,c); else solid.push(a,b,c);
  }
  const base = new THREE.BufferGeometry();
  base.setAttribute("position", new THREE.BufferAttribute(parsed.positions, 3));
  const gSolid = base.clone(); gSolid.setIndex(solid); gSolid.computeVertexNormals();
  const gAll = base.clone(); gAll.setIndex(Array.from(parsed.indices)); gAll.computeVertexNormals();
  const gCap = base.clone(); gCap.setIndex(caps); gCap.computeVertexNormals();
  return {gSolid, gAll, gCap};
}

function disposeObject(obj) {
  if (!obj) return;
  scene.remove(obj);
  if (obj.geometry) obj.geometry.dispose();
  if (obj.material) {
    if (Array.isArray(obj.material)) obj.material.forEach(m=>m.dispose());
    else obj.material.dispose();
  }
}

function clearMesh() {
  disposeObject(meshMain); disposeObject(meshCap);
  disposeObject(boxHelper); disposeObject(axesGroup);
  meshMain = meshCap = boxHelper = axesGroup = null;
}

function addAxesAndBox(size) {
  axesGroup = new THREE.Group();
  const L = size * 0.22;
  const mkArrow = (dir, color) => {
    const a = new THREE.ArrowHelper(dir.clone().normalize(), new THREE.Vector3(0,0,0), L, color, L*0.16, L*0.09);
    axesGroup.add(a);
  };
  mkArrow(new THREE.Vector3(1,0,0), 0xff5555);
  mkArrow(new THREE.Vector3(0,1,0), 0x55ff77);
  mkArrow(new THREE.Vector3(0,0,1), 0x5599ff);

  const label = (text, pos, color) => {
    const cv = document.createElement("canvas"); cv.width=64; cv.height=64;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = color; ctx.font = "bold 44px Arial"; ctx.textAlign="center"; ctx.textBaseline="middle";
    ctx.fillText(text, 32, 32);
    const tex = new THREE.CanvasTexture(cv);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({map:tex, transparent:true, depthTest:false}));
    sp.position.copy(pos); sp.scale.set(size*0.055, size*0.055, 1);
    axesGroup.add(sp);
  };
  label("X", new THREE.Vector3(L*1.12,0,0), "#ff7777");
  label("Y", new THREE.Vector3(0,L*1.12,0), "#77ff99");
  label("Z", new THREE.Vector3(0,0,L*1.12), "#77aaff");
  scene.add(axesGroup);

  const box = new THREE.Box3(new THREE.Vector3(-size/2,-size/2,-size/2),
                             new THREE.Vector3(size/2,size/2,size/2));
  boxHelper = new THREE.Box3Helper(box, 0x3d444d);
  scene.add(boxHelper);
}

function currentMeta() {
  const key = state.sid + "|" + state.phase + "|" + state.tile;
  return {key, meta: MESHES[key]};
}

function loadCurrent() {
  const {key, meta} = currentMeta();
  if (!meta) {
    document.getElementById("info").innerHTML =
      '<div class="warn">该变体未内嵌预览网格；完整 STL 见 outputs/geometry/。</div>';
    return;
  }
  if (state.currentKey === key && state.geoms) {
    applyCapMode();
    updateInfo(meta);
    return;
  }
  clearMesh();

  const parsed = parseTPMS(meta.b64);
  const geoms = buildGeometries(parsed);
  state.geoms = geoms;
  state.meta = meta;

  // 居中：几何坐标从 0 开始，平移到 [-size/2, +size/2]
  const xs = parsed.positions;
  let mn=[1e30,1e30,1e30], mx=[-1e30,-1e30,-1e30];
  for (let i=0;i<parsed.nv;i++){
    for (let k=0;k<3;k++){
      const v = xs[i*3+k];
      if (v<mn[k]) mn[k]=v; if (v>mx[k]) mx[k]=v;
    }
  }
  const center = new THREE.Vector3((mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2);
  const size = Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]);
  [geoms.gSolid, geoms.gAll, geoms.gCap].forEach(g => g.translate(-center.x,-center.y,-center.z));

  const mat = new THREE.MeshStandardMaterial({
    color: state.phase==="bulk" ? 0x5b9bd5 : 0xe08a3c,
    metalness: 0.12, roughness: 0.72, flatShading: true,
    side: THREE.DoubleSide,
  });
  meshMain = new THREE.Mesh(geoms.gSolid, mat);
  scene.add(meshMain);

  meshCap = new THREE.Mesh(geoms.gCap, new THREE.MeshStandardMaterial({
    color: 0xd8dde3, metalness: 0.0, roughness: 0.9, flatShading: true,
    transparent: true, opacity: 0.35, depthWrite: false,
    side: THREE.DoubleSide,
  }));
  scene.add(meshCap);

  addAxesAndBox(size);
  state.currentKey = key;
  state.meta = meta;
  applyCapMode();
  updateInfo(meta);
  fitView();
}

function applyCapMode() {
  if (!state.geoms || !meshMain || !meshCap) return;
  const m = state.capMode;
  if (m === "solid") {
    meshMain.geometry = state.geoms.gAll;
    meshCap.visible = false;
  } else if (m === "translucent") {
    meshMain.geometry = state.geoms.gSolid;
    meshCap.visible = true;
    meshCap.material.opacity = 0.35;
  } else {
    meshMain.geometry = state.geoms.gSolid;
    meshCap.visible = false;
  }
  meshMain.material.wireframe = state.wireframe;
}

function updateInfo(meta) {
  const capTxt = {solid:"实心", translucent:"半透明", hidden:"隐藏"}[state.capMode];
  const el = document.getElementById("info");
  el.innerHTML =
    '<div style="font-size:13px;color:#f0f6fc;margin-bottom:4px">' + meta.cn_name + '</div>' +
    '<div class="eq">' + meta.equation + '</div>' +
    '<div style="height:6px"></div>' +
    '<div><span class="k">几何代理分 score：</span>' + meta.score.toFixed(4) + '</div>' +
    '<div><span class="k">support / elastic：</span>' + meta.support.toFixed(4) + ' / ' + meta.elastic.toFixed(4) + '</div>' +
    '<div><span class="k">相对密度 ρ*：</span>' + meta.rho.toFixed(4) + '</div>' +
    '<div><span class="k">三角面 / 顶点：</span>' + meta.n_faces.toLocaleString() + ' / ' + meta.n_vertices.toLocaleString() + '</div>' +
    '<div><span class="k">watertight：</span><span class="' + (meta.watertight ? 'ok' : 'warn') + '">' + (meta.watertight ? "是" : "否") + '</span>' +
    '　<span class="k">边界边：</span>' + meta.boundary_edges + '</div>' +
    '<div><span class="k">周期实体连通分量：</span>' + meta.solid_components + '</div>' +
    '<div><span class="k">当前切面：</span>' + capTxt + '</div>' +
    '<div class="hint" style="margin-top:8px">' + meta.notes + '</div>';
}

function fitView() {
  if (!meshMain) return;
  const box = new THREE.Box3().setFromObject(meshMain);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const dist = sphere.radius / Math.sin(Math.PI/8);
  camera.position.copy(new THREE.Vector3(1.25, 1.0, 1.35).normalize().multiplyScalar(dist));
  camera.near = sphere.radius/100; camera.far = sphere.radius*20;
  camera.updateProjectionMatrix();
  controls.target.set(0,0,0);
  controls.minDistance = sphere.radius*0.4;
  controls.maxDistance = sphere.radius*6;
  controls.update();
}

function setView(kind) {
  if (!meshMain) return;
  const box = new THREE.Box3().setFromObject(meshMain);
  const r = box.getBoundingSphere(new THREE.Sphere()).radius;
  const d = r / Math.sin(Math.PI/8);
  const dirs = {iso:[1.25,1.0,1.35], front:[0,0,1.8], top:[0,1.8,0.001], right:[1.8,0,0.001]};
  const v = dirs[kind] || dirs.iso;
  const n = Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) || 1;
  camera.position.set(v[0]/n*d, v[1]/n*d, v[2]/n*d);
  controls.target.set(0,0,0);
  controls.update();
}

function animate(t) {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ------------------------------------------------------------------ UI
function btn(label, onclick, cls) {
  const b = document.createElement("button");
  b.textContent = label;
  if (cls) b.className = cls;
  b.onclick = onclick;
  return b;
}

function buildUI() {
  const sb = document.getElementById("structBtns");
  SOLUTIONS.forEach((s, idx) => {
    const b = btn(s.sid + " " + s.cn_name, () => { state.sid = s.sid; loadCurrent(); refreshActive(); }, "wide");
    b.dataset.sid = s.sid; sb.appendChild(b);
  });

  const pb = document.getElementById("phaseBtns");
  [["bulk","bulk 块体 F≤0"],["sheet","sheet 薄壁"]].forEach(([k,label]) => {
    const b = btn(label, () => { state.phase=k; loadCurrent(); refreshActive(); }, "wide");
    b.dataset.phase = k; pb.appendChild(b);
  });

  const tb = document.getElementById("tileBtns");
  [["1","1×1×1 单胞"],["2","2×2×2 平铺"]].forEach(([k,label]) => {
    const b = btn(label, () => { state.tile=parseInt(k); loadCurrent(); refreshActive(); }, "wide");
    b.dataset.tile = k; tb.appendChild(b);
  });

  const cb = document.getElementById("capBtns");
  [["solid","实心（含切面封口）"],["translucent","切面半透明"],["hidden","隐藏切面"]].forEach(([k,label]) => {
    const b = btn(label, () => { state.capMode=k; applyCapMode(); updateInfo(state.meta); refreshActive(); }, "wide");
    b.dataset.cap = k; cb.appendChild(b);
  });

  const mb = document.getElementById("miscBtns");
  const wb = btn("线框", () => { state.wireframe=!state.wireframe; applyCapMode(); refreshActive(); }, "wide");
  wb.dataset.misc="wire"; mb.appendChild(wb);

  const vb = document.getElementById("viewBtns");
  [["iso","等轴测"],["front","前视 +Z"],["right","右视 +X"],["top","顶视 +Y"]].forEach(([k,label]) => {
    const b = btn(label, () => setView(k), "wide"); vb.appendChild(b);
  });

  refreshActive();
}

function refreshActive() {
  document.querySelectorAll("#structBtns button").forEach(b => b.classList.toggle("active", b.dataset.sid===state.sid));
  document.querySelectorAll("#phaseBtns button").forEach(b => b.classList.toggle("active", b.dataset.phase===state.phase));
  document.querySelectorAll("#tileBtns button").forEach(b => b.classList.toggle("active", b.dataset.tile===String(state.tile)));
  document.querySelectorAll("#capBtns button").forEach(b => b.classList.toggle("active", b.dataset.cap===state.capMode));
  const wb = document.querySelector('#miscBtns button[data-misc="wire"]');
  if (wb) wb.classList.toggle("active", state.wireframe);
}

window.addEventListener("load", () => {
  try {
    initThree();
    buildUI();
    loadCurrent();
  } catch (e) {
    document.getElementById("stage").innerHTML =
      '<div style="padding:30px;color:#ff7b72;font-size:14px">WebGL 初始化失败：' + e.message + '</div>';
  }
});
</script>
</body>
</html>
"""


def build_viewer_html(
    out_path: str,
    mesh_b64: Dict[str, dict],
    cfg,
) -> int:
    """
    mesh_b64: key -> {b64, cn_name, equation, score, support, elastic, rho,
                     n_faces, n_vertices, boundary_edges, watertight,
                     solid_components, notes}
    """
    here = os.path.dirname(os.path.abspath(__file__))
    three = open(os.path.join(here, "assets", "three.min.js"), "r", encoding="utf-8").read()
    orbit = open(os.path.join(here, "assets", "OrbitControls.js"), "r", encoding="utf-8").read()

    solutions = [
        {"sid": s.sid, "cn_name": s.cn_name} for s in __import__("solutions").SOLUTIONS
    ]

    mesh_json = "{" + ",".join(
        f'"{k}":' + json.dumps(v, ensure_ascii=False) for k, v in mesh_b64.items()
    ) + "}"
    sol_json = json.dumps(solutions, ensure_ascii=False)

    html = (
        HTML_TEMPLATE
        .replace("__THREE__", three)
        .replace("__ORBIT__", orbit)
        .replace("__MESH_DATA__", mesh_json)
        .replace("__SOLUTIONS__", sol_json)
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return os.path.getsize(out_path)


def encode_mesh_b64(mesh, meta: dict, tmp_dir: str, key_name: str) -> dict:
    """把 MeshResult 写成索引二进制，再 base64 内嵌。"""
    tmp_path = os.path.join(tmp_dir, key_name.replace("|", "_") + ".bin")
    write_indexed_bin(tmp_path, mesh.vertices_mm, mesh.faces, mesh.cap_face_mask)
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    os.remove(tmp_path)

    meta = dict(meta)
    meta.update({
        "b64": b64,
        "rho": float(mesh.rho_voxel),
        "n_faces": int(mesh.n_faces),
        "n_vertices": int(mesh.n_vertices),
        "boundary_edges": int(mesh.boundary_edges),
        "watertight": bool(mesh.watertight),
        "solid_components": int(mesh.solid_components_periodic),
    })
    return meta
