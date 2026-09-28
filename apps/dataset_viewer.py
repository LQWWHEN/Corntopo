#!/usr/bin/env python3
"""CornTopo 轻量数据查看器。

支持两种模式：读取 paired_dataset.xlsx 查看 pre/post 配对，或直接读取
一个未配对 CUR 数据目录逐文件查看。两种模式都支持二维矩阵可视化和
通过 IGNORE_ 前缀进行可恢复的软删除。

示例：
    python dataset_viewer.py \
        --paired-excel data/processed/paired_dataset.xlsx \
        --pre-dir data/pre-op \
        --post-dir data/post-op

    python dataset_viewer.py --dataset-dir data/raw_cur
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from scripts.corn_topo_dataset import (
    DEFAULT_CURVATURE_MATRICES,
    convert_radius_to_curvature,
    parse_filename,
    parse_stacked_matrices,
)


DEFAULT_IGNORE_PREFIX = "IGNORE_"


def normalize(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def has_ignore_prefix(path: Path | None, prefix: str) -> bool:
    return bool(path and prefix and path.name.lower().startswith(prefix.lower()))


def make_key(row: pd.Series) -> str:
    return "_".join(normalize(row.get(c)) for c in ("姓氏", "名称", "眼别"))


def format_timestamp(date_value: Any, time_value: Any) -> str:
    """将文件名或 Excel 中的日期/时间统一显示为 YYYY-MM-DD HH:MM:SS。"""
    if date_value is None or time_value is None:
        return ""
    date_text = str(date_value).strip()
    time_text = str(time_value).strip()
    if not date_text or not time_text or date_text.lower() == "nan" or time_text.lower() == "nan":
        return ""
    date_digits = re.sub(r"\D", "", date_text)
    time_digits = re.sub(r"\D", "", time_text)
    for fmt in ("%d%m%Y%H%M%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(date_digits + time_digits, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return date_text + " " + time_text


class ViewerState:
    def __init__(
        self,
        paired_excel: str | Path | None,
        dataset_dir: str | Path | None,
        pre_dir: str | Path | None,
        post_dir: str | Path | None,
        ignore_prefix: str,
    ) -> None:
        if bool(paired_excel) == bool(dataset_dir):
            raise ValueError("必须且只能指定 paired_excel 或 dataset_dir 其中一种模式")
        self.pre_dir = Path(pre_dir).resolve() if pre_dir else None
        self.post_dir = Path(post_dir).resolve() if post_dir else None
        self.ignore_prefix = ignore_prefix
        self.mode = "paired" if paired_excel else "unpaired"
        if paired_excel:
            self.data = pd.read_excel(Path(paired_excel).resolve(), dtype=object)
            if "label_index" not in self.data.columns:
                raise ValueError("paired Excel 缺少 label_index 列")
        else:
            root = Path(dataset_dir).resolve()
            if not root.exists() or not root.is_dir():
                raise FileNotFoundError("未配对数据目录不存在: " + str(root))
            records = []
            for index, path in enumerate(self.discover_dataset_files(root), start=1):
                meta = parse_filename(path) or {}
                records.append({
                    "label_index": index,
                    "姓氏": meta.get("姓氏", ""),
                    "名称": meta.get("名称", ""),
                    "眼别": meta.get("眼别", ""),
                    "pre_检查日期": meta.get("检查日期", ""),
                    "pre_检查时间": meta.get("检查时间", ""),
                    "pre_source_file": str(path),
                    "file_name": path.name,
                })
            if not records:
                raise FileNotFoundError("目录中没有 CUR 文件: " + str(root))
            self.data = pd.DataFrame(records)
        self.data["_viewer_key"] = self.data.apply(make_key, axis=1)
        self.lock = threading.RLock()

    def discover_dataset_files(self, root: Path) -> list[Path]:
        return sorted(
            path for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".csv"
            and "cur" in path.stem.lower()
        )

    def row(self, sample_id: int) -> pd.Series:
        matches = self.data[self.data["label_index"].astype(str) == str(sample_id)]
        if matches.empty:
            raise KeyError("不存在的 label_index: " + str(sample_id))
        return matches.iloc[0]

    def value_path(self, value: Any) -> Path | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        text = str(value).strip()
        return Path(text) if text else None

    def fallback_find(self, folder: Path | None, key: str) -> Path | None:
        if folder is None or not folder.exists():
            return None
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".csv":
                continue
            if "cur" not in path.stem.lower() or has_ignore_prefix(path, self.ignore_prefix):
                continue
            meta = parse_filename(path)
            if not meta:
                continue
            candidate = "_".join(normalize(meta.get(c)) for c in ("姓氏", "名称", "眼别"))
            if candidate == key:
                return path.resolve()
        return None

    def resolve_path(self, value: Any, folder: Path | None, key: str) -> Path | None:
        path = self.value_path(value)
        if path and path.exists():
            return path
        if path and self.ignore_prefix:
            ignored = path.with_name(self.ignore_prefix + path.name)
            if ignored.exists():
                return ignored
        return self.fallback_find(folder, key)

    def pair_paths(self, row: pd.Series) -> tuple[Path | None, Path | None]:
        key = row["_viewer_key"]
        return (
            self.resolve_path(row.get("pre_source_file"), self.pre_dir, key),
            self.resolve_path(row.get("post_source_file"), self.post_dir, key),
        )

    def row_timestamp(self, row: pd.Series, prefix: str, path: Path | None) -> str:
        date_value = row.get(prefix + "检查日期", "")
        time_value = row.get(prefix + "检查时间", "")
        result = format_timestamp(date_value, time_value)
        if result or not path:
            return result
        meta = parse_filename(path)
        if not meta:
            return ""
        return format_timestamp(meta.get("检查日期"), meta.get("检查时间"))

    def samples(self, query: str = "", show_ignored: bool = False) -> list[dict[str, Any]]:
        query = normalize(query)
        result = []
        for _, row in self.data.iterrows():
            pre, post = self.pair_paths(row)
            ignored_pre = has_ignore_prefix(pre, self.ignore_prefix)
            ignored_post = has_ignore_prefix(post, self.ignore_prefix)
            ignored = ignored_pre or ignored_post
            searchable = " ".join(
                str(row.get(c, ""))
                for c in ("label_index", "姓氏", "名称", "眼别", "_viewer_key", "file_name")
            ).lower()
            if query and query not in searchable:
                continue
            if ignored and not show_ignored:
                continue
            pre_timestamp = self.row_timestamp(row, "pre_", pre)
            post_timestamp = self.row_timestamp(row, "post_", post)
            result.append({
                "id": int(row["label_index"]),
                "key": row["_viewer_key"],
                "surname": str(row.get("姓氏", "")),
                "given_name": str(row.get("名称", "")),
                "eye": str(row.get("眼别", "")),
                "pre_file": str(pre) if pre else "",
                "post_file": str(post) if post else "",
                "pre_timestamp": pre_timestamp,
                "post_timestamp": post_timestamp,
                "mode": self.mode,
                "ignored": ignored,
                "partial_ignore": ignored_pre != ignored_post,
            })
        return result

    def matrix_names(self, sample_id: int) -> list[str]:
        row = self.row(sample_id)
        names: list[str] = []
        for path in self.pair_paths(row):
            if not path or not path.exists():
                continue
            for matrix in parse_stacked_matrices(path):
                if matrix["name"] not in names:
                    names.append(matrix["name"])
        return names

    @staticmethod
    def matrix_json(values: np.ndarray) -> list[list[float | None]]:
        return [
            [None if not np.isfinite(v) else float(v) for v in line]
            for line in values
        ]

    def detail(self, sample_id: int, matrix_name: str) -> dict[str, Any]:
        row = self.row(sample_id)
        pre_path, post_path = self.pair_paths(row)
        parsed: dict[str, np.ndarray] = {}
        for phase, path in (("pre", pre_path), ("post", post_path)):
            if not path or not path.exists():
                continue
            matrices = convert_radius_to_curvature(
                parse_stacked_matrices(path),
                set(DEFAULT_CURVATURE_MATRICES),
            )
            for matrix in matrices:
                if matrix["name"] == matrix_name:
                    parsed[phase] = (
                        matrix["curvature"]
                        if matrix["curvature"] is not None
                        else matrix["values"]
                    )
                    break
        reference = parsed.get("pre", parsed.get("post"))
        if reference is None:
            raise FileNotFoundError("无法读取矩阵 " + matrix_name)
        pre = parsed.get("pre", np.full_like(reference, np.nan))
        post = parsed.get("post", np.full_like(reference, np.nan))
        both = np.concatenate([pre[np.isfinite(pre)], post[np.isfinite(post)]])
        vmin, vmax = (float(both.min()), float(both.max())) if both.size else (0.0, 1.0)
        difference = np.abs(post - pre)
        valid_difference = difference[np.isfinite(difference)]
        diff_max = float(valid_difference.max()) if valid_difference.size else 1.0
        return {
            "id": int(sample_id),
            "key": row["_viewer_key"],
            "matrix_name": matrix_name,
            "shape": list(reference.shape),
            "pre": self.matrix_json(pre),
            "post": self.matrix_json(post),
            "difference": self.matrix_json(difference),
            "vmin": vmin,
            "vmax": vmax,
            "diff_max": max(diff_max, 1.0),
            "pre_file": str(pre_path) if pre_path else "",
            "post_file": str(post_path) if post_path else "",
            "pre_timestamp": self.row_timestamp(row, "pre_", pre_path),
            "post_timestamp": self.row_timestamp(row, "post_", post_path),
            "mode": self.mode,
        }

    def rename_with_prefix(self, path: Path, ignored: bool) -> Path:
        if ignored:
            if has_ignore_prefix(path, self.ignore_prefix):
                return path
            target = path.with_name(self.ignore_prefix + path.name)
        else:
            if not has_ignore_prefix(path, self.ignore_prefix):
                return path
            target = path.with_name(path.name[len(self.ignore_prefix):])
        if target.exists():
            raise FileExistsError("目标文件已存在，拒绝覆盖: " + str(target))
        path.rename(target)
        return target

    def set_ignored(self, sample_id: int, ignored: bool) -> dict[str, Any]:
        row = self.row(sample_id)
        paths = [p for p in self.pair_paths(row) if p and p.exists()]
        expected_count = 2 if self.mode == "paired" else 1
        if len(paths) != expected_count:
            raise FileNotFoundError("pre/post 文件不完整，无法执行成对软删除")
        targets = []
        for path in paths:
            if ignored and not has_ignore_prefix(path, self.ignore_prefix):
                targets.append(path.with_name(self.ignore_prefix + path.name))
            elif not ignored and has_ignore_prefix(path, self.ignore_prefix):
                targets.append(path.with_name(path.name[len(self.ignore_prefix):]))
            else:
                targets.append(path)
        for source, target in zip(paths, targets):
            if source != target and target.exists():
                raise FileExistsError("目标文件已存在，拒绝覆盖: " + str(target))
        result = [str(self.rename_with_prefix(path, ignored)) for path in paths]
        return {"id": int(sample_id), "ignored": ignored, "paths": result}


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CornTopo 数据查看器</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#172033;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
header{height:56px;background:#182536;color:white;display:flex;align-items:center;padding:0 18px;gap:16px}header h1{font-size:18px;margin:0}header small{color:#b7c5d8}
.layout{display:grid;grid-template-columns:290px 1fr;height:calc(100vh - 56px)}.side{background:white;border-right:1px solid #dce2ea;display:flex;flex-direction:column;min-width:0}
.tools{padding:12px;border-bottom:1px solid #e3e8ef;display:grid;gap:8px}input,select,button{font:inherit;border:1px solid #cbd5e1;border-radius:6px;padding:7px 9px;background:white}button{cursor:pointer}button:hover{background:#eef5ff}
.danger{background:#fff1f0;border-color:#ffaaa4;color:#b42318}.primary{background:#2563eb;color:white;border-color:#2563eb}.list{overflow:auto;flex:1}.item{padding:10px 12px;border-bottom:1px solid #edf0f4;cursor:pointer}.item:hover,.item.active{background:#edf5ff}.item .name{font-weight:600}.item .meta{color:#667085;font-size:12px;margin-top:4px}.item.ignored{opacity:.55}.badge{font-size:11px;padding:2px 5px;border-radius:4px;background:#fff0d9;color:#9a5b00}
.main{overflow:auto;padding:18px 20px}.bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:14px}.bar h2{font-size:18px;margin:0 12px 0 0}.grow{flex:1}.status{color:#667085}.info{background:white;border:1px solid #dce2ea;border-radius:8px;padding:12px;margin-bottom:14px;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}.info b{display:block;font-size:12px;color:#667085;margin-bottom:3px}.info span{overflow-wrap:anywhere}.cards{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:14px}.card{background:white;border:1px solid #dce2ea;border-radius:8px;padding:10px}.card h3{font-size:14px;margin:0 0 8px}.card canvas{display:block;width:100%;aspect-ratio:1;border:1px solid #e3e8ef;background:#111;image-rendering:pixelated}.legend{display:flex;justify-content:space-between;color:#667085;font-size:12px;margin-top:5px}.error{color:#b42318;background:#fff1f0;border:1px solid #ffccc7;border-radius:6px;padding:10px;margin-bottom:12px}
@media(max-width:950px){.layout{grid-template-columns:230px 1fr}.cards{grid-template-columns:1fr 1fr}.info{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.layout{display:block;height:auto}.side{height:290px}.main{min-height:calc(100vh - 346px)}.cards{grid-template-columns:1fr}}
</style></head><body>
<header><h1>CornTopo 数据查看器</h1><small id="head">正在加载…</small></header>
<div class="layout"><aside class="side"><div class="tools">
<input id="search" placeholder="搜索姓名、眼别或 label"><label><input id="show" type="checkbox"> 显示已软删除</label><select id="matrix"></select>
</div><div id="list" class="list"></div></aside>
<main class="main"><div class="bar"><h2 id="title">请选择样本</h2><button id="prev">上一条</button><button id="next">下一条</button><span class="grow"></span><button id="toggle" class="danger">软删除当前配对</button><span id="status" class="status"></span></div>
<div id="error"></div><div id="info" class="info"></div><div class="cards">
<section class="card"><h3 id="preTitle">Pre-op</h3><canvas id="pre"></canvas><div id="preLegend" class="legend"></div></section>
<section class="card"><h3 id="postTitle">Post-op</h3><canvas id="post"></canvas><div id="postLegend" class="legend"></div></section>
<section class="card"><h3 id="diffTitle">绝对差异 |Post - Pre|</h3><canvas id="diff"></canvas><div id="diffLegend" class="legend"></div></section>
</div></main></div>
<script>
const S={items:[],id:null,detail:null};const $=x=>document.getElementById(x);
function esc(x){return String(x==null?'':x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function api(url,opt){const r=await fetch(url,opt);const j=await r.json();if(!r.ok)throw Error(j.error||'请求失败');return j;}
function col(t){t=Math.max(0,Math.min(1,t));const a=[[0,45,90,220],[.25,0,190,220],[.5,20,190,60],[.75,245,220,0],[1,210,30,30]];for(let i=1;i<a.length;i++)if(t<=a[i][0]){const x=a[i-1],y=a[i],p=(t-x[0])/(y[0]-x[0]);return[x[1]+(y[1]-x[1])*p,x[2]+(y[2]-x[2])*p,x[3]+(y[3]-x[3])*p];}return a[4].slice(1);}
function draw(c,vs,mi,ma){const h=vs.length,w=vs[0].length;c.width=w;c.height=h;const ctx=c.getContext('2d'),im=ctx.createImageData(w,h);for(let y=0;y<h;y++)for(let x=0;x<w;x++){const v=vs[y][x],i=((h-1-y)*w+x)*4;if(v===null){im.data[i]=im.data[i+1]=im.data[i+2]=12;im.data[i+3]=255;}else{const z=col((v-mi)/(ma-mi||1));im.data[i]=z[0];im.data[i+1]=z[1];im.data[i+2]=z[2];im.data[i+3]=255;}}ctx.putImageData(im,0,0);}
function legend(id,a,b){$(id).innerHTML='<span>'+Number(a).toFixed(3)+'</span><span>'+Number(b).toFixed(3)+'</span>';}
function renderList(){const box=$('list');box.innerHTML='';S.items.forEach(s=>{const d=document.createElement('div');d.className='item '+(s.id===S.id?'active ':'')+(s.ignored?'ignored':'');const time=s.post_timestamp?' · Pre '+s.pre_timestamp+' · Post '+s.post_timestamp:(s.pre_timestamp?' · '+s.pre_timestamp:'');d.innerHTML='<div class="name">'+esc(s.id)+' · '+esc(s.surname)+' '+esc(s.given_name)+' '+esc(s.eye)+(s.ignored?'<span class="badge">已忽略</span>':'')+'</div><div class="meta">'+esc(s.key)+esc(time)+(s.partial_ignore?' · 配对不完整':'')+'</div>';d.onclick=()=>select(s.id);box.appendChild(d);});}
async function loadList(){const q=new URLSearchParams({query:$('search').value,show_ignored:$('show').checked?'1':'0'});S.items=await api('/api/samples?'+q);$('head').textContent='当前显示 '+S.items.length+' 条';renderList();if(S.id===null&&S.items.length)select(S.items[0].id);else if(S.id!==null&&!S.items.some(x=>x.id===S.id)){S.id=null;S.detail=null;renderDetail();}}
async function select(id){S.id=id;renderList();$('error').innerHTML='';try{const ns=await api('/api/matrices?id='+id),old=$('matrix').value;$('matrix').innerHTML=ns.map(x=>'<option>'+esc(x)+'</option>').join('');if(ns.includes(old))$('matrix').value=old;await loadDetail();}catch(e){$('error').innerHTML='<div class="error">'+esc(e.message)+'</div>';}}
async function loadDetail(){if(S.id===null||!$('matrix').value)return;S.detail=await api('/api/detail?id='+S.id+'&matrix='+encodeURIComponent($('matrix').value));renderDetail();}
function renderDetail(){const d=S.detail;if(!d){$('title').textContent='请选择样本';$('info').innerHTML='';return;}const s=S.items.find(x=>x.id===S.id),unpaired=!d.post_file;$('title').textContent=d.id+' · '+d.key;$('preTitle').textContent=unpaired?'当前文件':'Pre-op';$('postTitle').textContent=unpaired?'Post-op（不适用）':'Post-op';$('diffTitle').textContent=unpaired?'绝对差异（不适用）':'绝对差异 |Post - Pre|';$('info').innerHTML='<div><b>患者</b><span>'+esc(s.surname)+' '+esc(s.given_name)+'</span></div><div><b>眼别</b><span>'+esc(s.eye)+'</span></div><div><b>矩阵</b><span>'+esc(d.matrix_name)+' ('+d.shape.join(' × ')+')</span></div><div><b>状态</b><span>'+esc(s.ignored?'已软删除':'正常')+'</span></div><div><b>'+(unpaired?'检查时间':'Pre 时间')+'</b><span>'+esc(d.pre_timestamp||'未知')+'</span></div><div><b>Post 时间</b><span>'+esc(d.post_timestamp||'不适用')+'</span></div><div style="grid-column:1/-1"><b>'+(unpaired?'当前文件':'Pre 文件')+'</b><span>'+esc(d.pre_file)+'</span></div><div style="grid-column:1/-1"><b>Post 文件</b><span>'+esc(d.post_file||'不适用')+'</span></div>';draw($('pre'),d.pre,d.vmin,d.vmax);draw($('post'),d.post,d.vmin,d.vmax);draw($('diff'),d.difference,0,d.diff_max);legend('preLegend',d.vmin,d.vmax);if(unpaired){$('postLegend').textContent='不适用';$('diffLegend').textContent='不适用';}else{legend('postLegend',d.vmin,d.vmax);legend('diffLegend',0,d.diff_max);}$('toggle').textContent=s.ignored?(unpaired?'恢复当前文件':'恢复当前配对'):(unpaired?'软删除当前文件':'软删除当前配对');$('toggle').className=s.ignored?'primary':'danger';}
function move(n){const i=S.items.findIndex(x=>x.id===S.id),x=S.items[i+n];if(x)select(x.id);}
async function toggle(){if(S.id===null)return;const s=S.items.find(x=>x.id===S.id),unpaired=!s.post_file;const action=s.ignored?'恢复':'软删除';const target=unpaired?'当前文件':'当前 pre/post 配对';if(!confirm(action+target+'？'))return;try{await api('/api/soft-delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:S.id,ignored:!s.ignored})});await loadList();}catch(e){$('error').innerHTML='<div class="error">'+esc(e.message)+'</div>';}}
$('search').oninput=()=>loadList();$('show').onchange=()=>loadList();$('matrix').onchange=()=>loadDetail();$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('toggle').onclick=toggle;loadList().catch(e=>$('error').innerHTML='<div class="error">'+esc(e.message)+'</div>');
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    state: ViewerState

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(size).decode("utf-8"))

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            query = parse_qs(parsed.query)
            if parsed.path == "/api/samples":
                self.send_json(self.state.samples(
                    query.get("query", [""])[0],
                    query.get("show_ignored", ["0"])[0] == "1",
                ))
            elif parsed.path == "/api/matrices":
                self.send_json(self.state.matrix_names(int(query["id"][0])))
            elif parsed.path == "/api/detail":
                self.send_json(self.state.detail(
                    int(query["id"][0]),
                    query.get("matrix", ["FRONT"])[0],
                ))
            else:
                self.send_json({"error": "Not Found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def do_POST(self) -> None:
        try:
            if urlparse(self.path).path != "/api/soft-delete":
                self.send_json({"error": "Not Found"}, 404)
                return
            payload = self.read_json()
            self.send_json(self.state.set_ignored(
                int(payload["id"]),
                bool(payload.get("ignored", True)),
            ))
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="CornTopo 轻量数据查看器")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--paired-excel", help="paired_dataset.xlsx进入 pre/post 配对查看模式")
    source.add_argument("--dataset-dir", help="未配对 CUR 数据目录，进入逐文件查看模式")
    parser.add_argument("--pre-dir")
    parser.add_argument("--post-dir")
    parser.add_argument("--ignore-prefix", default=DEFAULT_IGNORE_PREFIX)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    state = ViewerState(
        args.paired_excel,
        args.dataset_dir,
        args.pre_dir,
        args.post_dir,
        args.ignore_prefix,
    )
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://" + args.host + ":" + str(args.port) + "/"
    print("CornTopo 数据查看器已启动: " + url)
    print("软删除前缀: " + repr(args.ignore_prefix))
    print("按 Ctrl+C 停止服务")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
