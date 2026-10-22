#!/usr/bin/env python3
"""Render the full-stack report to HTML/PDF."""
import base64
import os
import shutil
import subprocess
from pathlib import Path
import markdown

ROOT = Path(".").resolve()
MD = ROOT / "docs" / "REPORT_QKV_V_FULLSTACK_HMMA.md"
FIG = ROOT / "docs" / "figures" / "qkv_v_fullstack_summary.png"
HTML = ROOT / "docs" / "REPORT_QKV_V_FULLSTACK_HMMA.html"
PDF = ROOT / "docs" / "REPORT_QKV_V_FULLSTACK_HMMA.pdf"
text = MD.read_text(encoding="utf-8")
body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
img = base64.b64encode(FIG.read_bytes()).decode("ascii")
body = body.replace('src="figures/qkv_v_fullstack_summary.png"',
                    'src="data:image/png;base64,%s"' % img)
html = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>QKV-V 全链路多检查点报告</title><style>
@page { size: A4; margin: 16mm 15mm; }
body { font-family: "Microsoft YaHei","SimSun","Times New Roman",serif;
       font-size: 10.5pt; line-height: 1.55; color:#222; max-width:920px; margin:0 auto; }
h1 { font-size:17pt; } h2 { font-size:13pt; border-bottom:1px solid #ddd; padding-bottom:2pt; }
h3 { font-size:11pt; }
table { border-collapse:collapse; width:100%; margin:7pt 0 10pt; font-size:9pt; page-break-inside:avoid; }
th,td { border:1px solid #bbb; padding:3pt 4pt; vertical-align:top; } th { background:#f2f2f2; }
code { font-family:Consolas,monospace; font-size:9pt; background:#f4f4f4; padding:1px 3px; }
pre { background:#f7f7f7; padding:7pt; border:1px solid #e2e2e2; white-space:pre-wrap; font-size:9pt; }
img { max-width:100%; display:block; margin:8pt auto; }
blockquote { border-left:3px solid #bbb; padding-left:9pt; color:#444; }
hr { border:none;border-top:1px solid #ddd;margin:14pt 0; }
</style></head><body>__BODY__</body></html>""".replace("__BODY__", body)
HTML.write_text(html, encoding="utf-8")
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
       "--allow-file-access-from-files", "--virtual-time-budget=5000",
       "--print-to-pdf=%s" % PDF, HTML.as_uri()]
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                   text=True, timeout=120)
print(p.stdout[-1000:])
for dst in (ROOT / "REPORT_QKV_V_FULLSTACK_HMMA.pdf",
            ROOT / "deliverables" / "REPORT_QKV_V_FULLSTACK_HMMA.pdf",
            Path(os.environ["USERPROFILE"]) / "Desktop" / "gpu-sdc-lab-llm-figures" / "REPORT_QKV_V_FULLSTACK_HMMA.pdf"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PDF, dst)
print("pdf", PDF.stat().st_size)
