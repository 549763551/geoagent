"""结果交付类工具：把当前会话的规划与工具调用轨迹汇总成结构化分析报告。

输出 Markdown（便于归档/粘贴）与 HTML（便于网页直接预览），内容全部来自真实工具结果，
不做任何编造；是智能体“结果交付”环节的落点。
"""

from __future__ import annotations

import time
from pathlib import Path

from .base import ToolResult, slugify, tool

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; max-width: 900px;
         margin: 32px auto; padding: 0 24px; color: #222; line-height: 1.7; }}
  h1 {{ font-size: 24px; border-bottom: 3px solid #4C8BF5; padding-bottom: 10px; }}
  h2 {{ font-size: 18px; margin-top: 28px; color: #1a1a1a; border-left: 4px solid #4C8BF5; padding-left: 10px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
  .step {{ background: #f7f9fc; border: 1px solid #e3e9f2; border-radius: 8px;
           padding: 12px 16px; margin: 10px 0; }}
  .step h3 {{ margin: 0 0 6px; font-size: 15px; color: #2c5f9e; }}
  .step .args {{ font-family: Consolas, monospace; font-size: 12px; color: #777; }}
  .step .out {{ font-size: 14px; margin-top: 6px; }}
  .fail {{ border-color: #f0b8b8; background: #fdf4f4; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 10px 0; }}
  ul {{ padding-left: 22px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f4fa; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">生成时间：{generated} ｜ 会话：{session_id} ｜ 工具调用：{n_records} 次 ｜ 产物：{n_artifacts} 项</div>
{body}
</body>
</html>
"""


def _arg_text(args: dict) -> str:
    parts = [f"{k}={v}" for k, v in args.items() if v not in (None, "", [])]
    return ", ".join(parts)


@tool(
    name="generate_report",
    description="把本次会话已经完成的分析过程汇总为一份结构化分析报告（Markdown + HTML），"
                "包含任务描述、使用数据与方法、逐步分析结论、关键图表与产物清单。"
                "适合在分析完成后回答“形成一份报告/总结/汇总”。报告内容完全来自真实工具结果。",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "报告标题，可留空自动生成"},
            "question": {"type": "string", "description": "任务描述/原始问题，可留空取最近一次用户提问"},
        },
        "required": [],
    },
    category="结果交付",
)
def generate_report(ctx, title: str | None = None, question: str | None = None) -> ToolResult:
    session = ctx.session
    records = [r for r in session.records if r.name != "generate_report"]
    if not records:
        return ToolResult.failure("当前会话尚无可汇总的分析步骤，请先完成至少一次分析")

    query = (question or session.last_user_message() or "").strip()
    report_title = (title or "").strip() or (query[:40] if query else "遥感栅格分析报告")

    ok_records = [r for r in records if r.ok]
    datasets: list[str] = []
    tools_used: list[str] = []
    for rec in records:
        if rec.name not in tools_used:
            tools_used.append(rec.name)
        for key in ("dataset", "index", "target"):
            val = rec.data.get(key) or rec.args.get(key)
            if isinstance(val, str) and val and val not in datasets:
                datasets.append(val)

    steps_md: list[str] = []
    steps_html: list[str] = []
    for rec in records:
        step_title = rec.step_title or rec.name
        args_text = _arg_text(rec.args)
        status = "✅" if rec.ok else "⚠️"
        steps_md.append(
            f"### 步骤 {rec.seq}：{step_title}\n\n"
            f"- 工具：`{rec.name}`{('（' + args_text + '）') if args_text else ''}\n"
            f"- 结果：{status} {rec.summary}\n"
        )
        cls = "step" if rec.ok else "step fail"
        steps_html.append(
            f'<div class="{cls}"><h3>步骤 {rec.seq}：{step_title}</h3>'
            f'<div class="args">{rec.name}({args_text})</div>'
            f'<div class="out">{status} {rec.summary}</div></div>'
        )

    figures = [a for a in session.artifacts if a.type == "image"]
    tables = [a for a in session.artifacts if a.type in ("csv", "tif")]
    fig_md: list[str] = []
    fig_html: list[str] = []
    for art in figures:
        rel = Path(art.path).relative_to(ctx.settings.output_dir).as_posix()
        fig_md.append(f"![{art.title}](../{rel})\n\n*图：{art.title}*")
        fig_html.append(f'<img src="../{rel}" alt="{art.title}"/><div class="meta">图：{art.title}</div>')

    products_md = [f"- {a.title}（{a.type}）" for a in session.artifacts]
    products_html = "".join(f"<li>{a.title}（{a.type}）</li>" for a in session.artifacts)

    body_parts = []
    body_parts.append("<h2>一、任务描述</h2>")
    body_parts.append(f"<p>{query or '（未记录任务描述）'}</p>")
    body_parts.append("<h2>二、数据与方法</h2>")
    body_parts.append(
        f"<p>共调用 {len(records)} 次工具（成功 {len(ok_records)} 次），"
        f"涉及工具：{('、'.join(tools_used)) or '—'}。</p>"
        f"<p>涉及数据/对象：{('、'.join(datasets)) or '—'}。</p>"
    )
    body_parts.append("<h2>三、分析过程与结论</h2>")
    body_parts.append("".join(steps_html))
    body_parts.append("<h2>四、关键图表</h2>")
    body_parts.append("".join(fig_html) or "<p>本次未生成图片。</p>")
    body_parts.append("<h2>五、产物清单</h2>")
    body_parts.append(f"<ul>{products_html}</ul>")
    body_html = "\n".join(body_parts)

    md_text = "\n".join([
        f"# {report_title}",
        "",
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} ｜ 会话：{session.id} ｜ "
        f"工具调用 {len(records)} 次（成功 {len(ok_records)} 次）｜ 产物 {len(session.artifacts)} 项",
        "",
        "## 一、任务描述", "", query or "（未记录任务描述）", "",
        "## 二、数据与方法", "",
        f"- 涉及工具：{'、'.join(tools_used) or '—'}",
        f"- 涉及数据/对象：{'、'.join(datasets) or '—'}",
        f"- 有效步骤：{len(ok_records)}/{len(records)}",
        "",
        "## 三、分析过程与结论", "",
        "\n".join(steps_md),
        "## 四、关键图表", "",
        "\n\n".join(fig_md) or "（本次未生成图片）", "",
        "## 五、产物清单", "",
        "\n".join(products_md) or "（无）", "",
    ])

    reports_dir = ctx.out_dir("reports")
    tag = slugify(report_title)
    md_path = reports_dir / f"{tag}.md"
    md_path.write_text(md_text, encoding="utf-8")
    html_path = reports_dir / f"{tag}.html"
    html_path.write_text(
        _HTML_TEMPLATE.format(title=report_title, session_id=session.id,
                              n_records=len(records), n_artifacts=len(session.artifacts),
                              generated=time.strftime("%Y-%m-%d %H:%M:%S"), body=body_html),
        encoding="utf-8",
    )

    summary = (
        f"已生成分析报告《{report_title}》，汇总 {len(records)} 次工具调用（成功 {len(ok_records)} 次）、"
        f"{len(session.artifacts)} 项产物（其中图 {len(figures)} 张、表/栅格 {len(tables)} 项）。"
        f"报告含任务描述、数据方法、逐步结论、关键图表与产物清单，已输出 Markdown 与 HTML 两种格式。"
    )
    return ToolResult(
        ok=True, summary=summary,
        data={"title": report_title, "records": len(records), "ok_records": len(ok_records),
              "artifacts": len(session.artifacts), "tools": tools_used, "datasets": datasets},
        artifacts=[
            ctx.artifact(md_path, "report", f"{report_title}（Markdown）"),
            ctx.artifact(html_path, "report", f"{report_title}（HTML）"),
        ],
    )
