# -*- coding: utf-8 -*-
"""生成「云栖科技」虚构知识库语料（项目一 RAG 测试数据）。

运行：
    D:\\Anaconda\\envs\\xhu\\python.exe build_corpus.py --pdf-cli <libreoffice-kit cli.js 路径>

产物写入 ../corpus/，其中 PDF 由 docx 经 LibreOffice Kit 转换后删除中间 docx。
本脚本只在 data/_generator 下留存，不参与检索。
"""
import argparse
import os
import shutil
import subprocess
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.normpath(os.path.join(BASE, "..", "corpus"))
BUILD = os.path.join(BASE, "_build")

FONT_CN = "宋体"
FONT_HEAD = "微软雅黑"


# --------------------------------------------------------------------------
# DOCX 基础工具
# --------------------------------------------------------------------------
def _set_run_font(run, size=12, bold=False, font=FONT_CN, color=None):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def _no_autospace(paragraph):
    """关闭中西文自动间距，避免转 PDF 后数字与汉字之间被插入空格。"""
    pPr = paragraph._p.get_or_add_pPr()
    for tag in ("w:autoSpaceDE", "w:autoSpaceDN"):
        el = OxmlElement(tag)
        el.set(qn("w:val"), "0")
        pPr.append(el)


def new_doc():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.6)
    sec.left_margin = sec.right_margin = Cm(2.7)
    normal = doc.styles["Normal"]
    normal.font.name = FONT_CN
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(5)
    return doc


def add_meta(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _set_run_font(p.add_run(text), size=10.5, color="595959")
    _no_autospace(p)
    p.paragraph_format.space_after = Pt(10)
    return p


def add_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_run_font(p.add_run(text), size=20, bold=True, font=FONT_HEAD)
    _no_autospace(p)
    p.paragraph_format.space_after = Pt(10)
    return p


def add_h(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    _set_run_font(p.add_run(text), size=14 if level == 1 else 12.5,
                  bold=True, font=FONT_HEAD, color="1F3864")
    _no_autospace(p)
    return p


def add_body(doc, text):
    p = doc.add_paragraph()
    _set_run_font(p.add_run(text), size=12)
    _no_autospace(p)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(3)
    _set_run_font(p.add_run(text), size=11.5)
    _no_autospace(p)
    return p


def add_related(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    _set_run_font(p.add_run("相关文档："), size=10.5, bold=True, color="404040")
    _set_run_font(p.add_run(text), size=10.5, color="404040")
    _no_autospace(p)
    return p


def write_docx(doc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc.save(path)
    print(f"[docx] {os.path.basename(path)}")


# --------------------------------------------------------------------------
# 1. 员工手册.docx
# --------------------------------------------------------------------------
def build_employee_handbook():
    doc = new_doc()
    add_meta(doc, "文档编号：YQ-HR-001 ｜ 版本：v3.0 ｜ 生效日期：2026-01-01")
    add_title(doc, "云栖科技员工手册")

    add_h(doc, "第一章 总则")
    add_body(doc, "本手册适用于云栖科技（杭州）有限公司全体正式员工、试用期员工与实习生。公司总部位于杭州市余杭区云栖小镇，现有员工480人。本手册与《绩效考核办法》（YQ-HR-007）、《信息安全规范》（YQ-SEC-008）共同构成员工行为规范体系；制度之间如有冲突，以最新发布版本为准，并由人力资源部负责解释。")

    add_h(doc, "第二章 入职与试用期")
    add_body(doc, "新员工入职当日须完成劳动合同签署、身份核验与银行卡登记，并在企业微信完成实名认证。公司统一签订三年期劳动合同，试用期为3个月，试用期工资为转正工资的80%。入职手续由人力资源部在入职前1个工作日通过OA发起，IT运维部在1个工作日内开通账号，具体流程见《IT运维手册》（YQ-IT-003）。新员工须在入职后5个工作日内完成信息安全线上培训与考试，80分及格。")

    add_h(doc, "第三章 考勤与工时")
    add_body(doc, "标准工作时间为9:00至18:00，午休12:00至13:00，每日实际在岗不少于8小时。员工可在10:00前弹性上班，但当日下班时间须相应顺延。打卡通过企业微信定位完成，漏打卡当月可补卡3次；迟到超过30分钟且未提交请假单的，记旷工半天；月度迟到累计3次，扣减当月绩效5分。每周三为“无会日”，原则上不安排跨部门会议。")

    add_h(doc, "第四章 请假与休假")
    add_body(doc, "年假按司龄计算：入职满1年5天，满3年10天，满5年15天；年假须提前3个工作日申请。病假1天以内由直属主管审批；连续病假3天以上须提交二级以上医院证明，并由人力资源部总监王琳审批。事假按日扣发工资。加班可申请调休：工作日1.5倍、周末2倍、法定节假日3倍，调休有效期为6个月。")

    add_h(doc, "第五章 转正管理")
    add_body(doc, "试用期届满前10个工作日，员工须在OA提交《转正申请表》。流程为：本人申请、直属主管评估、HRBP面谈、转正答辩30分钟，最终由人力资源部总监王琳审批。评估不合格的，可延长试用期一次，最长不超过1个月；延长后仍不合格的，公司依法解除劳动合同。")

    add_h(doc, "第六章 离职与工作交接")
    add_body(doc, "正式员工辞职须提前30天提交书面申请，试用期员工提前3天。最后工作日前须完成离职交接5项：工作交接清单、公司资产归还、系统账号注销、客户资料移交、财务借支结清。IT运维部在离职后3个工作日内完成账号回收与设备回收；离职证明由人力资源部在3个工作日内出具。未完成交接的，暂缓办理离职手续。")

    add_h(doc, "第七章 附则")
    add_body(doc, "本手册自2026年1月1日起生效。差旅与费用报销执行《报销与差旅制度》（YQ-FIN-002）；系统权限开通与变更执行《权限矩阵》（YQ-ADM-005）；绩效评估执行《绩效考核办法》（YQ-HR-007）。")

    add_related(doc, "YQ-FIN-002 报销与差旅制度、YQ-IT-003 IT运维手册、YQ-SEC-008 信息安全规范、YQ-HR-007 绩效考核办法、YQ-ADM-005 权限矩阵。")
    write_docx(doc, os.path.join(CORPUS, "员工手册.docx"))


# --------------------------------------------------------------------------
# 2. IT运维手册.md
# --------------------------------------------------------------------------
IT_MD = """文档编号：YQ-IT-003 ｜ 版本：v1.6 ｜ 生效日期：2026-02-01

# 云栖科技 IT 运维手册

> 维护部门：IT运维部（主管刘海）｜ 适用范围：总部杭州、深圳办事处、成都办事处全体员工及外包驻场人员

## 一、账号申请

新员工账号由人力资源部在入职前1个工作日通过OA批量提交，IT运维部在1个工作日内开通。在职员工申请业务系统账号，须在OA“IT服务台”提交《账号申请表》，经直属主管与部门总监审批后办理，IT运维部承诺4小时内响应。

账号命名规则为“姓名拼音首字母加工号后四位”，初始密码为工号后6位加“Yq@”，首次登录强制修改。账号权限范围以《权限矩阵》（YQ-ADM-005）为准，不得越权申请。

员工离职由人力资源部在OA发起，IT运维部须在离职当日18:00前禁用账号，并在3个工作日内完成账号删除与设备回收，回收结果同步人力资源部。转岗员工的旧权限在2个工作日内回收。

## 二、VPN 与远程办公

远程办公须使用飞连客户端接入VPN，连接地址为vpn.yunqi.example.com。VPN账号与工号一致，仅限本人使用，禁止共享或转借。连续5次密码错误将锁定30分钟，VPN连接每24小时强制重连一次。生产环境操作必须同时开启MFA，要求见《信息安全规范》（YQ-SEC-008）。

## 三、设备领用

技术岗标配MacBook Pro 14英寸（M3芯片，32GB内存，1TB硬盘），非技术岗标配ThinkPad X1 Carbon。设备折旧年限为3年，领用须在OA提交《设备领用单》，由直属主管与IT运维部双重确认。公司备用机共5台，借用不超过5个工作日。设备损坏按折旧残值赔偿，人为损坏须部门总监签字确认。

## 四、故障报修

报修入口为企业微信“IT报修”应用或服务热线0571-8888 6600。故障分级与处理时限如下：

- P1（核心系统不可用）：15分钟内响应，4小时内恢复；
- P2（部分功能异常）：2小时内响应，1个工作日内恢复；
- P3（一般使用问题）：1个工作日内响应，3个工作日内解决。

P1故障须同步通知值班人员与部门主管，并在恢复后2个工作日内提交故障复盘报告。

## 五、值班安排

IT运维部实行7×24小时轮值：工作日18:00至次日9:00，周末及法定节假日全天。值班表由刘海在每月25日前发布。值班补贴为200元/晚，法定节假日值班按3倍工资计算。值班期间须在值班群每小时巡检一次核心服务，并记录值班日志。

## 六、变更管理

生产环境变更须提前1个工作日提交《变更申请》，变更窗口为每周二、周四20:00至23:00。涉及核心数据库、网关与计费系统的重大变更，须经技术中心VP周敏审批。变更失败须在30分钟内回滚，并同步通知客户成功部。

## 七、相关文档

- YQ-SEC-008 信息安全规范
- YQ-ADM-005 权限矩阵
- YQ-HR-001 员工手册
- YQ-FIN-002 报销与差旅制度（设备采购与费用报销）
"""


def build_it_manual():
    os.makedirs(CORPUS, exist_ok=True)
    path = os.path.join(CORPUS, "IT运维手册.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(IT_MD)
    print("[md] IT运维手册.md")


# --------------------------------------------------------------------------
# 3. 产品FAQ.txt
# --------------------------------------------------------------------------
FAQ_TXT = """文档编号：YQ-PRD-004 ｜ 版本：v3.2 ｜ 生效日期：2026-04-15

云栖智能客服 产品FAQ
维护部门：产品部（总监何静）

【产品概况】
Q1：云栖智能客服是什么？
A1：云栖智能客服（YunqiBot）是面向企业的全渠道智能客服平台，包含在线客服、智能机器人与坐席辅助三大模块，当前版本为v3.2。

Q2：支持哪些接入渠道？
A2：支持网页、微信公众号、企业微信、钉钉以及API/SDK接入，可同时服务12种语言。

【版本与价格】
Q3：标准版多少钱？
A3：标准版9.8万元每年，包含50个坐席、200并发、5000条知识库容量。

Q4：企业版和私有化怎么收费？
A4：企业版28万元每年起，坐席200个、并发1000、知识库不限量；私有化部署68万元起，另收首年15%的服务费。

Q5：超出套餐怎么计费？
A5：超出并发按0.05元每次会话计费，API调用按0.02元每次计费。

【部署与运维】
Q6：支持哪些部署方式？
A6：提供公有云、私有化和混合云三种方式。公有云部署在阿里云杭州地域，多可用区容灾；私有化要求Kubernetes 1.24及以上，最低3节点、16核64G。

Q7：SLA是多少？
A7：标准版SLA为99.9%，企业版SLA为99.95%，首响P95低于800毫秒。月度可用性每低0.1%，减免当月服务费5%，累计减免不超过30%。

Q8：知识库更新后多久生效？
A8：增量更新5分钟内生效，全量重建约需30分钟。

【实施与集成】
Q9：标准实施周期多长？
A9：标准版6周，私有化10周。流程为需求确认、环境准备（3个工作日）、知识库导入、联调测试、试运行2周、正式上线。

Q10：能对接自有CRM吗？
A10：提供OpenAPI与Webhook，标准对接工作量约10个工作日。

【数据与安全】
Q11：对话记录保存多久？
A11：标准版保存180天，企业版保存365天，支持导出。数据安全要求见《信息安全规范》（YQ-SEC-008）。

Q12：是否支持信创环境？
A12：企业版支持鲲鹏、飞腾处理器与麒麟V10操作系统。

相关文档：YQ-MKT-009 客户案例集、YQ-IT-003 IT运维手册、YQ-SEC-008 信息安全规范。
"""


def build_product_faq():
    os.makedirs(CORPUS, exist_ok=True)
    path = os.path.join(CORPUS, "产品FAQ.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(FAQ_TXT)
    print("[txt] 产品FAQ.txt")


# --------------------------------------------------------------------------
# 4. 权限矩阵.xlsx
# --------------------------------------------------------------------------
def build_permission_matrix():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    head_font = Font(name="微软雅黑", size=11, bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="1F3864")
    body_font = Font(name="微软雅黑", size=10.5)
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap = Alignment(wrap_text=True, vertical="center")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def style_sheet(ws, widths, header_row=1):
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w
        for row in ws.iter_rows(min_row=header_row):
            for cell in row:
                cell.border = border
                cell.font = body_font
                cell.alignment = wrap if cell.column > 1 else center
        for cell in ws[header_row]:
            cell.font = head_font
            cell.fill = head_fill
            cell.alignment = center
        ws.freeze_panes = ws.cell(row=header_row + 1, column=1).coordinate
        ws.row_dimensions[header_row].height = 26

    # Sheet 1 说明与角色
    ws = wb.active
    ws.title = "说明与角色"
    ws["A1"] = "文档编号：YQ-ADM-005 ｜ 版本：v2.1 ｜ 生效日期：2026-05-01"
    ws["A1"].font = Font(name="微软雅黑", size=10, color="595959")
    ws["A2"] = "云栖科技 系统权限矩阵"
    ws["A2"].font = Font(name="微软雅黑", size=16, bold=True, color="1F3864")
    ws["A3"] = "维护部门：安全合规部（负责人郑凯）｜ 适用系统：客服工作台、知识库、报表中心、用户管理、计费中心、审计日志"
    ws["A3"].font = Font(name="微软雅黑", size=10, color="404040")
    ws.merge_cells("A1:F1")
    ws.merge_cells("A2:F2")
    ws.merge_cells("A3:F3")

    ws.append([])
    ws.append(["角色编号", "角色名称", "主要职责", "默认审批人", "人数上限", "备注"])
    roles = [
        ("R01", "超级管理员", "平台最高权限，负责租户、全局配置与计费参数", "技术中心VP周敏 + 安全合规部郑凯", "2人", "须双签开通"),
        ("R02", "系统管理员", "日常配置、坐席管理、知识库维护", "技术中心VP周敏", "6人", "每季度复核"),
        ("R03", "客服主管", "团队排班、质检、工单分配与升级", "客户成功部总监孙倩", "不限", "按团队设置"),
        ("R04", "客服专员", "在线接待、工单处理、知识库反馈", "客服主管", "不限", "仅限本团队数据"),
        ("R05", "数据分析师", "报表查看、导出脱敏数据", "部门总监", "10人", "导出须二次审批"),
        ("R06", "审计员", "只读审计日志，不可修改任何配置", "安全合规部郑凯", "3人", "独立于业务部门"),
        ("R07", "访客", "只读演示环境，用于售前演示", "系统管理员", "5人", "有效期7天"),
    ]
    for r in roles:
        ws.append(list(r))
    ws.append([])
    ws.append(["相关文档：YQ-SEC-008 信息安全规范、YQ-IT-003 IT运维手册、YQ-HR-001 员工手册、YQ-OPS-010 会议纪要-2026Q3。"])
    ws.merge_cells(start_row=ws.max_row, start_column=1, end_row=ws.max_row, end_column=6)
    style_sheet(ws, [10, 14, 42, 32, 10, 16], header_row=5)
    ws.row_dimensions[1].height = 20
    ws.row_dimensions[2].height = 30
    ws.row_dimensions[3].height = 22

    # Sheet 2 系统权限矩阵
    ws2 = wb.create_sheet("系统权限矩阵")
    ws2.append(["角色", "客服工作台", "知识库", "报表中心", "用户管理", "计费中心", "审计日志"])
    matrix = [
        ("超级管理员 R01", "读写", "读写", "读写", "读写", "读写", "只读"),
        ("系统管理员 R02", "读写", "读写", "读写", "读写", "只读", "只读"),
        ("客服主管 R03", "读写", "读写", "只读", "只读", "无", "无"),
        ("客服专员 R04", "读写", "只读", "无", "无", "无", "无"),
        ("数据分析师 R05", "只读", "只读", "读写", "无", "无", "无"),
        ("审计员 R06", "只读", "只读", "只读", "只读", "只读", "只读"),
        ("访客 R07", "只读", "无", "无", "无", "无", "无"),
    ]
    for r in matrix:
        ws2.append(list(r))
    style_sheet(ws2, [18, 13, 11, 11, 11, 11, 11])
    for row in ws2.iter_rows(min_row=2, min_col=2):
        for cell in row:
            cell.alignment = center

    # Sheet 3 高敏操作审批
    ws3 = wb.create_sheet("高敏操作审批")
    ws3.append(["高敏操作", "所需角色", "一级审批", "二级审批", "办理时限", "附加要求"])
    sensitive = [
        ("开通超级管理员", "R01", "技术中心VP周敏", "安全合规部郑凯", "2个工作日", "双签后方可开通，开通记录留存3年"),
        ("导出客户对话数据", "R03/R05", "部门总监", "安全合规部郑凯", "1个工作日", "数据须脱敏，导出文件加水印"),
        ("修改计费单价", "R01", "财务部总监赵磊", "CEO陈立", "3个工作日", "须附报价审批单"),
        ("删除知识库", "R02", "部门总监", "无", "1个工作日", "删除前须全量备份"),
        ("开通生产数据库直连", "R02", "技术中心VP周敏", "安全合规部郑凯", "2个工作日", "仅限排障期间，事后24小时内回收"),
        ("关闭审计日志采集", "R01", "安全合规部郑凯", "CEO陈立", "3个工作日", "仅限监管检查场景"),
    ]
    for r in sensitive:
        ws3.append(list(r))
    style_sheet(ws3, [22, 12, 20, 22, 12, 38])

    # Sheet 4 账号生命周期
    ws4 = wb.create_sheet("账号生命周期")
    ws4.append(["环节", "责任人", "时限", "系统动作", "依据文档"])
    lifecycle = [
        ("入职开通", "人力资源部", "入职前1个工作日", "自动创建基础账号并分配默认角色", "YQ-HR-001"),
        ("权限申请", "直属主管", "4小时响应", "按申请单授予最小权限", "YQ-ADM-005"),
        ("转岗调整", "直属主管", "2个工作日", "回收旧权限、授予新权限", "YQ-ADM-005"),
        ("离职禁用", "IT运维部", "离职当日18:00前", "禁用全部账号与VPN", "YQ-SEC-008"),
        ("离职删除", "IT运维部", "3个工作日内", "删除账号并归档审计日志", "YQ-SEC-008"),
        ("季度复核", "安全合规部", "每季度最后一周", "输出权限复核报告并回收冗余权限", "YQ-ADM-005"),
        ("超期未复核", "安全合规部", "复核结束后5个工作日", "冻结未复核账号", "YQ-SEC-008"),
    ]
    for r in lifecycle:
        ws4.append(list(r))
    style_sheet(ws4, [14, 16, 22, 40, 16])

    path = os.path.join(CORPUS, "权限矩阵.xlsx")
    wb.save(path)
    print("[xlsx] 权限矩阵.xlsx")


# --------------------------------------------------------------------------
# 5. 新人培训.pptx
# --------------------------------------------------------------------------
def build_onboarding_pptx():
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    NAVY = RGBColor(0x1F, 0x38, 0x64)
    GREY = RGBColor(0x59, 0x59, 0x59)
    ACCENT = RGBColor(0x2E, 0x75, 0xB6)

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    def add_slide(title, bullets, kicker=""):
        s = prs.slides.add_slide(blank)
        bar = s.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width, Inches(0.12))
        bar.fill.solid()
        bar.fill.fore_color.rgb = ACCENT
        bar.line.fill.background()
        if kicker:
            kb = s.shapes.add_textbox(Inches(0.8), Inches(0.55), Inches(11.7), Inches(0.4))
            kp = kb.text_frame.paragraphs[0]
            kr = kp.add_run()
            kr.text = kicker
            kr.font.size = Pt(12)
            kr.font.color.rgb = GREY
        tb = s.shapes.add_textbox(Inches(0.8), Inches(1.0), Inches(11.7), Inches(0.9))
        p = tb.text_frame.paragraphs[0]
        r = p.add_run()
        r.text = title
        r.font.size = Pt(30)
        r.font.bold = True
        r.font.color.rgb = NAVY
        body = s.shapes.add_textbox(Inches(0.9), Inches(2.1), Inches(11.5), Inches(4.9))
        tf = body.text_frame
        tf.word_wrap = True
        for i, item in enumerate(bullets):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.space_after = Pt(12)
            run = para.add_run()
            run.text = "• " + item
            run.font.size = Pt(19)
            run.font.color.rgb = RGBColor(0x26, 0x26, 0x26)
        return s

    # 1 封面
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    t = s.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(1.4))
    p = t.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "云栖科技 新人培训"
    r.font.size = Pt(44)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    st = s.shapes.add_textbox(Inches(1.0), Inches(3.8), Inches(11.3), Inches(1.2))
    p = st.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "文档编号：YQ-HR-006 ｜ 版本：v2.0 ｜ 生效日期：2026-06-01"
    r.font.size = Pt(16)
    r.font.color.rgb = RGBColor(0xD9, 0xE2, 0xF3)
    p2 = st.text_frame.add_paragraph()
    r2 = p2.add_run()
    r2.text = "人力资源部 ｜ 2026年6月"
    r2.font.size = Pt(16)
    r2.font.color.rgb = RGBColor(0xD9, 0xE2, 0xF3)

    add_slide("公司简介", [
        "2018年4月成立于杭州余杭区云栖小镇",
        "现有员工480人，深圳、成都设有办事处",
        "2025年营业收入3.6亿元，服务300+企业客户",
        "主营业务：云栖智能客服（YunqiBot）",
    ], "01 关于我们")
    add_slide("组织架构", [
        "技术中心120人，负责人为技术中心VP周敏",
        "产品部45人，总监何静",
        "客户成功部80人，总监孙倩；销售部90人，总监马涛",
        "职能团队145人，含人力资源总监王琳、财务总监赵磊",
    ], "02 组织与分工")
    add_slide("产品与客户", [
        "核心产品：云栖智能客服 v3.2",
        "标准版9.8万元每年，企业版SLA为99.95%",
        "覆盖银行、电商、运营商等行业，客户300+",
        "私有化部署最低3节点、16核64G",
    ], "03 我们在做什么")
    add_slide("文化与价值观", [
        "客户第一：先解决问题，再解释原因",
        "坦诚清晰：当面说、写清楚、留记录",
        "长期主义：不做一次性交付",
        "数据说话：结论必须能被指标验证",
        "每周三为无会日，每月最后一个周五全员会",
    ], "04 我们如何协作")
    add_slide("第一周任务清单", [
        "Day1：办理入职手续、领用设备（IT服务台）",
        "Day2：信息安全培训与考试，80分及格",
        "Day3：产品与业务培训",
        "Day4：部门跟岗，与导师见面",
        "Day5：提交《第一周学习总结》给导师",
    ], "05 第一周怎么过")
    add_slide("导师制与试用期", [
        "导师制为期3个月，导师每周1次1对1辅导",
        "试用期3个月，试用期工资为转正工资的80%",
        "转正材料须提前10个工作日提交",
        "转正答辩30分钟，由人力资源部总监王琳审批",
    ], "06 成长路径")
    add_slide("必读制度", [
        "员工手册 YQ-HR-001",
        "报销与差旅制度 YQ-FIN-002",
        "IT运维手册 YQ-IT-003",
        "信息安全规范 YQ-SEC-008",
        "权限矩阵 YQ-ADM-005",
    ], "07 制度地图")
    add_slide("安全与合规", [
        "密码90天更换一次，不得复用最近5次",
        "U盘默认禁用，需使用加密U盘",
        "安全事件30分钟内上报",
        "数据分L1至L4四级，L4禁止出办公网",
    ], "08 红线与底线")
    add_slide("联系人与资源", [
        "HRBP：王琳，分机8001",
        "IT服务台：分机8660，热线0571-8888 6600",
        "安全事件：security@yunqi.example.com",
        "内网知识库：wiki.yunqi.example.com",
    ], "09 有问题找谁")
    # 10 结尾：相关文档
    s = prs.slides.add_slide(blank)
    t = s.shapes.add_textbox(Inches(0.9), Inches(1.2), Inches(11.5), Inches(1.0))
    p = t.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "相关文档"
    r.font.size = Pt(32)
    r.font.bold = True
    r.font.color.rgb = NAVY
    body = s.shapes.add_textbox(Inches(0.9), Inches(2.4), Inches(11.5), Inches(4.2))
    tf = body.text_frame
    tf.word_wrap = True
    for i, line in enumerate([
        "YQ-HR-001 员工手册",
        "YQ-FIN-002 报销与差旅制度",
        "YQ-IT-003 IT运维手册",
        "YQ-SEC-008 信息安全规范",
        "YQ-ADM-005 权限矩阵",
        "YQ-HR-007 绩效考核办法",
    ]):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(10)
        run = para.add_run()
        run.text = "• " + line
        run.font.size = Pt(20)
        run.font.color.rgb = RGBColor(0x26, 0x26, 0x26)

    path = os.path.join(CORPUS, "新人培训.pptx")
    prs.save(path)
    print("[pptx] 新人培训.pptx")


# --------------------------------------------------------------------------
# 6. 绩效考核办法.docx
# --------------------------------------------------------------------------
def build_performance_docx():
    doc = new_doc()
    add_meta(doc, "文档编号：YQ-HR-007 ｜ 版本：v2.2 ｜ 生效日期：2026-01-15")
    add_title(doc, "云栖科技绩效考核办法")

    add_h(doc, "第一章 目的与适用范围")
    add_body(doc, "为客观评价员工贡献、支撑调薪与晋升决策，依据《员工手册》（YQ-HR-001）制定本办法。本办法适用于全体正式员工；试用期员工不参与半年考核，转正后满3个月起纳入考核范围。")

    add_h(doc, "第二章 考核周期")
    add_body(doc, "考核每半年一次，分别在6月和12月进行。年度综合得分由上半年得分占40%、下半年得分占60%加权计算。考核周期内累计请假超过30天的，按实际在岗月份折算。")

    add_h(doc, "第三章 等级与比例")
    add_body(doc, "考核结果分为S、A、B、C、D五个等级。S级不超过5%，A级占20%，B级占60%，C级占10%，D级占5%。对应绩效系数分别为1.5、1.2、1.0、0.8和0，系数直接与年终奖及调薪幅度挂钩。")

    add_h(doc, "第四章 考核流程")
    add_body(doc, "员工须在考核启动后3个工作日内完成自评；直属主管结合目标达成、协作表现与价值观评分；部门校准会由部门总监主持，HRBP列席并监督比例执行；校准结果由主管进行一对一沟通，员工须在系统中确认。")

    add_h(doc, "第五章 结果应用")
    add_body(doc, "S级与A级员工优先进入晋升与调薪名单。连续两次考核为C级的，启动绩效改进计划（PIP），周期为60天；PIP期满仍为C级或当年考核为D级的，公司可依法解除劳动合同。调薪窗口为每年4月，晋升评审为每年5月和11月各一次。")

    add_h(doc, "第六章 考勤与绩效衔接")
    add_body(doc, "月度迟到累计3次的，扣减当月绩效5分，与《员工手册》（YQ-HR-001）第三章一致。旷工1天扣10分，年度累计旷工3天以上直接评定为D级。")

    add_h(doc, "第七章 申诉")
    add_body(doc, "员工对考核结果有异议的，应在结果公布后5个工作日内向HRBP提交书面申诉。人力资源部应在收到申诉后10个工作日内组织复核并书面答复，复核结果为最终结果。")

    add_h(doc, "第八章 附则")
    add_body(doc, "本办法由人力资源部负责解释，自2026年1月15日起生效。奖金核算规则见《报销与差旅制度》（YQ-FIN-002）之外的财务专项通知；权限与考核系统操作见《权限矩阵》（YQ-ADM-005）。")

    add_related(doc, "YQ-HR-001 员工手册、YQ-ADM-005 权限矩阵、YQ-FIN-002 报销与差旅制度。")
    write_docx(doc, os.path.join(CORPUS, "绩效考核办法.docx"))


# --------------------------------------------------------------------------
# 7. 信息安全规范.md
# --------------------------------------------------------------------------
SEC_MD = """文档编号：YQ-SEC-008 ｜ 版本：v1.8 ｜ 生效日期：2026-04-01

# 云栖科技信息安全规范

> 维护部门：安全合规部（负责人郑凯）｜ 适用范围：全体员工、外包人员及实习人员

## 一、总则

信息安全是云栖科技的红线。任何员工不得以“业务紧急”为由绕过本规范；确需例外处理的，须由安全合规部书面批准并留存记录。本规范与《IT运维手册》（YQ-IT-003）、《权限矩阵》（YQ-ADM-005）配套执行。

## 二、数据分级

公司数据分为四级：L1公开、L2内部、L3敏感、L4核心。L4数据禁止离开办公网，任何导出、截屏、打印均须安全合规部郑凯审批；L3数据外发须加密并经部门总监审批。客户对话记录、合同、报价单默认按L3管理。

## 三、账号与密码

密码长度不少于12位，须包含大小写字母、数字和符号，每90天更换一次，不得复用最近5次密码。VPN、生产环境、计费系统强制开启MFA。账号仅限本人使用，禁止共享、转借或代登录；发现异常登录须立即修改密码并上报。

## 四、终端与移动存储

公司设备必须安装云盾EDR并保持在线，禁止安装盗版软件与来源不明的浏览器插件。个人设备未经审批不得接入办公网。U盘默认禁用；确需使用的，须申请加密U盘，审批人为IT运维部主管刘海。加密U盘遗失须在1小时内上报安全合规部。

## 五、外发与共享

对外传输文件统一使用加密网盘，分享链接有效期7天，到期自动失效。邮件外发L3及以上数据须填写《数据外发审批单》，并抄送部门总监。禁止使用个人网盘、个人邮箱传输公司数据。

## 六、安全事件

发现钓鱼邮件、数据泄露、账号异常、设备丢失等情况，须在30分钟内上报security@yunqi.example.com，并电话通知值班人员138-0000-6600。P1级安全事件1小时内成立应急小组，24小时内提交初步报告，72小时内完成复盘。

## 七、培训与考核

全员每年至少参加2次安全培训。年度安全考试80分及格，未通过者有一次补考机会；补考仍不通过的，暂停账号权限直至通过。新员工须在入职后5个工作日内完成首次考试。

## 八、违规处理

首次违规给予警告并参加补训；造成数据泄露或重大影响的，视情节给予记过、降级直至解除劳动合同，并依法追究法律责任。违规记录纳入年度绩效考核。

## 九、相关文档

- YQ-IT-003 IT运维手册
- YQ-ADM-005 权限矩阵
- YQ-HR-001 员工手册
- YQ-HR-007 绩效考核办法
"""


def build_security_md():
    os.makedirs(CORPUS, exist_ok=True)
    with open(os.path.join(CORPUS, "信息安全规范.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(SEC_MD)
    print("[md] 信息安全规范.md")


# --------------------------------------------------------------------------
# 8. 会议纪要-2026Q3.md
# --------------------------------------------------------------------------
MEETING_MD = """文档编号：YQ-OPS-010 ｜ 版本：v1.0 ｜ 生效日期：2026-09-30

# 云栖科技 2026年第三季度会议纪要

> 整理：总裁办 ｜ 分发范围：总监及以上、各HRBP ｜ 密级：L2内部

## 一、7月15日 产品评审会

主持：CEO陈立。参会：产品部何静、技术中心周敏、客户成功部孙倩。

决议：

1. v3.2上线首月活跃坐席增长18%，达到预期目标；
2. 下一版本v3.3定于2026-11-30发布，负责人何静；
3. 私有化部署最低配置统一为3节点、16核64G，与《产品FAQ》（YQ-PRD-004）保持一致；
4. 坐席辅助模块的客户满意度纳入Q4产品考核指标。

## 二、8月20日 信息安全专项会

主持：安全合规部郑凯。参会：IT运维部刘海、各部门HRBP。

决议：

1. 9月30日前完成全员安全考试，通过率须达到100%，未通过者暂停账号权限；
2. Q4开展一次钓鱼邮件演练，由刘海负责技术支撑；
3. 离职账号当日18:00前禁用，3个工作日内删除，责任人为刘海；
4. 加密U盘申请流程保持不变，审批人仍为刘海。

## 三、9月10日 Q4预算会

主持：财务部赵磊。参会：CEO陈立、销售部马涛、客户成功部孙倩。

决议：

1. Q4差旅预算整体压缩15%，通过合并行程、提前订票实现；
2. 一线城市住宿标准维持600元/晚不变，二线城市维持450元/晚不变；
3. 单笔超过10000元的报销仍须赵磊与陈立共同审批；
4. 市场预算向客户案例集推广倾斜，由孙倩牵头。

## 四、行动项跟踪

| 序号 | 事项 | 负责人 | 截止日期 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | v3.3发布 | 何静 | 2026-11-30 | 进行中 |
| 2 | 全员安全考试 | 郑凯 | 2026-09-30 | 已完成 |
| 3 | 钓鱼邮件演练 | 刘海 | 2026-12-15 | 未开始 |
| 4 | 差旅预算执行跟踪 | 赵磊 | 2026-12-31 | 进行中 |

## 五、相关文档

- YQ-PRD-004 产品FAQ
- YQ-FIN-002 报销与差旅制度
- YQ-SEC-008 信息安全规范
- YQ-IT-003 IT运维手册
"""


def build_meeting_md():
    os.makedirs(CORPUS, exist_ok=True)
    with open(os.path.join(CORPUS, "会议纪要-2026Q3.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(MEETING_MD)
    print("[md] 会议纪要-2026Q3.md")


# --------------------------------------------------------------------------
# 9. 报销与差旅制度（docx -> pdf）
# --------------------------------------------------------------------------
def build_reimbursement_source():
    doc = new_doc()
    add_meta(doc, "文档编号：YQ-FIN-002 ｜ 版本：v2.4 ｜ 生效日期：2026-03-01")
    add_title(doc, "云栖科技报销与差旅制度")

    add_h(doc, "第一章 总则")
    add_body(doc, "本制度依据《员工手册》（YQ-HR-001）制定，适用于云栖科技（杭州）有限公司全体员工、试用期员工与实习生。")
    add_body(doc, "差旅与费用报销由财务部负责解释，财务部总监为赵磊。")

    add_h(doc, "第二章 差旅申请")
    add_body(doc, "出差前2个工作日，员工须在OA提交《出差申请单》。")
    add_body(doc, "单次行程超过5天的，须由业务或技术VP审批。")
    add_body(doc, "国际差旅须提前10个工作日提交申请。")
    add_body(doc, "未经审批擅自出差的，费用不予报销。")

    add_h(doc, "第三章 差旅标准")
    add_body(doc, "住宿标准按城市等级执行。")
    add_body(doc, "一线城市住宿标准为600元每晚。")
    add_body(doc, "一线城市包括北京、上海、广州、深圳。")
    add_body(doc, "二线城市住宿标准为450元每晚。")
    add_body(doc, "其他城市住宿标准为350元每晚。")
    add_body(doc, "超标部分由个人承担。")
    add_body(doc, "餐补为100元每天，市内交通补贴为80元每天。")
    add_body(doc, "长途交通统一乘坐高铁二等座或飞机经济舱。")
    add_body(doc, "单程飞行时间4小时以上，可申请商务舱，须经VP审批。")

    add_h(doc, "第四章 审批层级")
    add_body(doc, "单笔报销不超过2000元，由直属主管审批。")
    add_body(doc, "单笔2000元至10000元，由部门总监审批。")
    add_body(doc, "单笔超过10000元，须财务总监赵磊与CEO陈立共同审批。")
    add_body(doc, "审批权限与《权限矩阵》（YQ-ADM-005）保持一致。")

    add_h(doc, "第五章 发票要求")
    add_body(doc, "发票抬头为“云栖科技（杭州）有限公司”。")
    add_body(doc, "公司税号为91330100MA2B7QK83X。")
    add_body(doc, "增值税专用发票须在开具后5个工作日内交财务部。")
    add_body(doc, "电子发票须上传OA费控模块。")
    add_body(doc, "发票信息不全的，予以退回重开。")

    add_h(doc, "第六章 报销时限")
    add_body(doc, "费用发生后30天内提交报销。")
    add_body(doc, "每月25日为当月报销截止日。")
    add_body(doc, "跨年费用须在12月20日前提交。")
    add_body(doc, "逾期超过60天的不予受理，特殊情况须财务总监特批。")

    add_h(doc, "第七章 借款与冲账")
    add_body(doc, "单笔借款不超过5000元。")
    add_body(doc, "借款后15天内须完成报销冲账。")
    add_body(doc, "未按时冲账的，从当月工资中扣回。")

    add_h(doc, "第八章 其他费用")
    add_body(doc, "培训与会议费用须附会议通知与签到表。")
    add_body(doc, "私车公用按1.2元每公里补贴，须提前备案。")
    add_body(doc, "招待费须注明客户名称与人数，单次不超过3000元。")
    add_body(doc, "办公用品单笔超过500元须附明细清单。")

    add_h(doc, "第九章 违规处理")
    add_body(doc, "虚报费用的，追回款项并在全公司通报。")
    add_body(doc, "情节严重的，依法解除劳动合同。")

    add_related(doc, "《员工手册》（YQ-HR-001）、《权限矩阵》（YQ-ADM-005）、《会议纪要-2026Q3》（YQ-OPS-010）。")
    path = os.path.join(BUILD, "报销与差旅制度.docx")
    write_docx(doc, path)
    return path, "报销与差旅制度.pdf"


# --------------------------------------------------------------------------
# 10. 客户案例集（docx -> pdf）
# --------------------------------------------------------------------------
def build_case_source():
    doc = new_doc()
    add_meta(doc, "文档编号：YQ-MKT-009 ｜ 版本：v1.2 ｜ 生效日期：2026-07-01")
    add_title(doc, "云栖科技客户案例集")

    add_h(doc, "一、客户概览")
    add_body(doc, "截至2026年6月，云栖智能客服已服务企业客户300余家。")
    add_body(doc, "客户覆盖银行、电商、运营商、制造、政务五大行业。")
    add_body(doc, "金融行业客户占比35%，电商行业占比28%。")
    add_body(doc, "客户续约率为92%，2025年客户满意度调研得分为4.7分。")

    add_h(doc, "二、案例一：杭州某城市商业银行")
    add_body(doc, "该客户拥有500个坐席，采用私有化部署。")
    add_body(doc, "原有客服团队200人，上线后60人转岗至营销岗位。")
    add_body(doc, "项目8周完成上线，人工咨询量下降60%。")
    add_body(doc, "平均首次响应时间从45秒降至12秒。")
    add_body(doc, "机器人解决率为76%，客户满意度提升12个百分点。")

    add_h(doc, "三、案例二：某头部电商平台")
    add_body(doc, "该客户日均会话12万次，大促并发1000。")
    add_body(doc, "大促峰值每分钟承接4200次咨询。")
    add_body(doc, "机器人解决率达到82%，P95响应时间为650毫秒。")
    add_body(doc, "大促期间零故障，SLA达成99.96%。")
    add_body(doc, "客户成功经理每周驻场1天，保障活动节奏。")

    add_h(doc, "四、案例三：某省级运营商")
    add_body(doc, "该客户知识库条目1.2万条，覆盖18个地市。")
    add_body(doc, "项目3个月完成上线，工单流转时长缩短40%。")
    add_body(doc, "坐席辅助模块使一次性解决率提升9个百分点。")
    add_body(doc, "系统与客户原有工单平台通过OpenAPI对接。")

    add_h(doc, "五、实施方法")
    add_body(doc, "标准实施周期为6周，私有化项目为10周。")
    add_body(doc, "流程为需求确认、环境准备、知识库导入、联调测试、试运行2周、正式上线。")
    add_body(doc, "实施团队由项目经理、实施顾问、算法工程师各1名组成。")
    add_body(doc, "每个客户配备1名客户成功经理，提供1对1服务。")
    add_body(doc, "上线后30天进行首次回访，之后每季度复盘一次。")

    add_h(doc, "六、客户评价")
    add_body(doc, "“上线两个月，客服人力成本下降了三分之一。”——某电商客户客服总监")
    add_body(doc, "“私有化部署满足了我们最严格的合规要求。”——某银行客户科技部负责人")
    add_body(doc, "“知识库维护比预期简单，业务部门可以自己更新。”——某运营商客户运营经理")

    add_related(doc, "《产品FAQ》（YQ-PRD-004）、《IT运维手册》（YQ-IT-003）、《信息安全规范》（YQ-SEC-008）。")
    path = os.path.join(BUILD, "客户案例集.docx")
    write_docx(doc, path)
    return path, "客户案例集.pdf"


# --------------------------------------------------------------------------
# PDF 转换与主流程
# --------------------------------------------------------------------------
def convert_pdf(node, cli, src_docx, out_pdf):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    if os.path.exists(out_pdf):
        os.remove(out_pdf)
    cmd = [node, cli, "convert", "--input", src_docx, "--output", out_pdf]
    print("[pdf] 转换", os.path.basename(src_docx))
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise SystemExit(f"PDF 转换失败：{src_docx}")
    with open(out_pdf, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise SystemExit(f"输出不是有效 PDF：{out_pdf}")
    print(f"[pdf] {os.path.basename(out_pdf)} ({os.path.getsize(out_pdf)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default=r"D:\Program Files\DeepSeekHarness\DSH\resources\runtime\primary-runtime\dependencies\node\bin\node.exe")
    ap.add_argument("--pdf-cli", default=r"D:\Program Files\DeepSeekHarness\DSH\resources\app.asar.unpacked\dsh\node_modules\@deepseek-ai\libreoffice-kit\lib\cli.js")
    args = ap.parse_args()

    os.makedirs(CORPUS, exist_ok=True)
    if os.path.isdir(BUILD):
        shutil.rmtree(BUILD)
    os.makedirs(BUILD, exist_ok=True)

    build_employee_handbook()
    build_it_manual()
    build_product_faq()
    build_permission_matrix()
    build_onboarding_pptx()
    build_performance_docx()
    build_security_md()
    build_meeting_md()

    for builder in (build_reimbursement_source, build_case_source):
        src, pdf_name = builder()
        convert_pdf(args.node, args.pdf_cli, src, os.path.join(CORPUS, pdf_name))

    shutil.rmtree(BUILD)

    files = sorted(os.listdir(CORPUS))
    print("\n语料目录：", CORPUS)
    for name in files:
        print("  -", name, os.path.getsize(os.path.join(CORPUS, name)), "bytes")
    print(f"共 {len(files)} 个文件")


if __name__ == "__main__":
    main()