from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .optimizer import (
    Employee,
    OptimizationResult,
    ScheduledShift,
    ShiftSlot,
    SLOT_TIME_RANGES,
    Weekday,
)


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Violation(BaseModel):
    severity: Severity
    category: str
    message: str
    employee_id: Optional[str] = None
    employee_name: Optional[str] = None
    shift_date: Optional[date] = None
    shift_index: Optional[int] = None
    suggestion: str = ""


class ValidationReport(BaseModel):
    is_compliant: bool
    total_shifts: int
    total_violations: int
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    violations: List[Violation] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.now)

    def to_markdown(self) -> str:
        lines = ["# 排班合规检查报告\n"]
        lines.append(f"- 生成时间: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- 检查班次总数: {self.total_shifts}")
        lines.append(f"- 合规状态: {'✅ 通过' if self.is_compliant else '❌ 存在违规'}")
        lines.append(f"- 违规总数: {self.total_violations} (严重:{self.critical_count} 警告:{self.warning_count} 建议:{self.info_count})")
        lines.append("")

        if self.violations:
            lines.append("## 违规明细\n")
            current_severity = None
            for v in self.violations:
                if v.severity != current_severity:
                    current_severity = v.severity
                    icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(v.severity.value, "")
                    label = {"critical": "严重", "warning": "警告", "info": "建议"}.get(v.severity.value, "")
                    lines.append(f"\n### {icon} {label}\n")
                line_parts = [f"- **[{v.category}]** {v.message}"]
                if v.employee_name:
                    line_parts.append(f"| 员工: {v.employee_name}({v.employee_id})")
                if v.shift_date:
                    line_parts.append(f"| 日期: {v.shift_date}")
                if v.shift_index is not None:
                    line_parts.append(f"| 行号: #{v.shift_index + 1}")
                lines.append(" ".join(line_parts))
                if v.suggestion:
                    lines.append(f"  - 建议修正: {v.suggestion}")
        else:
            lines.append("## 无违规项，排班完全合规！\n")

        return "\n".join(lines)


class ShiftValidator:
    LEGAL_MAX_WEEKLY_HOURS = 60
    WARNING_WEEKLY_HOURS = 50
    LEGAL_MAX_DAILY_HOURS = 11
    WARNING_DAILY_HOURS = 10
    LEGAL_MAX_CONSECUTIVE_DAYS = 6
    WARN_CONSECUTIVE_WEEKENDS = 3
    NIGHT_SHIFT_START = time(22, 0)
    NIGHT_SHIFT_END = time(6, 0)
    MINOR_MAX_DAILY_HOURS = 8
    MINOR_ALLOWED_END_HOUR = 22

    def __init__(self, employees: Optional[List[Employee]] = None):
        self.employees = employees or []
        self._emp_map: Dict[str, Employee] = {e.id: e for e in self.employees}

    def set_employees(self, employees: List[Employee]):
        self.employees = employees
        self._emp_map = {e.id: e for e in employees}

    def _get_employee(self, emp_id: str) -> Optional[Employee]:
        return self._emp_map.get(emp_id)

    def _get_weekday(self, d: date) -> Weekday:
        return Weekday(d.weekday())

    def _is_weekend(self, d: date) -> bool:
        return self._get_weekday(d) in (Weekday.SATURDAY, Weekday.SUNDAY)

    def _is_night_time(self, t: time) -> bool:
        return t >= self.NIGHT_SHIFT_START or t <= self.NIGHT_SHIFT_END

    def _shift_hours(self, start: time, end: time) -> float:
        today = date.today()
        s_dt = datetime.combine(today, start)
        e_dt = datetime.combine(today, end)
        if e_dt <= s_dt:
            e_dt += timedelta(days=1)
        return (e_dt - s_dt).total_seconds() / 3600.0

    def validate(
        self,
        shifts: List[ScheduledShift],
        week_start: Optional[date] = None,
    ) -> ValidationReport:
        violations: List[Violation] = []
        shifts_sorted = sorted(shifts, key=lambda s: (s.date, s.employee_id))

        per_emp_hours_weekly: Dict[str, float] = {}
        per_emp_hours_daily: Dict[Tuple[str, date], float] = {}
        per_emp_work_dates: Dict[str, set] = {}
        per_emp_weekend_count: Dict[str, int] = {}
        per_emp_consecutive_streak: Dict[str, int] = {}
        per_emp_last_work_date: Dict[str, date] = {}

        for idx, shift in enumerate(shifts_sorted):
            emp_id = shift.employee_id
            emp = self._get_employee(emp_id)
            shift_hours = self._shift_hours(shift.start_time, shift.end_time)

            per_emp_hours_weekly[emp_id] = per_emp_hours_weekly.get(emp_id, 0.0) + shift_hours

            day_key = (emp_id, shift.date)
            per_emp_hours_daily[day_key] = per_emp_hours_daily.get(day_key, 0.0) + shift_hours

            if emp_id not in per_emp_work_dates:
                per_emp_work_dates[emp_id] = set()
            per_emp_work_dates[emp_id].add(shift.date)

            if self._is_weekend(shift.date):
                per_emp_weekend_count[emp_id] = per_emp_weekend_count.get(emp_id, 0) + 1

            if shift_hours > 8:
                violations.append(Violation(
                    severity=Severity.WARNING,
                    category="单班次时长",
                    message=f"单次班次时长{shift_hours:.1f}小时超过标准8小时",
                    employee_id=emp_id,
                    employee_name=shift.employee_name,
                    shift_date=shift.date,
                    shift_index=idx,
                    suggestion="拆分为两个班次或安排休息时段",
                ))

            daily_total = per_emp_hours_daily[day_key]
            if daily_total > self.LEGAL_MAX_DAILY_HOURS:
                violations.append(Violation(
                    severity=Severity.CRITICAL,
                    category="日工时劳动法",
                    message=f"单日工时{daily_total:.1f}小时超过法定上限{self.LEGAL_MAX_DAILY_HOURS}小时",
                    employee_id=emp_id,
                    employee_name=shift.employee_name,
                    shift_date=shift.date,
                    shift_index=idx,
                    suggestion=f"减少当日班次，确保不超过{self.LEGAL_MAX_DAILY_HOURS}小时",
                ))
            elif daily_total > self.WARNING_DAILY_HOURS:
                violations.append(Violation(
                    severity=Severity.WARNING,
                    category="日工时预警",
                    message=f"单日工时{daily_total:.1f}小时接近上限{self.WARNING_DAILY_HOURS}小时",
                    employee_id=emp_id,
                    employee_name=shift.employee_name,
                    shift_date=shift.date,
                    shift_index=idx,
                    suggestion="考虑安排次日休息或减少次日排班",
                ))

            is_night = self._is_night_time(shift.start_time) or self._is_night_time(shift.end_time)
            if emp and emp.is_minor and is_night:
                violations.append(Violation(
                    severity=Severity.CRITICAL,
                    category="未成年夜班",
                    message=f"未成年员工安排夜班（{shift.start_time}-{shift.end_time}）",
                    employee_id=emp_id,
                    employee_name=shift.employee_name,
                    shift_date=shift.date,
                    shift_index=idx,
                    suggestion="将未成年员工调整到白天班次（6:00-22:00）",
                ))

            if emp and emp.is_minor:
                end_hour = shift.end_time.hour + shift.end_time.minute / 60
                if end_hour > self.MINOR_ALLOWED_END_HOUR:
                    violations.append(Violation(
                        severity=Severity.CRITICAL,
                        category="未成年下班时间",
                        message=f"未成年员工下班时间{shift.end_time}超过22:00",
                        employee_id=emp_id,
                        employee_name=shift.employee_name,
                        shift_date=shift.date,
                        shift_index=idx,
                        suggestion="调整班次结束时间至22:00前",
                    ))

            if emp and not emp.is_available(self._get_weekday(shift.date), shift.slot):
                violations.append(Violation(
                    severity=Severity.CRITICAL,
                    category="员工不可用时间",
                    message=f"员工在{shift.slot.value}时段（{shift.start_time}-{shift.end_time}）标记为不可用",
                    employee_id=emp_id,
                    employee_name=shift.employee_name,
                    shift_date=shift.date,
                    shift_index=idx,
                    suggestion="更换其他可用员工或调整时段",
                ))

        for emp_id, weekly_hours in per_emp_hours_weekly.items():
            emp = self._get_employee(emp_id)
            emp_name = emp.name if emp else emp_id
            max_allowed = emp.max_weekly_hours if emp else self.LEGAL_MAX_WEEKLY_HOURS

            if weekly_hours > self.LEGAL_MAX_WEEKLY_HOURS and weekly_hours > max_allowed:
                violations.append(Violation(
                    severity=Severity.CRITICAL,
                    category="周工时劳动法",
                    message=f"周总工时{weekly_hours:.1f}小时超过法定上限{self.LEGAL_MAX_WEEKLY_HOURS}小时和个人上限{max_allowed}小时",
                    employee_id=emp_id,
                    employee_name=emp_name,
                    suggestion=f"立即减少排班，确保周工时不超过{min(self.LEGAL_MAX_WEEKLY_HOURS, max_allowed)}小时",
                ))
            elif weekly_hours > max_allowed:
                violations.append(Violation(
                    severity=Severity.CRITICAL,
                    category="周工时个人上限",
                    message=f"周总工时{weekly_hours:.1f}小时超过个人上限{max_allowed}小时",
                    employee_id=emp_id,
                    employee_name=emp_name,
                    suggestion=f"减少排班至个人上限{max_allowed}小时以内",
                ))
            elif weekly_hours > self.LEGAL_MAX_WEEKLY_HOURS:
                violations.append(Violation(
                    severity=Severity.CRITICAL,
                    category="周工时劳动法",
                    message=f"周总工时{weekly_hours:.1f}小时超过法定上限{self.LEGAL_MAX_WEEKLY_HOURS}小时",
                    employee_id=emp_id,
                    employee_name=emp_name,
                    suggestion=f"减少排班至法定{self.LEGAL_MAX_WEEKLY_HOURS}小时以内",
                ))
            elif weekly_hours > self.WARNING_WEEKLY_HOURS:
                violations.append(Violation(
                    severity=Severity.WARNING,
                    category="周工时预警",
                    message=f"周总工时{weekly_hours:.1f}小时接近上限{self.WARNING_WEEKLY_HOURS}小时",
                    employee_id=emp_id,
                    employee_name=emp_name,
                    suggestion=f"下周考虑减少排班，本周建议安排补休",
                ))

        for emp_id, dates in per_emp_work_dates.items():
            if len(dates) == 0:
                continue
            emp = self._get_employee(emp_id)
            emp_name = emp.name if emp else emp_id
            sorted_dates = sorted(dates)
            max_streak = 1
            current_streak = 1
            for i in range(1, len(sorted_dates)):
                if (sorted_dates[i] - sorted_dates[i-1]).days == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1

            if max_streak > self.LEGAL_MAX_CONSECUTIVE_DAYS:
                violations.append(Violation(
                    severity=Severity.CRITICAL,
                    category="连续工作天数",
                    message=f"连续工作{max_streak}天超过法定上限{self.LEGAL_MAX_CONSECUTIVE_DAYS}天",
                    employee_id=emp_id,
                    employee_name=emp_name,
                    suggestion=f"插入休息日，确保任意{self.LEGAL_MAX_CONSECUTIVE_DAYS + 1}天内至少休息1天",
                ))
            elif max_streak == self.LEGAL_MAX_CONSECUTIVE_DAYS:
                violations.append(Violation(
                    severity=Severity.WARNING,
                    category="连续工作天数预警",
                    message=f"连续工作已达{max_streak}天上限",
                    employee_id=emp_id,
                    employee_name=emp_name,
                    suggestion="后续务必安排休息，避免超时",
                ))

        if week_start:
            weekend_counts = list(per_emp_weekend_count.items())
            if weekend_counts:
                counts = [c for _, c in weekend_counts]
                avg = sum(counts) / len(counts) if counts else 0
                for emp_id, cnt in weekend_counts:
                    if cnt >= self.WARN_CONSECUTIVE_WEEKENDS and cnt > avg + 1:
                        emp = self._get_employee(emp_id)
                        emp_name = emp.name if emp else emp_id
                        violations.append(Violation(
                            severity=Severity.INFO,
                            category="周末班公平性",
                            message=f"该员工本周承担{cnt}个周末班次，明显高于平均（{avg:.1f}）",
                            employee_id=emp_id,
                            employee_name=emp_name,
                            suggestion="下周优先安排其他员工轮替周末班",
                        ))

        severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        violations.sort(key=lambda v: (severity_order.get(v.severity, 99), v.employee_id or "", v.shift_date or date.min))

        critical = sum(1 for v in violations if v.severity == Severity.CRITICAL)
        warning = sum(1 for v in violations if v.severity == Severity.WARNING)
        info = sum(1 for v in violations if v.severity == Severity.INFO)

        category_counts: Dict[str, int] = {}
        for v in violations:
            category_counts[v.category] = category_counts.get(v.category, 0) + 1

        return ValidationReport(
            is_compliant=critical == 0,
            total_shifts=len(shifts),
            total_violations=len(violations),
            critical_count=critical,
            warning_count=warning,
            info_count=info,
            violations=violations,
            summary=category_counts,
        )

    def export_pdf(self, report: ValidationReport, output_path: str) -> bool:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError:
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=18,
            spaceAfter=12,
        )
        elements.append(Paragraph("排班合规检查报告", title_style))
        elements.append(Spacer(1, 6))

        info_style = styles["Normal"]
        elements.append(Paragraph(f"生成时间: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}", info_style))
        elements.append(Paragraph(f"检查班次总数: {report.total_shifts}", info_style))
        status_text = "✅ 通过" if report.is_compliant else "❌ 存在违规"
        elements.append(Paragraph(f"合规状态: {status_text}", info_style))
        elements.append(Paragraph(
            f"违规总数: {report.total_violations} (严重:{report.critical_count} 警告:{report.warning_count} 建议:{report.info_count})",
            info_style,
        ))
        elements.append(Spacer(1, 12))

        if report.violations:
            header_style = ParagraphStyle(
                "Header",
                parent=styles["Heading2"],
                fontSize=14,
                spaceAfter=8,
            )
            elements.append(Paragraph("违规明细", header_style))

            table_data = [["级别", "分类", "员工", "日期", "问题描述", "修正建议"]]
            for v in report.violations:
                label = {"critical": "严重", "warning": "警告", "info": "建议"}.get(v.severity.value, v.severity.value)
                table_data.append([
                    label,
                    v.category,
                    f"{v.employee_name or '-'}\n({v.employee_id or '-'})",
                    str(v.shift_date) if v.shift_date else "-",
                    v.message,
                    v.suggestion,
                ])

            col_widths = [20*mm, 30*mm, 30*mm, 22*mm, 55*mm, 45*mm]
            table = Table(table_data, colWidths=col_widths, repeatRows=1)
            severity_colors = {
                "严重": colors.hex2color("#ffcccc"),
                "警告": colors.hex2color("#fff3cc"),
                "建议": colors.hex2color("#cce5ff"),
            }

            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.gray),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
            ]
            for i, v in enumerate(report.violations, 1):
                label = {"critical": "严重", "warning": "警告", "info": "建议"}.get(v.severity.value, "")
                if label in severity_colors:
                    style_cmds.append(("BACKGROUND", (0, i), (0, i), severity_colors[label]))
            table.setStyle(TableStyle(style_cmds))
            elements.append(table)

        doc.build(elements)
        return True
