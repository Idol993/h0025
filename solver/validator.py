import io
import os
from dataclasses import dataclass
from datetime import date as Date, datetime as DateTime, time as Time, timedelta
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
    shift_date: Optional[Date] = None
    shift_index: Optional[int] = None
    suggestion: str = ""
    details: Dict = Field(default_factory=dict)


class ValidationReport(BaseModel):
    is_compliant: bool
    total_shifts: int
    total_violations: int
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    violations: List[Violation] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)
    generated_at: DateTime = Field(default_factory=lambda: DateTime.now())

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
                if "weekends" in v.details and v.details["weekends"]:
                    wknds = ", ".join(f"{d[0].month}/{d[0].day}-{d[1].month}/{d[1].day}" for d in v.details["weekends"])
                    line_parts.append(f"| 涉及周末: [{wknds}]")
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
    WARN_CONSECUTIVE_DAYS = 6
    WARN_CONSECUTIVE_WEEKENDS = 4
    NIGHT_SHIFT_START = Time(22, 0)
    NIGHT_SHIFT_END = Time(6, 0)
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

    def _get_weekday(self, d: Date) -> Weekday:
        return Weekday(d.weekday())

    def _is_weekend(self, d: Date) -> bool:
        return self._get_weekday(d) in (Weekday.SATURDAY, Weekday.SUNDAY)

    def _is_night_time(self, t: Time) -> bool:
        return t >= self.NIGHT_SHIFT_START or t <= self.NIGHT_SHIFT_END

    def _shift_hours(self, start: Time, end: Time) -> float:
        today = Date.today()
        s_dt = DateTime.combine(today, start)
        e_dt = DateTime.combine(today, end)
        if e_dt <= s_dt:
            e_dt += timedelta(days=1)
        return (e_dt - s_dt).total_seconds() / 3600.0

    def _get_weekend_range(self, any_date_in_week: Date) -> Tuple[Date, Date]:
        monday = any_date_in_week - timedelta(days=any_date_in_week.weekday())
        saturday = monday + timedelta(days=5)
        sunday = monday + timedelta(days=6)
        return (saturday, sunday)

    def _find_consecutive_weekends(
        self,
        work_dates: List[Date],
        week_start: Optional[Date],
        extra_history_weekends: Optional[List[Date]] = None,
    ) -> List[Tuple[Date, Date]]:
        weekend_week_keys: set = set()
        all_weekends_worked: Dict[str, Tuple[Date, Date]] = {}

        dates_to_check = list(work_dates)
        if extra_history_weekends:
            dates_to_check.extend(extra_history_weekends)

        for d in dates_to_check:
            if self._is_weekend(d):
                sat, sun = self._get_weekend_range(d)
                key = f"{sat.isoformat()}"
                weekend_week_keys.add(key)
                all_weekends_worked[key] = (sat, sun)

        if not weekend_week_keys:
            return []

        sorted_keys = sorted(weekend_week_keys)
        runs: List[List[str]] = []
        current_run = [sorted_keys[0]]
        for i in range(1, len(sorted_keys)):
            prev_key = sorted_keys[i - 1]
            curr_key = sorted_keys[i]
            prev_sat = Date.fromisoformat(prev_key)
            curr_sat = Date.fromisoformat(curr_key)
            if (curr_sat - prev_sat).days == 7:
                current_run.append(curr_key)
            else:
                if len(current_run) >= self.WARN_CONSECUTIVE_WEEKENDS:
                    runs.append(current_run)
                current_run = [curr_key]
        if len(current_run) >= self.WARN_CONSECUTIVE_WEEKENDS:
            runs.append(current_run)

        result = []
        for run in runs:
            for key in run:
                result.append(all_weekends_worked[key])
        return result

    def validate(
        self,
        shifts: List[ScheduledShift],
        week_start: Optional[Date] = None,
        history_weekend_dates: Optional[Dict[str, List[Date]]] = None,
    ) -> ValidationReport:
        history_weekend_dates = history_weekend_dates or {}
        violations: List[Violation] = []
        shifts_sorted = sorted(shifts, key=lambda s: (s.date, s.employee_id))

        per_emp_hours_weekly: Dict[str, float] = {}
        per_emp_hours_daily: Dict[Tuple[str, Date], float] = {}
        per_emp_work_dates: Dict[str, set] = {}
        per_emp_weekend_count: Dict[str, int] = {}

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
            streak_start = sorted_dates[0]
            current_start = sorted_dates[0]
            worst_start = sorted_dates[0]

            for i in range(1, len(sorted_dates)):
                if (sorted_dates[i] - sorted_dates[i - 1]).days == 1:
                    current_streak += 1
                    if current_streak > max_streak:
                        max_streak = current_streak
                        worst_start = current_start
                else:
                    current_streak = 1
                    current_start = sorted_dates[i]

            worst_end = worst_start + timedelta(days=max_streak - 1)

            if max_streak > self.LEGAL_MAX_CONSECUTIVE_DAYS:
                violations.append(Violation(
                    severity=Severity.CRITICAL,
                    category="连续工作天数",
                    message=f"连续工作{max_streak}天（{worst_start}至{worst_end}），超过法定上限{self.LEGAL_MAX_CONSECUTIVE_DAYS}天",
                    employee_id=emp_id,
                    employee_name=emp_name,
                    shift_date=worst_start,
                    suggestion=f"插入休息日，确保任意{self.LEGAL_MAX_CONSECUTIVE_DAYS + 1}天内至少休息1天",
                    details={"streak": max_streak, "start": str(worst_start), "end": str(worst_end)},
                ))
            elif max_streak == self.WARN_CONSECUTIVE_DAYS:
                violations.append(Violation(
                    severity=Severity.WARNING,
                    category="连续工作天数预警",
                    message=f"连续工作已达{max_streak}天（{worst_start}至{worst_end}），触碰法定上限边界",
                    employee_id=emp_id,
                    employee_name=emp_name,
                    shift_date=worst_start,
                    suggestion="后续务必安排休息，避免超时；建议当日减少工作量",
                    details={"streak": max_streak, "start": str(worst_start), "end": str(worst_end)},
                ))

        for emp_id, dates in per_emp_work_dates.items():
            emp = self._get_employee(emp_id)
            emp_name = emp.name if emp else emp_id
            hist_dates = history_weekend_dates.get(emp_id, [])
            wknds_with_work = self._find_consecutive_weekends(list(dates), week_start, hist_dates)
            if wknds_with_work:
                n_weeks = len(wknds_with_work)
                first = wknds_with_work[0][0]
                last = wknds_with_work[-1][1]
                desc = "、".join(
                    f"{r[0].strftime('%m/%d')}~{r[1].strftime('%m/%d')}" for r in wknds_with_work
                )
                violations.append(Violation(
                    severity=Severity.INFO,
                    category="连续周末排班",
                    message=f"已连续{n_weeks}个周末被安排上班（{desc}），达到提醒阈值",
                    employee_id=emp_id,
                    employee_name=emp_name,
                    shift_date=last,
                    suggestion=f"建议在后续周末安排休息，已连续{n_weeks}次，下一个周末优先考虑轮空",
                    details={"weekends": [(str(a), str(b)) for a, b in wknds_with_work], "streak_weeks": n_weeks},
                ))

        if week_start:
            weekend_counts = list(per_emp_weekend_count.items())
            if weekend_counts:
                counts = [c for _, c in weekend_counts]
                avg = sum(counts) / len(counts) if counts else 0
                for emp_id, cnt in weekend_counts:
                    if cnt >= 3 and cnt > avg + 0.5:
                        emp = self._get_employee(emp_id)
                        emp_name = emp.name if emp else emp_id
                        violations.append(Violation(
                            severity=Severity.INFO,
                            category="周末班公平性",
                            message=f"该员工本周承担{cnt}个周末班次，高于平均（{avg:.1f}）",
                            employee_id=emp_id,
                            employee_name=emp_name,
                            suggestion="下周优先安排其他员工轮替周末班，注意公平分配",
                        ))

        severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        violations.sort(key=lambda v: (
            severity_order.get(v.severity, 99),
            v.employee_id or "",
            v.shift_date or Date.min,
        ))

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
                extra_msg = v.message
                if "weekends" in v.details and v.details["weekends"]:
                    wknds = ", ".join(
                        f"{a[5:]}~{b[5:]}" for a, b in v.details["weekends"]
                    )
                    extra_msg = f"{v.message} [涉及周末: {wknds}]"
                table_data.append([
                    label,
                    v.category,
                    f"{v.employee_name or '-'}\n({v.employee_id or '-'})",
                    str(v.shift_date) if v.shift_date else "-",
                    extra_msg,
                    v.suggestion,
                ])

            col_widths = [20 * mm, 30 * mm, 30 * mm, 22 * mm, 55 * mm, 45 * mm]
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
