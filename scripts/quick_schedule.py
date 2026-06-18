#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm
from sqlalchemy import func
from sqlalchemy.orm import Session

from server.main import (
    DBAvailability,
    DBEmployee,
    DBShift,
    DBStore,
    SessionLocal,
)
from solver.optimizer import (
    Availability as OptimizerAvailability,
    Employee as OptimizerEmployee,
    OptimizationResult,
    ShiftDemand as OptimizerShiftDemand,
    ShiftOptimizer,
    ShiftSlot,
    Weekday,
)
from solver.validator import Severity, ShiftValidator


SLOT_CN = {
    ShiftSlot.MORNING: "早班",
    ShiftSlot.MIDDAY: "中班",
    ShiftSlot.EVENING: "晚班",
    ShiftSlot.NIGHT: "夜班",
}

SLOT_EMOJI = {
    ShiftSlot.MORNING: "🌅",
    ShiftSlot.MIDDAY: "☀️",
    ShiftSlot.EVENING: "🌆",
    ShiftSlot.NIGHT: "🌙",
}

WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

console = Console()


def parse_week_string(week_str: str) -> date:
    m = re.match(r"(\d{4})-W(\d{1,2})", week_str)
    if not m:
        raise ValueError(f"周格式错误，应为YYYY-WNN，例如 2026-W12")
    year = int(m.group(1))
    week = int(m.group(2))
    if week < 1 or week > 53:
        raise ValueError("周数应在1-53之间")
    jan4 = date(year, 1, 4)
    monday = jan4 - timedelta(days=jan4.weekday()) + timedelta(weeks=week - 1)
    return monday


def get_or_create_store(db: Session, store_name: str) -> DBStore:
    store = db.query(DBStore).filter(DBStore.name == store_name).first()
    if store:
        return store
    store_id = f"store_{store_name}"
    store = DBStore(id=store_id, name=store_name, address="")
    db.add(store)
    db.commit()
    db.refresh(store)
    return store


def _to_opt_emp(db_emp: DBEmployee) -> OptimizerEmployee:
    avs = [
        OptimizerAvailability(
            weekday=Weekday(av.weekday),
            slot=ShiftSlot(av.slot),
            available=av.available,
            preferred=av.preferred,
        )
        for av in db_emp.availabilities
    ]
    return OptimizerEmployee(
        id=db_emp.id,
        name=db_emp.name,
        position=db_emp.position,
        hourly_rate=db_emp.hourly_rate,
        max_weekly_hours=db_emp.max_weekly_hours,
        skills=list(db_emp.skills or []),
        is_minor=db_emp.is_minor,
        availabilities=avs,
        feedback_weight=db_emp.feedback_weight,
    )


def ensure_sample_employees(db: Session, store: DBStore):
    existing = db.query(DBEmployee).filter(DBEmployee.store_id == store.id).count()
    if existing > 0:
        return

    sample = [
        {"name": "张小明", "pos": "收银员", "rate": 25.0, "max": 48, "skills": ["收银", "扫码"], "minor": False},
        {"name": "李美丽", "pos": "收银员", "rate": 28.0, "max": 44, "skills": ["收银", "会员卡"], "minor": False},
        {"name": "王大力", "pos": "理货员", "rate": 22.0, "max": 50, "skills": ["理货", "搬运"], "minor": False},
        {"name": "刘芳芳", "pos": "理货员", "rate": 24.0, "max": 40, "skills": ["理货", "陈列"], "minor": False},
        {"name": "陈星星", "pos": "导购员", "rate": 30.0, "max": 40, "skills": ["导购", "收银"], "minor": False},
        {"name": "赵小花", "pos": "收银员", "rate": 20.0, "max": 30, "skills": ["收银"], "minor": True},
        {"name": "孙国庆", "pos": "导购员", "rate": 26.0, "max": 45, "skills": ["导购", "陈列"], "minor": False},
        {"name": "周晓华", "pos": "理货员", "rate": 23.0, "max": 48, "skills": ["理货", "收货"], "minor": False},
    ]

    for i, s in enumerate(sample):
        eid = f"emp_{store.id}_{i+1:03d}"
        emp = DBEmployee(
            id=eid, name=s["name"], position=s["pos"], hourly_rate=s["rate"],
            max_weekly_hours=s["max"], skills=s["skills"], is_minor=s["minor"],
            store_id=store.id,
        )
        for wd in Weekday:
            for slot in (ShiftSlot.MORNING, ShiftSlot.MIDDAY, ShiftSlot.EVENING):
                av = DBAvailability(
                    weekday=wd.value, slot=slot.value, available=True, preferred=False
                )
                if s["name"] == "赵小花" and slot == ShiftSlot.EVENING:
                    av.available = False
                if s["name"] == "刘芳芳" and wd == Weekday.FRIDAY and slot in (ShiftSlot.EVENING, ShiftSlot.NIGHT):
                    av.available = False
                if s["name"] == "李美丽" and wd in (Weekday.WEDNESDAY, Weekday.THURSDAY) and slot == ShiftSlot.EVENING:
                    av.available = False
                emp.availabilities.append(av)
            if s["minor"]:
                night_av = DBAvailability(
                    weekday=wd.value, slot=ShiftSlot.NIGHT.value, available=False, preferred=False
                )
                emp.availabilities.append(night_av)
        db.add(emp)
    db.commit()


def build_default_demands(positions: List[str]) -> List[OptimizerShiftDemand]:
    demands = []
    for pos in positions:
        for wd in Weekday:
            is_weekend = wd in (Weekday.SATURDAY, Weekday.SUNDAY)
            for slot in (ShiftSlot.MORNING, ShiftSlot.MIDDAY, ShiftSlot.EVENING):
                if is_weekend:
                    cnt = 3 if slot in (ShiftSlot.MIDDAY, ShiftSlot.EVENING) else 2
                else:
                    if slot == ShiftSlot.MIDDAY:
                        cnt = 2
                    elif slot == ShiftSlot.MORNING:
                        cnt = 1
                    else:
                        cnt = 2 if pos == "收银员" else 1
                demands.append(OptimizerShiftDemand(
                    weekday=wd, slot=slot, position=pos, required_count=cnt
                ))
    return demands


def display_schedule(result: OptimizationResult, week_start: date, store_name: str):
    shifts_by_emp: Dict[str, List] = {}
    for s in result.shifts:
        shifts_by_emp.setdefault(s.employee_id, []).append(s)

    title = Text(f"门店: {store_name}  |  排班周期: {week_start} ~ {week_start + timedelta(days=6)}", style="bold cyan")
    subtitle = Text(
        f"状态: {result.message}  |  总班次: {len(result.shifts)}  |  "
        f"总工时: {result.total_hours}h  |  总成本: ¥{result.total_cost:.2f}  |  耗时: {result.solve_time_seconds}s",
        style="italic"
    )
    header = Text.assemble(title, "\n", subtitle)
    console.print(Panel(header, border_style="blue", title="📋 排班方案"))

    emp_names = {}
    for s in result.shifts:
        emp_names[s.employee_id] = s.employee_name

    table = Table(show_header=True, header_style="bold magenta", show_lines=False, expand=True)
    table.add_column("员工", style="bold", width=10)
    table.add_column("岗位", style="dim", width=8)
    for d in range(7):
        dt = week_start + timedelta(days=d)
        label = f"{WEEKDAY_CN[d]}\n{dt.month}/{dt.day}"
        col_style = "bold red" if d >= 5 else None
        table.add_column(label, justify="center", style=col_style)
    table.add_column("工时", justify="right", width=6)
    table.add_column("成本", justify="right", style="green", width=8)

    for eid, shifts in sorted(shifts_by_emp.items(), key=lambda kv: emp_names.get(kv[0], kv[0])):
        day_cells: Dict[int, str] = {}
        total_h = 0.0
        total_c = 0.0
        pos = ""
        for s in shifts:
            day_idx = (s.date - week_start).days
            if 0 <= day_idx < 7:
                emoji = SLOT_EMOJI.get(s.slot, "")
                cn = SLOT_CN.get(s.slot, s.slot.value)
                day_cells[day_idx] = f"{emoji}{cn}"
            total_h += s.hours
            total_c += s.cost
            pos = s.position

        row = [emp_names.get(eid, eid)[:8], pos]
        for d in range(7):
            row.append(day_cells.get(d, "-"))
        row.append(f"{total_h:.0f}h")
        row.append(f"¥{total_c:.0f}")
        table.add_row(*row)

    console.print(table)

    legend = Table(show_header=False, box=None, padding=(0, 2))
    legend.add_column("时段", style="bold")
    for slot in ShiftSlot:
        legend.add_row(f"{SLOT_EMOJI.get(slot)} {SLOT_CN.get(slot)}")
    console.print(legend)


def run_validation(result: OptimizationResult, opt_emps: List[OptimizerEmployee], week_start: date):
    validator = ShiftValidator(opt_emps)
    vr = validator.validate(result.shifts, week_start=week_start)

    if vr.is_compliant and vr.total_violations == 0:
        console.print("\n[bold green]✅ 合规检查：全部通过[/bold green]")
        return

    console.print()
    v_title = Text(
        f"合规检查: {vr.total_violations}项问题 (严重:{vr.critical_count} 警告:{vr.warning_count} 建议:{vr.info_count})",
        style="bold red" if vr.critical_count > 0 else "bold yellow"
    )
    console.print(Panel(v_title, border_style="red" if vr.critical_count > 0 else "yellow", title="⚠️ 合规检查"))

    v_table = Table(show_header=True, header_style="bold")
    v_table.add_column("级别", width=8)
    v_table.add_column("分类", width=14)
    v_table.add_column("问题描述")
    v_table.add_column("员工", width=10)
    v_table.add_column("建议")

    sev_style = {
        Severity.CRITICAL: "bold red",
        Severity.WARNING: "bold yellow",
        Severity.INFO: "bold blue",
    }
    sev_label = {
        Severity.CRITICAL: "🔴 严重",
        Severity.WARNING: "🟡 警告",
        Severity.INFO: "🔵 建议",
    }

    for v in vr.violations[:20]:
        v_table.add_row(
            sev_label.get(v.severity, v.severity.value),
            v.category,
            v.message[:40],
            (v.employee_name or "-")[:8],
            v.suggestion[:30],
            style=sev_style.get(v.severity),
        )
    if len(vr.violations) > 20:
        v_table.add_row("", "", f"... 另有 {len(vr.violations) - 20} 项未显示", "", "")
    console.print(v_table)


def write_to_db(db: Session, result: OptimizationResult, store_id: str, week_start: date):
    db.query(DBShift).filter(
        DBShift.week_start == week_start,
        DBShift.store_id == store_id,
    ).delete(synchronize_session=False)

    count = 0
    for s in result.shifts:
        sh = DBShift(
            employee_id=s.employee_id, date=s.date,
            start_time=s.start_time, end_time=s.end_time,
            slot=s.slot.value, position=s.position,
            hours=s.hours, cost=s.cost, week_start=week_start,
            store_id=store_id,
        )
        db.add(sh)
        count += 1
    db.commit()
    console.print(f"\n[bold green]✅ 已写入数据库: {count} 条排班记录[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="门店员工快速排班工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--store", required=True, help="门店名称，例如 门店A")
    parser.add_argument("--week", required=True, help="排班周，格式 YYYY-WNN 例如 2026-W12")
    parser.add_argument("--time-limit", type=float, default=30.0, help="求解器时间限制（秒），默认30")
    parser.add_argument("--dry-run", action="store_true", help="仅预览不写入数据库")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认直接写入")
    parser.add_argument("--seed-demo", action="store_true", help="如果数据库为空则注入示例员工")

    args = parser.parse_args()

    try:
        week_start = parse_week_string(args.week)
    except ValueError as e:
        console.print(f"[bold red]❌ 参数错误: {e}[/bold red]")
        sys.exit(1)

    console.print(f"[bold]🚀 启动排班引擎[/bold]")
    console.print(f"  门店: [cyan]{args.store}[/cyan]")
    console.print(f"  周期: [cyan]{week_start} ~ {week_start + timedelta(days=6)}[/cyan]")
    console.print(f"  时限: [cyan]{args.time_limit}s[/cyan]")
    console.print()

    db = SessionLocal()
    try:
        store = get_or_create_store(db, args.store)
        console.print(f"[dim]门店ID: {store.id}[/dim]")

        if args.seed_demo:
            ensure_sample_employees(db, store)

        q = db.query(DBEmployee).filter(DBEmployee.store_id == store.id)
        db_emps = q.all()

        if not db_emps:
            console.print(f"[bold yellow]⚠️  该门店暂无员工数据，请先通过API导入。使用 --seed-demo 可生成示例数据。[/bold yellow]")
            sys.exit(2)

        console.print(f"[green]✓ 加载 {len(db_emps)} 名员工数据[/green]")

        history_q = db.query(DBShift.employee_id, func.count(DBShift.id)).filter(
            DBShift.date >= week_start - timedelta(weeks=4),
            DBShift.date < week_start,
            func.strftime("%w", DBShift.date).in_(["0", "6"]),
        ).group_by(DBShift.employee_id)
        history_weekend = {eid: cnt for eid, cnt in history_q.all()}

        opt_emps = [_to_opt_emp(e) for e in db_emps]
        positions = list(dict.fromkeys([e.position for e in opt_emps]))

        demands = build_default_demands(positions)

        console.print(f"[green]✓ 构建 {len(demands)} 个人力需求约束[/green]")
        console.print()

        with console.status("[bold cyan]🤖 OR-Tools求解中，请稍候...[/bold cyan]", spinner="dots"):
            optimizer = ShiftOptimizer(time_limit_seconds=args.time_limit)
            result = optimizer.optimize(opt_emps, demands, week_start, history_weekend)

        if result.success:
            console.print(f"[green]✓ 求解完成[/green]")
            console.print()

            display_schedule(result, week_start, args.store)
            run_validation(result, opt_emps, week_start)

            if not args.dry_run:
                if args.yes or Confirm.ask("\n[bold yellow]是否写入数据库？", default=True):
                    write_to_db(db, result, store.id, week_start)
                else:
                    console.print("[dim]（未写入数据库）[/dim]")
            else:
                console.print("\n[dim]📌 --dry-run 模式：未写入数据库[/dim]")
        else:
            console.print(f"[bold red]❌ {result.message}[/bold red]")
    finally:
        db.close()


if __name__ == "__main__":
    main()
