"""三场景验收：全成功/部分成功/失败 + 补救建议"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
from solver.optimizer import (
    ShiftOptimizer, Employee, ShiftDemand, Weekday, ShiftSlot,
    Availability, ScheduleStatus,
)


def _avails(pos="营业员", pref_slots=None):
    pref_slots = pref_slots or set()
    avs = []
    for w in range(7):
        for slot in [ShiftSlot.MORNING, ShiftSlot.MIDDAY, ShiftSlot.EVENING]:
            avs.append(Availability(
                weekday=Weekday(w), slot=slot,
                available=True, preferred=slot in pref_slots,
            ))
    return avs


def test_1_full_success():
    """场景1：全成功 - 员工充足，所有需求满足"""
    print("=" * 60)
    print("✅ 场景1：全成功 full_success")
    print("=" * 60)

    emps = [
        Employee(id="A1", name="员工1", position="营业员", hourly_rate=25,
                 max_weekly_hours=40, skills=[], availabilities=_avails()),
        Employee(id="A2", name="员工2", position="营业员", hourly_rate=28,
                 max_weekly_hours=40, skills=[], availabilities=_avails()),
        Employee(id="A3", name="员工3", position="营业员", hourly_rate=30,
                 max_weekly_hours=40, skills=[], availabilities=_avails()),
    ]
    demands = [
        ShiftDemand(weekday=Weekday.MONDAY, slot=ShiftSlot.MORNING, position="营业员",
                    required_count=2, required_skills=[]),
        ShiftDemand(weekday=Weekday.MONDAY, slot=ShiftSlot.EVENING, position="营业员",
                    required_count=1, required_skills=[]),
    ]
    opt = ShiftOptimizer(time_limit_seconds=15)
    result = opt.optimize(employees=emps, demands=demands, week_start=date(2025, 1, 6))

    print(f"  status: {result.status.value}")
    print(f"  success: {result.success}")
    print(f"  message: {result.message}")
    print(f"  班次数: {len(result.shifts)}")
    print(f"  缺口数: {len(result.unmet_demands)}")
    print(f"  补救建议: {'有' if result.remedy_suggestion else '无'}")

    assert result.status == ScheduleStatus.FULL_SUCCESS, f"应为full_success，实际{result.status}"
    assert result.success is True, "success应为True"
    assert len(result.unmet_demands) == 0, "应无缺口"
    assert result.remedy_suggestion is None, "全成功不应有补救建议"
    assert len(result.shifts) >= 3, "至少排3个班"
    print("  ✅ PASS\n")


def test_2_partial_success():
    """场景2：部分成功 - 有人但不够，有缺口有班次"""
    print("=" * 60)
    print("✅ 场景2：部分成功 partial_success")
    print("=" * 60)

    emps = [
        Employee(id="B1", name="收银员1", position="收银员", hourly_rate=25,
                 max_weekly_hours=40, skills=[], availabilities=_avails()),
    ]
    demands = [
        ShiftDemand(weekday=Weekday.MONDAY, slot=ShiftSlot.MORNING, position="收银员",
                    required_count=3, required_skills=[]),
        ShiftDemand(weekday=Weekday.MONDAY, slot=ShiftSlot.MORNING, position="导购员",
                    required_count=2, required_skills=[]),
    ]
    opt = ShiftOptimizer(time_limit_seconds=15)
    result = opt.optimize(employees=emps, demands=demands, week_start=date(2025, 1, 6))

    print(f"  status: {result.status.value}")
    print(f"  success: {result.success}")
    print(f"  message: {result.message}")
    print(f"  班次数: {len(result.shifts)}")
    print(f"  缺口数: {len(result.unmet_demands)}")
    if result.remedy_suggestion:
        print(f"  临时工需求: {result.remedy_suggestion.temp_worker_count}人")
        print(f"  补救摘要: {result.remedy_suggestion.summary}")
        for item in result.remedy_suggestion.items:
            print(f"    - [{item.category}] {item.description}")

    assert result.status == ScheduleStatus.PARTIAL_SUCCESS, f"应为partial_success，实际{result.status}"
    assert result.success is False, "success应为False（有缺口）"
    assert len(result.unmet_demands) >= 2, "至少2个缺口"
    assert len(result.shifts) >= 1, "至少排了1个班（部分成功）"
    assert result.remedy_suggestion is not None, "应有补救建议"
    assert result.remedy_suggestion.temp_worker_count >= 2, "至少需要2名临时工（导购员没人会）"
    print("  ✅ PASS\n")


def test_3_failed():
    """场景3：完全失败 - 需求岗位没人，0班次"""
    print("=" * 60)
    print("✅ 场景3：完全失败 failed")
    print("=" * 60)

    emps = [
        Employee(id="C1", name="厨师1", position="厨师", hourly_rate=35,
                 max_weekly_hours=40, skills=[], availabilities=_avails()),
    ]
    demands = [
        ShiftDemand(weekday=Weekday.MONDAY, slot=ShiftSlot.MORNING, position="电工",
                    required_count=2, required_skills=["电工证"]),
    ]
    opt = ShiftOptimizer(time_limit_seconds=15)
    result = opt.optimize(employees=emps, demands=demands, week_start=date(2025, 1, 6))

    print(f"  status: {result.status.value}")
    print(f"  success: {result.success}")
    print(f"  message: {result.message}")
    print(f"  班次数: {len(result.shifts)}")
    print(f"  缺口数: {len(result.unmet_demands)}")

    assert result.status == ScheduleStatus.FAILED, f"应为failed，实际{result.status}"
    assert result.success is False, "success应为False"
    assert len(result.shifts) == 0, "应为0个班次"
    assert len(result.unmet_demands) >= 1, "应有1个缺口"
    print("  ✅ PASS\n")


def test_4_available_remedy():
    """场景4：补救建议 - 可用时间调整类"""
    print("=" * 60)
    print("✅ 场景4：补救建议 - 可用时间调整类")
    print("=" * 60)

    # 两个员工岗位都符合，但其中1个该时段设为不可用
    avails_D1 = [
        Availability(weekday=Weekday.MONDAY, slot=ShiftSlot.MORNING, available=False, preferred=False),
    ] + [
        Availability(weekday=Weekday(w), slot=s, available=True, preferred=False)
        for w in range(7) for s in [ShiftSlot.MORNING, ShiftSlot.MIDDAY, ShiftSlot.EVENING]
        if not (w == 0 and s == ShiftSlot.MORNING)
    ]
    avails_D2 = _avails()

    emps = [
        Employee(id="D1", name="员工D1", position="营业员", hourly_rate=25,
                 max_weekly_hours=40, skills=[], availabilities=avails_D1),
        Employee(id="D2", name="员工D2", position="营业员", hourly_rate=30,
                 max_weekly_hours=40, skills=[], availabilities=avails_D2),
    ]
    demands = [
        ShiftDemand(weekday=Weekday.MONDAY, slot=ShiftSlot.MORNING, position="营业员",
                    required_count=2, required_skills=[]),
    ]
    opt = ShiftOptimizer(time_limit_seconds=15)
    result = opt.optimize(employees=emps, demands=demands, week_start=date(2025, 1, 6))

    print(f"  status: {result.status.value}")
    print(f"  success: {result.success}")
    print(f"  缺口数: {len(result.unmet_demands)}")
    if result.remedy_suggestion:
        print(f"  补救摘要: {result.remedy_suggestion.summary}")
        for item in result.remedy_suggestion.items:
            print(f"    - [{item.category}] {item.description}")
            if item.suggested_employees:
                print(f"      建议调整: {', '.join(item.suggested_employees)}")

    adjust_items = [i for i in result.remedy_suggestion.items if i.category == "adjust_availability"]
    assert len(adjust_items) >= 1, "应有可用时间调整类建议"
    assert any("员工D1" in s for s in adjust_items[0].suggested_employees), "应推荐D1调整可用时间"
    print("  ✅ PASS\n")


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  三场景状态 + 补救建议 验收")
    print("█" * 60 + "\n")
    test_1_full_success()
    test_2_partial_success()
    test_3_failed()
    test_4_available_remedy()
    print("\n🎉 全部4个场景验收通过！\n")
