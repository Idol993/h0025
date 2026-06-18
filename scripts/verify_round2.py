"""快速验收脚本：验证第二轮4个改进点"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta, time
from solver.optimizer import ShiftOptimizer, Employee, ShiftDemand, Weekday, ShiftSlot, Availability
from solver.validator import ShiftValidator, ScheduledShift


def _build_avails(morning_pref=False, midday_pref=False, evening_pref=False, night_avail=False):
    """构造一周7天所有时段的可用性列表"""
    avs = []
    for w in range(7):
        for slot_key, pref in [
            (ShiftSlot.MORNING, morning_pref),
            (ShiftSlot.MIDDAY, midday_pref),
            (ShiftSlot.EVENING, evening_pref),
        ]:
            avs.append(Availability(weekday=Weekday(w), slot=slot_key, available=True, preferred=pref))
        if night_avail:
            avs.append(Availability(weekday=Weekday(w), slot=ShiftSlot.NIGHT, available=True, preferred=False))
    return avs


def test_1_unmet_demand_detection():
    """改进1：generate接口检测缺口并返回"""
    print("=" * 60)
    print("✅ 测试1：未满足需求（缺口）检测")
    print("=" * 60)

    emps = [
        Employee(
            id="E1", name="张三", position="收银员", hourly_rate=30,
            max_weekly_hours=48, skills=[], is_minor=False,
            availabilities=_build_avails()
        ),
    ]
    demands = [
        ShiftDemand(weekday=Weekday.MONDAY, slot=ShiftSlot.MORNING, position="收银员", required_count=3, required_skills=[]),
        ShiftDemand(weekday=Weekday.MONDAY, slot=ShiftSlot.MORNING, position="导购员", required_count=2, required_skills=[]),
    ]
    ws = date(2025, 1, 6)
    opt = ShiftOptimizer(time_limit_seconds=20)
    result = opt.optimize(employees=emps, demands=demands, week_start=ws)

    print(f"  消息: {result.message}")
    print(f"  成功: {result.success}")
    print(f"  总缺口数: {len(result.unmet_demands)}")
    for u in result.unmet_demands:
        print(f"  📌 {u.weekday.name}/{u.slot.value} 岗位[{u.position}] 需{u.required_count}人 排了{u.assigned_count}人 缺口{u.gap}人 → {u.reason}")
    assert result.success, f"求解失败：{result.message}"
    assert len(result.unmet_demands) >= 2, "应至少检测到2个缺口"
    print("  ✅ PASS\n")


def test_2_preference_weight_influence():
    """改进2：偏好权重足够高，能影响排班决策"""
    print("=" * 60)
    print("✅ 测试2：偏好/反馈/周末轮替权重校准 + 连续6天允许")
    print("=" * 60)

    # A员工时薪更低（25）但讨厌晚班（早/中班偏好=True→奖励，晚班偏好=False→惩罚×feedback_weight=2.0
    # B员工时薪稍高（30）但对晚班无所谓（全部偏好=False→中性惩罚×1.0）
    # 成本差：晚班8h×(30-25)=40元/次，5次共200元=20000分
    # 偏好惩罚差：A的晚班惩罚6000×2.0 - B的晚班惩罚6000×1.0 = 6000分/次，5次共30000分
    # 所以偏好项超过成本差 → 应该给B多排晚班
    emps = [
        Employee(
            id="A", name="便宜但讨厌晚班", position="营业员", hourly_rate=25,
            max_weekly_hours=60, skills=[], is_minor=False,
            feedback_weight=2.0,
            availabilities=_build_avails(morning_pref=True, midday_pref=True, evening_pref=False)
        ),
        Employee(
            id="B", name="稍贵但无所谓", position="营业员", hourly_rate=30,
            max_weekly_hours=60, skills=[], is_minor=False,
            feedback_weight=1.0,
            availabilities=_build_avails()
        ),
    ]

    # 周一到周五每天晚班1人，共5个晚班
    demands = [
        ShiftDemand(weekday=Weekday(w), slot=ShiftSlot.EVENING, position="营业员", required_count=1, required_skills=[])
        for w in range(5)
    ]
    # 同时添加6天连续排班测试：周一到周六每天1个早班（2人需求所以A+B各一次，保证6天连续）
    demands += [
        ShiftDemand(weekday=Weekday(w), slot=ShiftSlot.MORNING, position="营业员", required_count=1, required_skills=[])
        for w in range(6)  # 周一到周六 6天
    ]

    ws = date(2025, 1, 6)
    opt = ShiftOptimizer(time_limit_seconds=20)
    result = opt.optimize(employees=emps, demands=demands, week_start=ws)
    print(f"  消息: {result.message}")
    print(f"  成功: {result.success}")
    assert result.success, f"连续6天应允许求解，实际{result.message}"

    evening_by_emp: dict = {}
    for a in result.shifts:
        if a.slot == ShiftSlot.EVENING:
            evening_by_emp[a.employee_id] = evening_by_emp.get(a.employee_id, 0) + 1
    print(f"  晚班分配：便宜但讨厌晚班={evening_by_emp.get('A',0)}次；稍贵但无所谓={evening_by_emp.get('B',0)}次")
    # 偏好权重生效：晚班应尽量给不怕晚班的B
    b_evenings = evening_by_emp.get("B", 0)
    assert b_evenings >= 3, f"偏好惩罚应生效，B至少承担3次晚班，实际{b_evenings}"
    print("  ✅ 连续6天不阻止求解 + 偏好权重生效")
    print("  ✅ PASS\n")


def test_3_smart_bool_parsing():
    """改进3：智能布尔解析"""
    print("=" * 60)
    print("✅ 测试3：智能布尔解析")
    print("=" * 60)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))
    # 导入main里的smart_parse_bool
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_mod", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server", "main.py"))
    mod = importlib.util.module_from_spec(spec)
    # 避免执行Base.metadata.create_all
    import types
    fake_sqlalchemy = types.ModuleType("fake")
    # 但更简单直接拷贝函数测试
    from server.main import smart_parse_bool

    false_cases = ["否", "不可用", "false", "FALSE", "No", 0, "0", "不", "休息", "请假"]
    true_cases = ["是", "可用", "true", "TRUE", "Yes", 1, "1", "可", "能", "上班"]
    all_pass = True
    for c in false_cases:
        got = smart_parse_bool(c, default=True)
        status = "✅" if got is False else "❌"
        if got is not False:
            all_pass = False
        print(f"  {status} smart_parse_bool({repr(c)}) = {got} (expect False)")
    for c in true_cases:
        got = smart_parse_bool(c, default=False)
        status = "✅" if got is True else "❌"
        if got is not True:
            all_pass = False
        print(f"  {status} smart_parse_bool({repr(c)}) = {got} (expect True)")
    # 默认值
    assert smart_parse_bool(None, default=True) is True, "None应取默认值True"
    assert smart_parse_bool(None, default=False) is False, "None应取默认值False"
    print(f"  默认值检查: smart_parse_bool(None,True)={smart_parse_bool(None,True)} smart_parse_bool(None,False)={smart_parse_bool(None,False)}")
    assert all_pass
    print("  ✅ PASS\n")


def test_4_validator_consecutive_weekends_and_days():
    """改进4：连续周末+连续6/7天阈值"""
    print("=" * 60)
    print("✅ 测试4：validator连续周末检查 + 连续6/7天阈值")
    print("=" * 60)

    emps = [
        Employee(id="W1", name="周末劳模", position="营业员", hourly_rate=25,
                 max_weekly_hours=60, skills=[], is_minor=False,
                 availability={(w, s): (True, False) for w in range(7) for s in ["morning", "midday", "evening"]}),
        Employee(id="W2", name="连续6天员工", position="营业员", hourly_rate=25,
                 max_weekly_hours=60, skills=[], is_minor=False,
                 availability={(w, s): (True, False) for w in range(7) for s in ["morning", "midday", "evening"]}),
        Employee(id="W3", name="连续7天员工", position="营业员", hourly_rate=25,
                 max_weekly_hours=60, skills=[], is_minor=False,
                 availability={(w, s): (True, False) for w in range(7) for s in ["morning", "midday", "evening"]}),
    ]
    validator = ShiftValidator(employees=emps)

    # 构造连续周末：2025年1月每周六/日都排班 → 连续4个周末
    # 2025-01-04(Sat) 01-05(Sun)  01-11(Sat) 01-12(Sun)  01-18(Sat) 01-19(Sun)  01-25(Sat) 01-26(Sun)
    weekend_dates = [date(2025,1,4),date(2025,1,5), date(2025,1,11),date(2025,1,12),
                     date(2025,1,18),date(2025,1,19), date(2025,1,25),date(2025,1,26)]
    shifts_W1 = [ScheduledShift(
        employee_id="W1", employee_name="周末劳模", date=d,
        start_time=time(9,0), end_time=time(17,0), slot=ShiftSlot.MORNING,
        position="营业员", hours=8, cost=200,
    ) for d in weekend_dates]

    # W2: 1月6日-11日 连续6天
    shifts_W2 = [ScheduledShift(
        employee_id="W2", employee_name="连续6天员工", date=date(2025,1,6+i),
        start_time=time(9,0), end_time=time(17,0), slot=ShiftSlot.MORNING,
        position="营业员", hours=8, cost=200,
    ) for i in range(6)]

    # W3: 1月6日-12日 连续7天
    shifts_W3 = [ScheduledShift(
        employee_id="W3", employee_name="连续7天员工", date=date(2025,1,6+i),
        start_time=time(9,0), end_time=time(17,0), slot=ShiftSlot.MORNING,
        position="营业员", hours=8, cost=200,
    ) for i in range(7)]

    all_shifts = shifts_W1 + shifts_W2 + shifts_W3
    report = validator.validate(all_shifts, week_start=date(2025,1,6))

    print(f"  报告: 违规={report.total_violations} (严重={report.critical_count} 警告={report.warning_count} 建议={report.info_count})")
    for v in report.violations:
        print(f"    [{v.severity.value}] {v.employee_name}: {v.message}")

    # 断言
    info_msgs = [v.message for v in report.violations if v.severity.value == "info"]
    warn_msgs = [v.message for v in report.violations if v.severity.value == "warning"]
    crit_msgs = [v.message for v in report.violations if v.severity.value == "critical"]

    # 1. W1应有连续周末提示（INFO）
    assert any("连续4个周末" in m for m in info_msgs), "连续4周末应触发INFO建议"
    # 2. W2连续6天是警告（WARNING），不是严重
    assert any("达6天" in m for m in warn_msgs), "连续6天应是WARNING预警"
    assert not any(("达6天" in m) or ("连6天" in m) for m in crit_msgs), "连续6天不应是严重违规"
    # 3. W3连续7天是严重违规（CRITICAL）
    assert any(("7天" in m) and ("超" in m) for m in crit_msgs), "连续7天应是严重违规"

    print("  ✅ 连续4周末INFO、连续6天WARNING、连续7天CRITICAL")
    print("  ✅ PASS\n")


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  第二轮4个改进点 快速验收")
    print("█" * 60 + "\n")
    test_1_unmet_demand_detection()
    test_2_preference_weight_influence()
    test_3_smart_bool_parsing()
    test_4_validator_consecutive_weekends_and_days()
    print("\n🎉 全部验收通过！\n")
