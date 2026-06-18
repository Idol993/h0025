import math
from dataclasses import dataclass, field
from datetime import date as Date, datetime as DateTime, time as Time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model
from pydantic import BaseModel, Field


class Weekday(int, Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class ShiftSlot(str, Enum):
    MORNING = "morning"
    MIDDAY = "midday"
    EVENING = "evening"
    NIGHT = "night"


class ScheduleStatus(str, Enum):
    FULL_SUCCESS = "full_success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


SLOT_TIME_RANGES: Dict[ShiftSlot, Tuple[Time, Time]] = {
    ShiftSlot.MORNING: (Time(8, 0), Time(16, 0)),
    ShiftSlot.MIDDAY: (Time(10, 0), Time(18, 0)),
    ShiftSlot.EVENING: (Time(14, 0), Time(22, 0)),
    ShiftSlot.NIGHT: (Time(22, 0), Time(6, 0)),
}


WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
SLOT_NAMES = {
    ShiftSlot.MORNING: "早班(08:00-16:00)",
    ShiftSlot.MIDDAY: "中班(10:00-18:00)",
    ShiftSlot.EVENING: "晚班(14:00-22:00)",
    ShiftSlot.NIGHT: "夜班(22:00-06:00)",
}


class Availability(BaseModel):
    weekday: Weekday
    slot: ShiftSlot
    available: bool = True
    preferred: bool = False


class Employee(BaseModel):
    id: str
    name: str
    position: str
    hourly_rate: float = Field(..., gt=0)
    max_weekly_hours: int = Field(default=60, ge=0, le=100)
    skills: List[str] = Field(default_factory=list)
    is_minor: bool = False
    availabilities: List[Availability] = Field(default_factory=list)
    feedback_weight: float = Field(default=1.0, ge=0.5, le=2.0)

    def is_available(self, weekday: Weekday, slot: ShiftSlot) -> bool:
        for av in self.availabilities:
            if av.weekday == weekday and av.slot == slot:
                return av.available
        return True

    def is_preferred(self, weekday: Weekday, slot: ShiftSlot) -> bool:
        for av in self.availabilities:
            if av.weekday == weekday and av.slot == slot:
                return av.preferred
        return False

    def get_slot_hours(self, slot: ShiftSlot) -> float:
        start, end = SLOT_TIME_RANGES[slot]
        if slot == ShiftSlot.NIGHT:
            return 8.0
        delta = DateTime.combine(Date.today(), end) - DateTime.combine(Date.today(), start)
        return delta.total_seconds() / 3600.0


class ShiftDemand(BaseModel):
    weekday: Weekday
    slot: ShiftSlot
    position: str
    required_count: int = Field(..., ge=0)
    required_skills: List[str] = Field(default_factory=list)


class ScheduledShift(BaseModel):
    employee_id: str
    employee_name: str
    date: Date
    start_time: Time
    end_time: Time
    slot: ShiftSlot
    position: str
    hours: float
    cost: float


class UnmetDemand(BaseModel):
    weekday: Weekday
    weekday_name: str
    date: Date | None = None
    slot: ShiftSlot
    slot_name: str
    position: str
    required_count: int
    assigned_count: int
    gap: int
    required_skills: List[str] = Field(default_factory=list)
    reason: str = ""


class RemedyItem(BaseModel):
    category: str
    position: str = ""
    weekday: Weekday | None = None
    slot: ShiftSlot | None = None
    gap: int = 0
    suggested_employees: List[str] = Field(default_factory=list)
    description: str = ""


class RemedySuggestion(BaseModel):
    temp_worker_count: int = 0
    items: List[RemedyItem] = Field(default_factory=list)
    summary: str = ""


class OptimizationResult(BaseModel):
    success: bool
    status: ScheduleStatus
    converged: bool
    message: str
    total_cost: float
    total_hours: float
    shifts: List[ScheduledShift] = Field(default_factory=list)
    unmet_demands: List[UnmetDemand] = Field(default_factory=list)
    remedy_suggestion: RemedySuggestion | None = None
    solve_time_seconds: float = 0.0


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, model, x_vars, employees, week_start, demands):
        super().__init__()
        self.model = model
        self.x_vars = x_vars
        self.employees = employees
        self.week_start = week_start
        self.demands = demands
        self.best_solution = None
        self.best_objective = 10**18
        self.current_best_assignment: Dict[Tuple[str, Weekday, ShiftSlot], int] = {}

    def on_solution_callback(self):
        objective = int(self.ObjectiveValue())
        shifts = []
        assignment: Dict[Tuple[str, Weekday, ShiftSlot], int] = {}
        for emp in self.employees:
            for wd in Weekday:
                for slot in ShiftSlot:
                    key = (emp.id, wd, slot)
                    val = 0
                    if key in self.x_vars:
                        val = self.Value(self.x_vars[key])
                    assignment[key] = val
                    if val:
                        start, end = SLOT_TIME_RANGES[slot]
                        shift_date = self.week_start + timedelta(days=wd.value)
                        hours = emp.get_slot_hours(slot)
                        shifts.append({
                            "employee_id": emp.id,
                            "employee_name": emp.name,
                            "date": shift_date,
                            "start_time": start,
                            "end_time": end,
                            "slot": slot,
                            "position": emp.position,
                            "hours": hours,
                            "cost": round(hours * emp.hourly_rate, 2),
                        })
        if objective < self.best_objective:
            self.best_objective = objective
            self.best_solution = {"objective": objective, "shifts": shifts}
            self.current_best_assignment = assignment


class ShiftOptimizer:
    COST_WEIGHT = 100
    PREFERENCE_BASE_PENALTY = 6000
    WEEKEND_BASE_PENALTY = 8000
    WEEKEND_HISTORY_PENALTY = 4000
    SLOT_HOURS_STANDARD = 8

    def __init__(self, time_limit_seconds: float = 30.0, num_workers: int = 8):
        self.time_limit_seconds = time_limit_seconds
        self.num_workers = num_workers
        self.model = cp_model.CpModel()
        self.x: Dict[Tuple[str, Weekday, ShiftSlot], cp_model.IntVar] = {}
        self.weekly_hours: Dict[str, cp_model.IntVar] = {}
        self.daily_slots: Dict[Tuple[str, Weekday], List[cp_model.IntVar]] = {}

    def _employee_has_skills(self, emp: Employee, required: List[str]) -> bool:
        if not required:
            return True
        return all(skill in emp.skills for skill in required)

    def _is_weekend(self, wd: Weekday) -> bool:
        return wd in (Weekday.SATURDAY, Weekday.SUNDAY)

    def _is_night_shift(self, slot: ShiftSlot) -> bool:
        return slot == ShiftSlot.NIGHT

    def _compute_unmet_demands(
        self,
        demands: List[ShiftDemand],
        assignment: Dict[Tuple[str, Weekday, ShiftSlot], int],
        employees: List[Employee],
        week_start: Date,
    ) -> List[UnmetDemand]:
        result: List[UnmetDemand] = []
        emp_by_id = {e.id: e for e in employees}

        for demand in demands:
            assigned = 0
            for emp in employees:
                if emp.position != demand.position and demand.position:
                    continue
                if not self._employee_has_skills(emp, demand.required_skills):
                    continue
                key = (emp.id, demand.weekday, demand.slot)
                if assignment.get(key, 0):
                    assigned += 1

            gap = demand.required_count - assigned
            if gap <= 0:
                continue

            qual_count = 0
            avail_count = 0
            for emp in employees:
                if emp.position != demand.position and demand.position:
                    continue
                if not self._employee_has_skills(emp, demand.required_skills):
                    continue
                qual_count += 1
                if emp.is_available(demand.weekday, demand.slot):
                    if not (emp.is_minor and self._is_night_shift(demand.slot)):
                        avail_count += 1

            reasons = []
            if qual_count == 0:
                reasons.append("无符合岗位或技能的员工")
            elif avail_count == 0:
                reasons.append("符合条件的员工均不可用/未成年禁夜班")
            elif avail_count < demand.required_count:
                reasons.append(f"仅{avail_count}人可用，需求{demand.required_count}人")
            else:
                reasons.append("可用人数足够但受工时/班次约束不足")
            reason = "；".join(reasons)

            result.append(UnmetDemand(
                weekday=demand.weekday,
                weekday_name=WEEKDAY_NAMES[demand.weekday.value],
                date=week_start + timedelta(days=demand.weekday.value),
                slot=demand.slot,
                slot_name=SLOT_NAMES[demand.slot],
                position=demand.position,
                required_count=demand.required_count,
                assigned_count=assigned,
                gap=gap,
                required_skills=list(demand.required_skills),
                reason=reason,
            ))
        return result

    def _generate_remedy_suggestions(
        self,
        unmet_demands: List[UnmetDemand],
        demands: List[ShiftDemand],
        employees: List[Employee],
    ) -> RemedySuggestion:
        items: List[RemedyItem] = []
        total_temp_needed = 0

        for ud in unmet_demands:
            gap = ud.gap
            if gap <= 0:
                continue

            fully_qualified: List[str] = []
            avail_adjust: List[str] = []
            skill_gap_list: List[str] = []
            cross_position: List[str] = []

            for emp in employees:
                position_match = emp.position == ud.position or not ud.position
                skill_match = self._employee_has_skills(emp, ud.required_skills) if ud.required_skills else True
                available = emp.is_available(ud.weekday, ud.slot)
                minor_night_ok = not (emp.is_minor and self._is_night_shift(ud.slot))

                if position_match and skill_match and available and minor_night_ok:
                    fully_qualified.append(f"{emp.name}({emp.id})")
                elif position_match and skill_match and not available and minor_night_ok:
                    avail_adjust.append(f"{emp.name}({emp.id})")
                elif position_match and not skill_match and available and minor_night_ok:
                    missing = [s for s in ud.required_skills if s not in emp.skills]
                    skill_gap_list.append(f"{emp.name}({emp.id})缺{','.join(missing)}")
                elif not position_match and skill_match and available and minor_night_ok:
                    cross_position.append(f"{emp.name}({emp.id})-{emp.position}")

            qual_count = len(fully_qualified)
            category = "understaffed"
            suggested = fully_qualified[:5]
            desc = ""

            if avail_adjust:
                category = "adjust_availability"
                suggested = avail_adjust[:5]
                if qual_count > 0:
                    desc = (f"{ud.weekday_name}{ud.slot_name} {ud.position} 已有{qual_count}人在岗，"
                            f"另有{len(avail_adjust)}人调整可用时间后可上岗，差{gap}人，建议先内部调整")
                else:
                    desc = f"{ud.weekday_name}{ud.slot_name} {ud.position} 有{len(avail_adjust)}人岗位技能都符合，调整可用时间即可补上{gap}个缺口"
            elif skill_gap_list:
                category = "skill_gap"
                suggested = skill_gap_list[:5]
                if qual_count > 0:
                    desc = (f"{ud.weekday_name}{ud.slot_name} {ud.position} 已有{qual_count}人在岗，"
                            f"另有{len(skill_gap_list)}人培训后可上岗，差{gap}人，建议先技能培训")
                else:
                    desc = f"{ud.weekday_name}{ud.slot_name} {ud.position} 有{len(skill_gap_list)}人岗位符合但缺技能，培训后可补上缺口"
            elif cross_position:
                category = "position_mismatch"
                suggested = cross_position[:5]
                if qual_count > 0:
                    desc = (f"{ud.weekday_name}{ud.slot_name} {ud.position} 已有{qual_count}人在岗，"
                            f"另有{len(cross_position)}人可跨岗支援，差{gap}人，建议先跨岗调配")
                else:
                    desc = f"{ud.weekday_name}{ud.slot_name} {ud.position} 有{len(cross_position)}人技能符合但岗位不同，可跨岗支援"
            elif qual_count == 0:
                category = "temp_worker"
                suggested = []
                total_temp_needed += gap
                desc = f"{ud.weekday_name}{ud.slot_name} {ud.position} 无合适内部员工，建议招{gap}名临时工"
            elif qual_count >= ud.required_count:
                category = "constraint_limited"
                suggested = fully_qualified[:5]
                desc = f"{ud.weekday_name}{ud.slot_name} {ud.position} 可用人数充足但受工时/班次约束无法排满，可调整排班规则"
            else:
                category = "understaffed"
                suggested = fully_qualified[:5]
                total_temp_needed += gap
                desc = f"{ud.weekday_name}{ud.slot_name} {ud.position} 人手不足，仅有{qual_count}人可用，差{gap}人，建议增派人手或招临时工"

            items.append(RemedyItem(
                category=category,
                position=ud.position,
                weekday=ud.weekday,
                slot=ud.slot,
                gap=gap,
                suggested_employees=suggested,
                description=desc,
            ))

        if total_temp_needed > 0:
            internal_items = [i for i in items if i.category not in ("temp_worker", "understaffed")]
            summary = f"共需{total_temp_needed}名人手补充缺口（含临时工/增派人手），其中{len(internal_items)}项可通过内部调整（可用时间/技能/跨岗）解决"
        elif items:
            summary = f"全部{len(items)}项缺口均可通过内部调整解决"
        else:
            summary = "无缺口，排班完美满足"

        return RemedySuggestion(
            temp_worker_count=total_temp_needed,
            items=items,
            summary=summary,
        )

    def optimize(
        self,
        employees: List[Employee],
        demands: List[ShiftDemand],
        week_start: Date,
        history_weekend_counts: Optional[Dict[str, int]] = None,
    ) -> OptimizationResult:
        start_time = DateTime.now()
        self.model = cp_model.CpModel()
        self.x.clear()
        self.weekly_hours.clear()
        self.daily_slots.clear()
        history_weekend_counts = history_weekend_counts or {}

        for emp in employees:
            weekly_minutes_var = self.model.NewIntVar(0, emp.max_weekly_hours * 60, f"weekly_min_{emp.id}")
            self.weekly_hours[emp.id] = weekly_minutes_var
            day_minutes_list: List[cp_model.IntVar] = []

            for wd in Weekday:
                daily_vars: List[cp_model.IntVar] = []
                for slot in ShiftSlot:
                    if not emp.is_available(wd, slot):
                        continue
                    if emp.is_minor and self._is_night_shift(slot):
                        continue
                    var = self.model.NewBoolVar(f"x_{emp.id}_{wd.value}_{slot.value}")
                    self.x[(emp.id, wd, slot)] = var
                    daily_vars.append(var)
                    minutes = int(emp.get_slot_hours(slot) * 60)
                    day_minutes_list.append(var * minutes)

                self.daily_slots[(emp.id, wd)] = daily_vars
                self.model.Add(sum(daily_vars) <= 1)

            self.model.Add(sum(day_minutes_list) == weekly_minutes_var)
            self.model.Add(weekly_minutes_var <= emp.max_weekly_hours * 60)

        demand_variables_map: Dict[int, List[cp_model.IntVar]] = {}
        for di, demand in enumerate(demands):
            matching_vars: List[cp_model.IntVar] = []
            for emp in employees:
                if emp.position != demand.position and demand.position:
                    continue
                if not self._employee_has_skills(emp, demand.required_skills):
                    continue
                key = (emp.id, demand.weekday, demand.slot)
                if key in self.x:
                    matching_vars.append(self.x[key])
            demand_variables_map[di] = matching_vars

            slack_var = self.model.NewIntVar(0, max(demand.required_count, 0), f"slack_d{di}")
            if matching_vars:
                self.model.Add(sum(matching_vars) + slack_var >= demand.required_count)
            else:
                if demand.required_count > 0:
                    self.model.Add(slack_var >= demand.required_count)

        for emp in employees:
            work_day_flags: List[cp_model.IntVar] = []
            for wd in Weekday:
                day_vars = self.daily_slots.get((emp.id, wd), [])
                if day_vars:
                    worked = self.model.NewBoolVar(f"worked_{emp.id}_d{wd.value}")
                    self.model.AddMaxEquality(worked, day_vars + [self.model.NewConstant(0)])
                    work_day_flags.append(worked)
                else:
                    work_day_flags.append(self.model.NewConstant(0))

            for start_day in range(7):
                window = []
                for offset in range(7):
                    if start_day + offset < 7:
                        window.append(work_day_flags[start_day + offset])
                if len(window) >= 7:
                    self.model.Add(sum(window) <= 6)

        objective_terms = []
        preference_terms = []
        weekend_balance_terms = []
        slack_terms = []

        COST_W = self.COST_WEIGHT
        PREF_W = self.PREFERENCE_BASE_PENALTY
        WKND_W = self.WEEKEND_BASE_PENALTY
        WKND_HIST_W = self.WEEKEND_HISTORY_PENALTY
        SLACK_PENALTY = 1000000

        for di, demand in enumerate(demands):
            if demand.required_count > 0:
                sv = self.model.NewIntVar(0, max(demand.required_count, 1), f"slack_pen_d{di}")
                matching = demand_variables_map.get(di, [])
                if matching:
                    self.model.Add(sv >= demand.required_count - sum(matching))
                else:
                    self.model.Add(sv == demand.required_count)
                slack_terms.append(sv * SLACK_PENALTY)

        for (emp_id, wd, slot), var in self.x.items():
            emp = next(e for e in employees if e.id == emp_id)
            if not emp:
                continue
            hours = emp.get_slot_hours(slot)
            cost_coeff = int(hours * emp.hourly_rate * COST_W)
            objective_terms.append(var * cost_coeff)

            if not emp.is_preferred(wd, slot):
                penalty = int(PREF_W * emp.feedback_weight)
                preference_terms.append(var * penalty)
            else:
                bonus = int(PREF_W * 0.5 * emp.feedback_weight)
                objective_terms.append(var * (-bonus))

            if self._is_weekend(wd):
                hist_count = history_weekend_counts.get(emp_id, 0)
                balance_penalty = WKND_W + hist_count * WKND_HIST_W
                weekend_balance_terms.append(var * balance_penalty)

        total_objective = (
            sum(objective_terms)
            + sum(preference_terms)
            + sum(weekend_balance_terms)
            + sum(slack_terms)
        )
        self.model.Minimize(total_objective)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.num_workers = self.num_workers
        solver.parameters.log_search_progress = False

        status = solver.Solve(self.model)

        solve_time = (DateTime.now() - start_time).total_seconds()

        converged = status == cp_model.OPTIMAL
        success = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        shifts: List[ScheduledShift] = []
        total_cost = 0.0
        total_hours = 0.0
        final_assignment: Dict[Tuple[str, Weekday, ShiftSlot], int] = {}

        if success:
            for emp in employees:
                for wd in Weekday:
                    for slot in ShiftSlot:
                        key = (emp.id, wd, slot)
                        if key not in self.x:
                            continue
                        val = int(solver.Value(self.x[key]))
                        final_assignment[key] = val
                        if val:
                            start_t, end_t = SLOT_TIME_RANGES[slot]
                            shift_date = week_start + timedelta(days=wd.value)
                            hours = emp.get_slot_hours(slot)
                            cost = round(hours * emp.hourly_rate, 2)
                            shifts.append(ScheduledShift(
                                employee_id=emp.id, employee_name=emp.name,
                                date=shift_date, start_time=start_t, end_time=end_t,
                                slot=slot, position=emp.position,
                                hours=hours, cost=cost,
                            ))
                            total_cost += cost
                            total_hours += hours
            if status == cp_model.OPTIMAL:
                message = "求解成功（全局最优）"
            else:
                message = "求解成功（可行解，未收敛到全局最优）"
        elif status == cp_model.INFEASIBLE:
            message = "求解失败：约束冲突，无法找到任何可行解"
        else:
            message = f"求解失败（状态码={status}）"

        unmet = self._compute_unmet_demands(demands, final_assignment, employees, week_start)

        has_unmet = len(unmet) > 0
        has_shifts = len(shifts) > 0

        if status == cp_model.INFEASIBLE or not has_shifts:
            schedule_status = ScheduleStatus.FAILED
            success_flag = False
            if not has_shifts and status != cp_model.INFEASIBLE:
                message = "排班失败：无任何班次可排"
        elif has_unmet:
            schedule_status = ScheduleStatus.PARTIAL_SUCCESS
            success_flag = False
            message = f"部分成功：已排{len(shifts)}个班次，{len(unmet)}项需求未完全满足"
        else:
            schedule_status = ScheduleStatus.FULL_SUCCESS
            success_flag = True
            if status == cp_model.OPTIMAL:
                message = "完全成功：所有需求已满足（全局最优）"
            else:
                message = "完全成功：所有需求已满足（可行解）"

        remedy = None
        if has_unmet:
            remedy = self._generate_remedy_suggestions(unmet, demands, employees)

        return OptimizationResult(
            success=success_flag,
            status=schedule_status,
            converged=converged,
            message=message,
            total_cost=round(total_cost, 2),
            total_hours=round(total_hours, 2),
            shifts=shifts,
            unmet_demands=unmet,
            remedy_suggestion=remedy,
            solve_time_seconds=round(solve_time, 3),
        )
