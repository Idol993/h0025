from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
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


SLOT_TIME_RANGES: Dict[ShiftSlot, Tuple[time, time]] = {
    ShiftSlot.MORNING: (time(8, 0), time(16, 0)),
    ShiftSlot.MIDDAY: (time(10, 0), time(18, 0)),
    ShiftSlot.EVENING: (time(14, 0), time(22, 0)),
    ShiftSlot.NIGHT: (time(22, 0), time(6, 0)),
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
        delta = datetime.combine(date.today(), end) - datetime.combine(date.today(), start)
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
    date: date
    start_time: time
    end_time: time
    slot: ShiftSlot
    position: str
    hours: float
    cost: float


class OptimizationResult(BaseModel):
    success: bool
    converged: bool
    message: str
    total_cost: float
    total_hours: float
    shifts: List[ScheduledShift] = Field(default_factory=list)
    solve_time_seconds: float = 0.0


@dataclass
class SolutionCollector(cp_model.CpSolverSolutionCallback):
    model: cp_model.CpModel
    x_vars: Dict[Tuple[str, Weekday, ShiftSlot], cp_model.IntVar]
    employees: List[Employee]
    week_start: date
    best_solution: Optional[Dict] = field(default=None)
    best_cost: int = field(default=10**18)

    def on_solution_callback(self):
        total_cost = 0
        shifts = []
        for emp in self.employees:
            for wd in Weekday:
                for slot in ShiftSlot:
                    key = (emp.id, wd, slot)
                    if key in self.x_vars and self.Value(self.x_vars[key]):
                        start, end = SLOT_TIME_RANGES[slot]
                        shift_date = self.week_start + timedelta(days=wd.value)
                        hours = emp.get_slot_hours(slot)
                        cost = int(hours * emp.hourly_rate * 100)
                        total_cost += cost
                        shifts.append({
                            "employee_id": emp.id,
                            "employee_name": emp.name,
                            "date": shift_date,
                            "start_time": start,
                            "end_time": end,
                            "slot": slot,
                            "position": emp.position,
                            "hours": hours,
                            "cost": cost / 100.0,
                        })
        if total_cost < self.best_cost:
            self.best_cost = total_cost
            self.best_solution = {"cost": total_cost, "shifts": shifts}


class ShiftOptimizer:
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

    def optimize(
        self,
        employees: List[Employee],
        demands: List[ShiftDemand],
        week_start: date,
        history_weekend_counts: Optional[Dict[str, int]] = None,
    ) -> OptimizationResult:
        start_time = datetime.now()
        self.model = cp_model.CpModel()
        self.x.clear()
        self.weekly_hours.clear()
        self.daily_slots.clear()
        history_weekend_counts = history_weekend_counts or {}

        valid_employee_ids = {e.id for e in employees}

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

        for demand in demands:
            matching_vars: List[cp_model.IntVar] = []
            for emp in employees:
                if emp.position != demand.position and demand.position:
                    continue
                if not self._employee_has_skills(emp, demand.required_skills):
                    continue
                key = (emp.id, demand.weekday, demand.slot)
                if key in self.x:
                    matching_vars.append(self.x[key])
            if matching_vars:
                self.model.Add(sum(matching_vars) >= demand.required_count)

        for emp in employees:
            for d in range(7):
                if d + 5 < 7:
                    consec_vars = []
                    for offset in range(6):
                        wd = Weekday(d + offset)
                        for slot in ShiftSlot:
                            key = (emp.id, wd, slot)
                            if key in self.x:
                                consec_vars.append(self.x[key])
                    if consec_vars:
                        self.model.Add(sum(consec_vars) <= 5)

        objective_terms = []
        preference_terms = []
        weekend_balance_terms = []

        for (emp_id, wd, slot), var in self.x.items():
            emp = next(e for e in employees if e.id == emp_id)
            if not emp:
                continue
            hours = emp.get_slot_hours(slot)
            cost_cents = int(hours * emp.hourly_rate * 100)
            objective_terms.append(var * cost_cents)

            if not emp.is_preferred(wd, slot):
                penalty = int(200 * emp.feedback_weight)
                preference_terms.append(var * penalty)

            if self._is_weekend(wd):
                hist_count = history_weekend_counts.get(emp_id, 0)
                balance_penalty = 300 + hist_count * 150
                weekend_balance_terms.append(var * balance_penalty)

        total_objective = sum(objective_terms) + sum(preference_terms) + sum(weekend_balance_terms)
        self.model.Minimize(total_objective)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.num_workers = self.num_workers
        solver.parameters.log_search_progress = False

        collector = SolutionCollector(
            self.model, self.x, employees, week_start
        )
        status = solver.SolveWithSolutionCallback(self.model, collector)

        solve_time = (datetime.now() - start_time).total_seconds()

        converged = status == cp_model.OPTIMAL
        success = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        shifts: List[ScheduledShift] = []
        total_cost = 0.0
        total_hours = 0.0

        if collector.best_solution is not None:
            for s in collector.best_solution["shifts"]:
                shift = ScheduledShift(**s)
                shifts.append(shift)
                total_cost += shift.cost
                total_hours += shift.hours
            if not success:
                success = True
                message = "超时，未收敛到全局最优，返回当前最优解"
            else:
                message = "求解成功" if converged else "找到可行解"
        elif success:
            for emp in employees:
                for wd in Weekday:
                    for slot in ShiftSlot:
                        key = (emp.id, wd, slot)
                        if key in self.x and solver.Value(self.x[key]):
                            start, end = SLOT_TIME_RANGES[slot]
                            shift_date = week_start + timedelta(days=wd.value)
                            hours = emp.get_slot_hours(slot)
                            cost = hours * emp.hourly_rate
                            shifts.append(ScheduledShift(
                                employee_id=emp.id,
                                employee_name=emp.name,
                                date=shift_date,
                                start_time=start,
                                end_time=end,
                                slot=slot,
                                position=emp.position,
                                hours=hours,
                                cost=cost,
                            ))
                            total_cost += cost
                            total_hours += hours
            message = "求解成功" if converged else "找到可行解"
        else:
            message = "求解失败：无法满足所有约束条件"

        return OptimizationResult(
            success=success,
            converged=converged,
            message=message,
            total_cost=round(total_cost, 2),
            total_hours=round(total_hours, 2),
            shifts=shifts,
            solve_time_seconds=round(solve_time, 3),
        )
