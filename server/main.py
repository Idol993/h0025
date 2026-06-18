from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    Time,
    UniqueConstraint,
    create_engine,
    and_,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from solver.optimizer import (
    Availability as OptimizerAvailability,
    Employee as OptimizerEmployee,
    OptimizationResult,
    ScheduledShift,
    ShiftDemand as OptimizerShiftDemand,
    ShiftOptimizer,
    ShiftSlot,
    SLOT_TIME_RANGES,
    Weekday,
)
from solver.validator import Severity, ShiftValidator, ValidationReport, Violation


DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'scheduling.db')}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class FeedbackRating(str, Enum):
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    DISSATISFIED = "dissatisfied"


class DBStore(Base):
    __tablename__ = "stores"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    address: Mapped[Optional[str]] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)


class DBEmployee(Base):
    __tablename__ = "employees"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    position: Mapped[str] = mapped_column(String(64))
    hourly_rate: Mapped[float] = mapped_column(Float, nullable=False)
    max_weekly_hours: Mapped[int] = mapped_column(Integer, default=60)
    skills: Mapped[List] = mapped_column(JSON, default=list)
    is_minor: Mapped[bool] = mapped_column(Boolean, default=False)
    feedback_weight: Mapped[float] = mapped_column(Float, default=1.0)
    store_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("stores.id"))
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)
    availabilities: Mapped[List["DBAvailability"]] = relationship(back_populates="employee", cascade="all, delete-orphan")
    shifts: Mapped[List["DBShift"]] = relationship(back_populates="employee")
    feedbacks: Mapped[List["DBFeedback"]] = relationship(back_populates="employee", cascade="all, delete-orphan")


class DBAvailability(Base):
    __tablename__ = "availability"
    __table_args__ = (UniqueConstraint("employee_id", "weekday", "slot", name="uq_emp_weekday_slot"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[str] = mapped_column(String(64), ForeignKey("employees.id"), nullable=False)
    weekday: Mapped[int] = mapped_column(Integer, nullable=False)
    slot: Mapped[str] = mapped_column(String(32), nullable=False)
    available: Mapped[bool] = mapped_column(Boolean, default=True)
    preferred: Mapped[bool] = mapped_column(Boolean, default=False)
    employee: Mapped[DBEmployee] = relationship(back_populates="availabilities")


class DBShift(Base):
    __tablename__ = "shifts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[str] = mapped_column(String(64), ForeignKey("employees.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    end_time: Mapped[time] = mapped_column(Time, nullable=False)
    slot: Mapped[str] = mapped_column(String(32), nullable=False)
    position: Mapped[str] = mapped_column(String(64))
    hours: Mapped[float] = mapped_column(Float, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)
    week_start: Mapped[date] = mapped_column(Date, nullable=False)
    store_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("stores.id"))
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)
    employee: Mapped[DBEmployee] = relationship(back_populates="shifts")


class DBFeedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[str] = mapped_column(String(64), ForeignKey("employees.id"), nullable=False)
    week_start: Mapped[date] = mapped_column(Date, nullable=False)
    rating: Mapped[str] = mapped_column(String(32), nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)
    employee: Mapped[DBEmployee] = relationship(back_populates="feedbacks")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI(
    title="员工排班优化与合规检查系统",
    description="基于OR-Tools的连锁零售门店智能排班系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", timestamp=datetime.now())


class StoreCreate(BaseModel):
    id: Optional[str] = None
    name: str
    address: Optional[str] = None


class StoreResponse(BaseModel):
    id: str
    name: str
    address: Optional[str] = None
    created_at: datetime


@app.post("/api/stores", response_model=StoreResponse)
def create_store(store: StoreCreate, db: Session = Depends(get_db)):
    store_id = store.id or f"store_{uuid4().hex[:8]}"
    db_store = DBStore(id=store_id, name=store.name, address=store.address)
    db.add(db_store)
    db.commit()
    db.refresh(db_store)
    return db_store


@app.get("/api/stores", response_model=List[StoreResponse])
def list_stores(db: Session = Depends(get_db)):
    return db.query(DBStore).all()


class AvailabilityCreate(BaseModel):
    weekday: Weekday
    slot: ShiftSlot
    available: bool = True
    preferred: bool = False


class EmployeeCreate(BaseModel):
    id: Optional[str] = None
    name: str
    position: str
    hourly_rate: float = Field(..., gt=0)
    max_weekly_hours: int = Field(default=60, ge=1, le=100)
    skills: List[str] = Field(default_factory=list)
    is_minor: bool = False
    availabilities: List[AvailabilityCreate] = Field(default_factory=list)
    store_id: Optional[str] = None
    feedback_weight: float = Field(default=1.0, ge=0.5, le=2.0)


class EmployeeResponse(BaseModel):
    id: str
    name: str
    position: str
    hourly_rate: float
    max_weekly_hours: int
    skills: List[str]
    is_minor: bool
    feedback_weight: float
    store_id: Optional[str] = None
    created_at: datetime
    availabilities: List[dict] = Field(default_factory=list)


def _to_optimizer_employee(db_emp: DBEmployee) -> OptimizerEmployee:
    opts_avs = [
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
        availabilities=opts_avs,
        feedback_weight=db_emp.feedback_weight,
    )


@app.post("/api/employees", response_model=EmployeeResponse, status_code=201)
def create_employee(emp: EmployeeCreate, db: Session = Depends(get_db)):
    emp_id = emp.id or f"emp_{uuid4().hex[:8]}"
    db_emp = DBEmployee(
        id=emp_id,
        name=emp.name,
        position=emp.position,
        hourly_rate=emp.hourly_rate,
        max_weekly_hours=emp.max_weekly_hours,
        skills=emp.skills,
        is_minor=emp.is_minor,
        store_id=emp.store_id,
        feedback_weight=emp.feedback_weight,
    )
    for av in emp.availabilities:
        db_av = DBAvailability(
            weekday=av.weekday.value,
            slot=av.slot.value,
            available=av.available,
            preferred=av.preferred,
        )
        db_emp.availabilities.append(db_av)
    db.add(db_emp)
    db.commit()
    db.refresh(db_emp)
    resp = EmployeeResponse(
        id=db_emp.id, name=db_emp.name, position=db_emp.position,
        hourly_rate=db_emp.hourly_rate, max_weekly_hours=db_emp.max_weekly_hours,
        skills=list(db_emp.skills or []), is_minor=db_emp.is_minor,
        feedback_weight=db_emp.feedback_weight, store_id=db_emp.store_id,
        created_at=db_emp.created_at,
        availabilities=[{
            "weekday": a.weekday, "slot": a.slot,
            "available": a.available, "preferred": a.preferred,
        } for a in db_emp.availabilities],
    )
    return resp


@app.get("/api/employees", response_model=List[EmployeeResponse])
def list_employees(store_id: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(DBEmployee)
    if store_id:
        q = q.filter(DBEmployee.store_id == store_id)
    db_emps = q.all()
    result = []
    for db_emp in db_emps:
        result.append(EmployeeResponse(
            id=db_emp.id, name=db_emp.name, position=db_emp.position,
            hourly_rate=db_emp.hourly_rate, max_weekly_hours=db_emp.max_weekly_hours,
            skills=list(db_emp.skills or []), is_minor=db_emp.is_minor,
            feedback_weight=db_emp.feedback_weight, store_id=db_emp.store_id,
            created_at=db_emp.created_at,
            availabilities=[{
                "weekday": a.weekday, "slot": a.slot,
                "available": a.available, "preferred": a.preferred,
            } for a in db_emp.availabilities],
        ))
    return result


@app.get("/api/employees/{employee_id}", response_model=EmployeeResponse)
def get_employee(employee_id: str, db: Session = Depends(get_db)):
    db_emp = db.query(DBEmployee).filter(DBEmployee.id == employee_id).first()
    if not db_emp:
        raise HTTPException(status_code=404, detail="员工不存在")
    return EmployeeResponse(
        id=db_emp.id, name=db_emp.name, position=db_emp.position,
        hourly_rate=db_emp.hourly_rate, max_weekly_hours=db_emp.max_weekly_hours,
        skills=list(db_emp.skills or []), is_minor=db_emp.is_minor,
        feedback_weight=db_emp.feedback_weight, store_id=db_emp.store_id,
        created_at=db_emp.created_at,
        availabilities=[{
            "weekday": a.weekday, "slot": a.slot,
            "available": a.available, "preferred": a.preferred,
        } for a in db_emp.availabilities],
    )


@app.post("/api/employees/import")
async def import_employees_from_excel(
    file: UploadFile = File(...),
    store_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="仅支持Excel文件")
    contents = await file.read()
    try:
        xls = pd.ExcelFile(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Excel解析失败: {str(e)}")

    emp_df = xls.parse("员工信息") if "员工信息" in xls.sheet_names else xls.parse(xls.sheet_names[0])
    av_df = None
    if len(xls.sheet_names) > 1 and "可用时间" in xls.sheet_names:
        av_df = xls.parse("可用时间")
    elif "可用时段" in xls.sheet_names:
        av_df = xls.parse("可用时段")

    imported_count = 0
    skipped = 0
    errors: List[str] = []

    av_by_emp: Dict[str, List[dict]] = {}
    if av_df is not None and not av_df.empty:
        av_cols = av_df.columns.tolist()
        for _, row in av_df.iterrows():
            eid = str(row.get("员工ID", row.get("employee_id", ""))).strip()
            if not eid:
                continue
            weekday_raw = row.get("星期", row.get("weekday"))
            slot_raw = row.get("时段", row.get("slot"))
            if weekday_raw is None or slot_raw is None:
                continue
            try:
                wd_int = None
                if isinstance(weekday_raw, (int, float)):
                    wd_int = int(weekday_raw)
                else:
                    wd_str = str(weekday_raw).strip()
                    wd_map = {
                        "周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6,
                        "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3, "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6,
                        "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6,
                    }
                    wd_int = wd_map.get(wd_str.upper(), wd_map.get(wd_str))
                    if wd_int is None:
                        try:
                            wd_int = Weekday[wd_str.upper()].value
                        except KeyError:
                            continue
                slot_str = str(slot_raw).strip().lower()
                slot_map = {"早班": "morning", "中班": "midday", "晚班": "evening", "夜班": "night"}
                if slot_str in slot_map:
                    slot_val = slot_map[slot_str]
                else:
                    try:
                        slot_val = ShiftSlot(slot_str).value
                    except ValueError:
                        continue
                av_entry = {
                    "weekday": wd_int,
                    "slot": slot_val,
                    "available": bool(row.get("可用", row.get("available", True))),
                    "preferred": bool(row.get("偏好", row.get("preferred", False))),
                }
                av_by_emp.setdefault(eid, []).append(av_entry)
            except Exception:
                continue

    for _, row in emp_df.iterrows():
        try:
            eid = str(row.get("员工ID", row.get("id", row.get("employee_id", "")))).strip()
            name = str(row.get("姓名", row.get("name", ""))).strip()
            if not name:
                skipped += 1
                continue
            if not eid:
                eid = f"emp_{uuid4().hex[:8]}"
            position = str(row.get("岗位", row.get("position", ""))).strip() or "营业员"
            hourly_rate = float(row.get("时薪", row.get("hourly_rate", 20.0)))
            max_hours = int(row.get("最大周工时", row.get("max_weekly_hours", 60)))
            skills_raw = str(row.get("技能", row.get("skills", ""))).strip()
            skills = [s.strip() for s in skills_raw.replace("，", ",").split(",") if s.strip()] if skills_raw else []
            is_minor = bool(row.get("未成年", row.get("is_minor", False)))

            existing = db.query(DBEmployee).filter(DBEmployee.id == eid).first()
            if existing:
                existing.name = name
                existing.position = position
                existing.hourly_rate = hourly_rate
                existing.max_weekly_hours = max_hours
                existing.skills = skills
                existing.is_minor = is_minor
                if store_id:
                    existing.store_id = store_id
                for av in av_by_emp.get(eid, []):
                    found = False
                    for exist_av in existing.availabilities:
                        if exist_av.weekday == av["weekday"] and exist_av.slot == av["slot"]:
                            exist_av.available = av["available"]
                            exist_av.preferred = av["preferred"]
                            found = True
                            break
                    if not found:
                        existing.availabilities.append(DBAvailability(
                            weekday=av["weekday"], slot=av["slot"],
                            available=av["available"], preferred=av["preferred"],
                        ))
            else:
                db_emp = DBEmployee(
                    id=eid, name=name, position=position,
                    hourly_rate=hourly_rate, max_weekly_hours=max_hours,
                    skills=skills, is_minor=is_minor, store_id=store_id,
                )
                for av in av_by_emp.get(eid, []):
                    db_emp.availabilities.append(DBAvailability(
                        weekday=av["weekday"], slot=av["slot"],
                        available=av["available"], preferred=av["preferred"],
                    ))
                db.add(db_emp)
            imported_count += 1
        except Exception as e:
            errors.append(f"行{imported_count + skipped + 1}: {str(e)}")

    db.commit()
    return {
        "success": True,
        "imported": imported_count,
        "skipped": skipped,
        "errors": errors,
    }


class ShiftDemandItem(BaseModel):
    weekday: Weekday
    slot: ShiftSlot
    position: str
    required_count: int = Field(..., ge=0)
    required_skills: List[str] = Field(default_factory=list)


class ScheduleGenerateRequest(BaseModel):
    week_start: date
    store_id: Optional[str] = None
    demands: List[ShiftDemandItem] = Field(default_factory=list)
    employee_ids: Optional[List[str]] = None
    time_limit_seconds: float = Field(default=30.0, ge=5.0, le=300.0)
    write_to_db: bool = False

    @field_validator("week_start")
    @classmethod
    def check_monday(cls, v: date) -> date:
        if v.weekday() != 0:
            raise ValueError("week_start必须是周一")
        return v


@app.post("/api/schedule/generate", response_model=OptimizationResult)
def generate_schedule(req: ScheduleGenerateRequest, db: Session = Depends(get_db)):
    q = db.query(DBEmployee)
    if req.store_id:
        q = q.filter(DBEmployee.store_id == req.store_id)
    if req.employee_ids:
        q = q.filter(DBEmployee.id.in_(req.employee_ids))
    db_emps = q.all()
    if not db_emps:
        raise HTTPException(status_code=400, detail="未找到符合条件的员工")

    week_end = req.week_start + timedelta(days=6)
    history_q = db.query(DBShift.employee_id, func.count(DBShift.id)).filter(
        DBShift.date >= req.week_start - timedelta(weeks=4),
        DBShift.date < req.week_start,
        func.strftime("%w", DBShift.date).in_(["0", "6"]),
    ).group_by(DBShift.employee_id)
    history_weekend = {eid: cnt for eid, cnt in history_q.all()}

    optimizer_emps = [_to_optimizer_employee(e) for e in db_emps]
    optimizer_demands = [
        OptimizerShiftDemand(
            weekday=d.weekday, slot=d.slot, position=d.position,
            required_count=d.required_count, required_skills=d.required_skills,
        )
        for d in req.demands
    ]
    if not optimizer_demands:
        default_positions = list({e.position for e in db_emps})
        for pos in default_positions:
            for wd in Weekday:
                is_wknd = wd in (Weekday.SATURDAY, Weekday.SUNDAY)
                count = 3 if is_wknd else 2
                for slot in (ShiftSlot.MORNING, ShiftSlot.MIDDAY, ShiftSlot.EVENING):
                    optimizer_demands.append(OptimizerShiftDemand(
                        weekday=wd, slot=slot, position=pos, required_count=count,
                    ))

    optimizer = ShiftOptimizer(time_limit_seconds=req.time_limit_seconds)
    result = optimizer.optimize(optimizer_emps, optimizer_demands, req.week_start, history_weekend)

    if result.success and req.write_to_db:
        db.query(DBShift).filter(
            DBShift.week_start == req.week_start,
            (DBShift.store_id == req.store_id) if req.store_id else True,
        ).delete(synchronize_session=False)
        for s in result.shifts:
            emp = db.query(DBEmployee).filter(DBEmployee.id == s.employee_id).first()
            store = emp.store_id if emp else req.store_id
            db_shift = DBShift(
                employee_id=s.employee_id, date=s.date,
                start_time=s.start_time, end_time=s.end_time,
                slot=s.slot.value, position=s.position,
                hours=s.hours, cost=s.cost, week_start=req.week_start,
                store_id=store,
            )
            db.add(db_shift)
        db.commit()

    return result


class ShiftResponse(BaseModel):
    id: Optional[int] = None
    employee_id: str
    employee_name: str
    date: date
    start_time: time
    end_time: time
    slot: ShiftSlot
    position: str
    hours: float
    cost: float
    week_start: date
    store_id: Optional[str] = None


@app.get("/api/schedule/shifts", response_model=List[ShiftResponse])
def list_shifts(
    week_start: Optional[date] = None,
    store_id: Optional[str] = None,
    employee_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(DBShift, DBEmployee.name).join(DBEmployee, DBShift.employee_id == DBEmployee.id)
    if week_start:
        q = q.filter(DBShift.week_start == week_start)
    if store_id:
        q = q.filter(DBShift.store_id == store_id)
    if employee_id:
        q = q.filter(DBShift.employee_id == employee_id)
    rows = q.order_by(DBShift.date, DBShift.employee_id).all()
    result = []
    for s, name in rows:
        result.append(ShiftResponse(
            id=s.id, employee_id=s.employee_id, employee_name=name,
            date=s.date, start_time=s.start_time, end_time=s.end_time,
            slot=ShiftSlot(s.slot), position=s.position,
            hours=s.hours, cost=s.cost, week_start=s.week_start,
            store_id=s.store_id,
        ))
    return result


@app.get("/api/schedule/validate")
def validate_schedule(
    week_start: Optional[date] = Query(None),
    store_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(DBShift, DBEmployee.name, DBEmployee).join(DBEmployee, DBShift.employee_id == DBEmployee.id)
    if week_start:
        q = q.filter(DBShift.week_start == week_start)
    if store_id:
        q = q.filter(DBShift.store_id == store_id)
    rows = q.all()
    if not rows:
        return JSONResponse({
            "is_compliant": True,
            "total_shifts": 0,
            "total_violations": 0,
            "message": "无排班数据",
        })

    shifts = []
    emps_map: Dict[str, DBEmployee] = {}
    for s, name, db_emp in rows:
        shifts.append(ScheduledShift(
            employee_id=s.employee_id, employee_name=name,
            date=s.date, start_time=s.start_time, end_time=s.end_time,
            slot=ShiftSlot(s.slot), position=s.position,
            hours=s.hours, cost=s.cost,
        ))
        emps_map[s.employee_id] = db_emp

    optimizer_emps = [_to_optimizer_employee(e) for e in emps_map.values()]
    validator = ShiftValidator(optimizer_emps)
    report = validator.validate(shifts, week_start=week_start)
    return JSONResponse(json.loads(report.model_dump_json()))


@app.get("/api/schedule/validate/report.pdf")
def download_validation_report_pdf(
    week_start: Optional[date] = Query(None),
    store_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(DBShift, DBEmployee.name, DBEmployee).join(DBEmployee, DBShift.employee_id == DBEmployee.id)
    if week_start:
        q = q.filter(DBShift.week_start == week_start)
    if store_id:
        q = q.filter(DBShift.store_id == store_id)
    rows = q.all()
    shifts = []
    emps_map: Dict[str, DBEmployee] = {}
    for s, name, db_emp in rows:
        shifts.append(ScheduledShift(
            employee_id=s.employee_id, employee_name=name,
            date=s.date, start_time=s.start_time, end_time=s.end_time,
            slot=ShiftSlot(s.slot), position=s.position,
            hours=s.hours, cost=s.cost,
        ))
        emps_map[s.employee_id] = db_emp

    optimizer_emps = [_to_optimizer_employee(e) for e in emps_map.values()]
    validator = ShiftValidator(optimizer_emps)
    report = validator.validate(shifts, week_start=week_start)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    ok = validator.export_pdf(report, tmp.name)
    if not ok:
        raise HTTPException(status_code=500, detail="PDF生成失败，请安装reportlab")

    filename = f"合规报告_{week_start or '全周'}_{store_id or 'all'}.pdf"
    return FileResponse(tmp.name, media_type="application/pdf", filename=filename)


class FeedbackCreate(BaseModel):
    employee_id: str
    week_start: date
    rating: FeedbackRating
    comment: Optional[str] = None


@app.post("/api/feedback", status_code=201)
def submit_feedback(fb: FeedbackCreate, db: Session = Depends(get_db)):
    emp = db.query(DBEmployee).filter(DBEmployee.id == fb.employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="员工不存在")
    db_fb = DBFeedback(
        employee_id=fb.employee_id, week_start=fb.week_start,
        rating=fb.rating.value, comment=fb.comment,
    )
    db.add(db_fb)
    if fb.rating == FeedbackRating.SATISFIED:
        emp.feedback_weight = max(0.5, emp.feedback_weight * 0.95)
    elif fb.rating == FeedbackRating.DISSATISFIED:
        emp.feedback_weight = min(2.0, emp.feedback_weight * 1.15)
    db.commit()
    return {"success": True, "new_weight": emp.feedback_weight}


@app.get("/api/reports/cost")
def cost_report(
    period: str = Query("week", pattern="^(week|month)$"),
    start_date: date = Query(...),
    store_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    if period == "week":
        end_date = start_date + timedelta(weeks=1)
        group_expr = func.date(DBShift.week_start)
    else:
        end_date = start_date + timedelta(days=30)
        group_expr = func.strftime("%Y-%m", DBShift.date)

    q = db.query(DBShift).filter(
        DBShift.date >= start_date, DBShift.date < end_date,
    )
    if store_id:
        q = q.filter(DBShift.store_id == store_id)
    shifts = q.all()
    if not shifts:
        return {"period": period, "start_date": start_date, "end_date": end_date, "data": [], "summary": {"total_cost": 0, "total_hours": 0, "total_shifts": 0, "compliance_rate": 1.0}}

    total_cost = sum(s.cost for s in shifts)
    total_hours = sum(s.hours for s in shifts)
    overtime_hours = sum(max(0, s.hours - 8) for s in shifts)

    per_store: Dict[str, dict] = {}
    per_emp: Dict[str, dict] = {}
    for s in shifts:
        key = s.store_id or "unknown"
        if key not in per_store:
            per_store[key] = {"store_id": key, "cost": 0, "hours": 0, "shifts": 0, "overtime": 0}
        per_store[key]["cost"] += s.cost
        per_store[key]["hours"] += s.hours
        per_store[key]["shifts"] += 1
        per_store[key]["overtime"] += max(0, s.hours - 8)

        ek = s.employee_id
        if ek not in per_emp:
            per_emp[ek] = {"cost": 0, "hours": 0, "shifts": 0}
        per_emp[ek]["cost"] += s.cost
        per_emp[ek]["hours"] += s.hours
        per_emp[ek]["shifts"] += 1

    emps_map = {e.id: e for e in db.query(DBEmployee).all()}
    scheduled_objs = []
    for s in shifts:
        e = emps_map.get(s.employee_id)
        scheduled_objs.append(ScheduledShift(
            employee_id=s.employee_id, employee_name=e.name if e else s.employee_id,
            date=s.date, start_time=s.start_time, end_time=s.end_time,
            slot=ShiftSlot(s.slot), position=s.position, hours=s.hours, cost=s.cost,
        ))
    opt_emps = [_to_optimizer_employee(e) for e in emps_map.values()]
    validator = ShiftValidator(opt_emps)
    week_st = None
    if period == "week":
        week_st = start_date
    vr = validator.validate(scheduled_objs, week_start=week_st)
    compliance_rate = 1.0 - (vr.critical_count / max(1, len(shifts)))

    return {
        "period": period,
        "start_date": start_date,
        "end_date": end_date,
        "summary": {
            "total_cost": round(total_cost, 2),
            "total_hours": round(total_hours, 2),
            "overtime_hours": round(overtime_hours, 2),
            "total_shifts": len(shifts),
            "employee_count": len(per_emp),
            "compliance_rate": round(compliance_rate, 4),
            "violations": vr.critical_count + vr.warning_count + vr.info_count,
            "critical_violations": vr.critical_count,
        },
        "by_store": list(per_store.values()),
    }


@app.get("/api/reports/cost/export")
def export_cost_report(
    period: str = Query("week", pattern="^(week|month)$"),
    start_date: date = Query(...),
    store_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    data = cost_report(period=period, start_date=start_date, store_id=store_id, db=db)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_df = pd.DataFrame([data["summary"]])
        summary_df.to_excel(writer, sheet_name="汇总", index=False)
        if data["by_store"]:
            store_df = pd.DataFrame(data["by_store"])
            store_df.to_excel(writer, sheet_name="门店明细", index=False)

        q = db.query(DBShift, DBEmployee.name).join(DBEmployee, DBShift.employee_id == DBEmployee.id)
        end_date = start_date + (timedelta(weeks=1) if period == "week" else timedelta(days=30))
        q = q.filter(DBShift.date >= start_date, DBShift.date < end_date)
        if store_id:
            q = q.filter(DBShift.store_id == store_id)
        rows = q.all()
        if rows:
            shift_rows = []
            for s, name in rows:
                shift_rows.append({
                    "日期": s.date,
                    "员工ID": s.employee_id,
                    "员工姓名": name,
                    "岗位": s.position,
                    "时段": s.slot,
                    "开始": s.start_time,
                    "结束": s.end_time,
                    "工时": s.hours,
                    "成本": s.cost,
                    "门店ID": s.store_id,
                })
            pd.DataFrame(shift_rows).to_excel(writer, sheet_name="排班明细", index=False)
    buf.seek(0)
    filename = f"成本报告_{period}_{start_date}.xlsx"
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"}
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
