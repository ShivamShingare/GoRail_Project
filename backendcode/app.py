# app.py
"""
Single-file FastAPI backend for Railway Booking (PostgreSQL + SQLAlchemy).

Requirements (pip install):
    fastapi uvicorn sqlalchemy psycopg2-binary passlib[bcrypt] pydantic python-dotenv

Run (development):
    export DATABASE_URL="postgresql://user:pass@localhost:5432/railway_db"
    uvicorn app:app --reload --port 8000
"""

import os
from datetime import datetime
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from fastapi import FastAPI, HTTPException, Depends

from pydantic import BaseModel, EmailStr, conint
from passlib.context import CryptContext

# -------- Configuration --------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://gorail_db_adb0_user:mn2El92se6WArmRAIFjoM2RpB9FSH0Xy@dpg-d42qb315pdvs73dbdu20-a.oregon-postgres.render.com/gorail_db_adb0?sslmode=require"
)


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -------- SQLAlchemy setup --------
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# -------- Models --------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(200), nullable=False)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(200), unique=True, nullable=False, index=True)
    password_hash = Column(String(200), nullable=False)
    phone = Column(String(32), nullable=True)
    dob = Column(String(32), nullable=True)
    gender = Column(String(32), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    bookings = relationship("Booking", back_populates="user")


class Train(Base):
    __tablename__ = "trains"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    from_station = Column(String(100), nullable=False)
    to_station = Column(String(100), nullable=False)
    time = Column(String(10), nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    price = Column(Float, nullable=False)
    train_type = Column(String(50), nullable=True)
    available = Column(Boolean, default=True)
    info = Column(Text, nullable=True)

    bookings = relationship("Booking", back_populates="train")


class Booking(Base):
    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    train_id = Column(Integer, ForeignKey("trains.id"), nullable=False)

    passenger_name = Column(String(200), nullable=False)
    seats = Column(Integer, nullable=False, default=1)

    travel_date = Column(DateTime, nullable=False)
    travel_time = Column(String(10), nullable=False)
    booked_price = Column(Float, nullable=False)
    total_price = Column(Float, nullable=False)

    payment_id = Column(String(100), nullable=True)   # ✅ NEW COLUMN

    booked_at = Column(DateTime, server_default=func.now())
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="bookings")
    train = relationship("Train", back_populates="bookings")


# Create tables
Base.metadata.create_all(bind=engine)

# -------- Pydantic Schemas --------
class RegisterIn(BaseModel):
    full_name: str
    username: str
    email: EmailStr
    password: str
    phone: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    full_name: str
    username: str
    email: EmailStr

    class Config:
        from_attributes = True


class TrainOut(BaseModel):
    id: int
    name: str
    from_station: str
    to_station: str
    time: str
    date: datetime
    price: float
    train_type: Optional[str]
    available: bool
    info: Optional[str]

    class Config:
        from_attributes = True


class TrainIn(BaseModel):
    name: str
    from_station: str
    to_station: str
    time: str
    date: Optional[datetime] = None
    price: float
    train_type: Optional[str] = None
    available: bool = True
    info: Optional[str] = None


from pydantic import BaseModel
from typing import Optional

class BookIn(BaseModel):
    user_id: int
    train_id: int
    passenger_name: str
    seats: int
    notes: Optional[str] = None
    payment_id: Optional[str] = None  # ✅ accept payment id


class BookingOut(BaseModel):
    id: int
    user_id: int
    train_id: int
    passenger_name: str
    seats: int
    travel_date: datetime
    travel_time: str
    booked_price: float
    total_price: float
    payment_id: Optional[str]
    booked_at: datetime
    notes: Optional[str]

    class Config:
        from_attributes = True

# -------- Utilities --------
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------- App & CORS --------
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your React frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # allow custom headers
)
# -------- Root endpoint for Render health check --------
@app.get("/")
def root():
    return {"status": "ok", "message": "🚆 GoRail FastAPI backend is running!"}

# -------- Seed demo trains --------
def parse_price(price_str):
    return float(price_str.replace("₹", "").replace(",", "").strip())

def seed_demo_trains():
    db = SessionLocal()
    try:
        existing_names = {t.name for t in db.query(Train.name).all()}
        from datetime import datetime
        from datetime import date
        from datetime import datetime
        from datetime import datetime

        trains = [
            {"name": "Tokyo Local 101", "from_station": "Tokyo", "to_station": "Shinjuku", "time": "06:15 AM", "date": datetime(2025, 9, 10), "price": 120, "train_type": "Local", "available": True, "info": "Fast local service in central Tokyo"},
            {"name": "Osaka Rapid 202", "from_station": "Osaka", "to_station": "Kobe", "time": "07:00 AM", "date": datetime(2025, 9, 10), "price": 200, "train_type": "Rapid", "available": True, "info": "Popular rapid commuter train"},
            {"name": "Kyoto Express 303", "from_station": "Kyoto", "to_station": "Nara", "time": "07:45 AM", "date": datetime(2025, 9, 10), "price": 180, "train_type": "Express", "available": True, "info": "Connects cultural cities"},
            {"name": "Sapporo Local 404", "from_station": "Sapporo", "to_station": "Otaru", "time": "08:10 AM", "date": datetime(2025, 9, 10), "price": 150, "train_type": "Local", "available": True, "info": "Short scenic ride"},
            {"name": "Nagoya Express 505", "from_station": "Nagoya", "to_station": "Gifu", "time": "08:40 AM", "date": datetime(2025, 9, 10), "price": 220, "train_type": "Express", "available": True, "info": "Business commuters train"},
            {"name": "Hiroshima Local 606", "from_station": "Hiroshima", "to_station": "Miyajima", "time": "09:00 AM", "date": datetime(2025, 9, 10), "price": 140, "train_type": "Local", "available": True, "info": "Tourist-friendly route"},
            {"name": "Sendai Rapid 707", "from_station": "Sendai", "to_station": "Matsushima", "time": "09:20 AM", "date": datetime(2025, 9, 10), "price": 250, "train_type": "Rapid", "available": True, "info": "Scenic coastal line"},
            {"name": "Tokyo Express 808", "from_station": "Tokyo", "to_station": "Yokohama", "time": "09:45 AM", "date": datetime(2025, 9, 10), "price": 300, "train_type": "Express", "available": True, "info": "Most used commuter express"},
            {"name": "Osaka Local 909", "from_station": "Osaka", "to_station": "Nara", "time": "10:10 AM", "date": datetime(2025, 9, 10), "price": 170, "train_type": "Local", "available": True, "info": "Budget-friendly local service"},
            {"name": "Kyoto Rapid 111", "from_station": "Kyoto", "to_station": "Osaka", "time": "10:40 AM", "date": datetime(2025, 9, 10), "price": 280, "train_type": "Rapid", "available": True, "info": "Common intercity rapid"},
            {"name": "Nagoya Local 212", "from_station": "Nagoya", "to_station": "Toyota", "time": "11:00 AM", "date": datetime(2025, 9, 10), "price": 130, "train_type": "Local", "available": True, "info": "Short commuter route"},
            {"name": "Hokkaido Express 313", "from_station": "Sapporo", "to_station": "Asahikawa", "time": "11:30 AM", "date": datetime(2025, 9, 10), "price": 320, "train_type": "Express", "available": True, "info": "Northern express line"},
            {"name": "Tokyo Local 414", "from_station": "Tokyo", "to_station": "Chiba", "time": "12:00 PM", "date": datetime(2025, 9, 10), "price": 190, "train_type": "Local", "available": True, "info": "Daily work commute train"},
            {"name": "Osaka Express 515", "from_station": "Osaka", "to_station": "Kyoto", "time": "12:30 PM", "date": datetime(2025, 9, 10), "price": 260, "train_type": "Express", "available": True, "info": "Very frequent service"},
            {"name": "Nagoya Rapid 616", "from_station": "Nagoya", "to_station": "Shizuoka", "time": "01:00 PM", "date": datetime(2025, 9, 10), "price": 400, "train_type": "Rapid", "available": True, "info": "Mid-range rapid"},
            {"name": "Kyoto Local 717", "from_station": "Kyoto", "to_station": "Uji", "time": "01:30 PM", "date": datetime(2025, 9, 10), "price": 110, "train_type": "Local", "available": True, "info": "Tea town route"},
            {"name": "Sendai Local 818", "from_station": "Sendai", "to_station": "Furukawa", "time": "02:00 PM", "date": datetime(2025, 9, 10), "price": 160, "train_type": "Local", "available": True, "info": "Short rural line"},
            {"name": "Hiroshima Express 919", "from_station": "Hiroshima", "to_station": "Okayama", "time": "02:30 PM", "date": datetime(2025, 9, 10), "price": 350, "train_type": "Express", "available": True, "info": "Regional connector"},
            {"name": "Tokyo Rapid 121", "from_station": "Tokyo", "to_station": "Kawasaki", "time": "03:00 PM", "date": datetime(2025, 9, 10), "price": 210, "train_type": "Rapid", "available": True, "info": "Busy commuter rapid"},
            {"name": "Osaka Local 222", "from_station": "Osaka", "to_station": "Sakai", "time": "03:30 PM", "date": datetime(2025, 9, 10), "price": 100, "train_type": "Local", "available": True, "info": "Very cheap local travel"},
        ]

        for t in trains:
            if t["name"] not in existing_names:  # only add if not already in DB
                db.add(Train(**t))
        db.commit()
    finally:
        db.close()


seed_demo_trains()


# -------- Endpoints --------
@app.post("/api/register", response_model=UserOut)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email.lower()).first():
        raise HTTPException(status_code=400, detail="Email already registered.")
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="Username already taken.")

    user = User(
        full_name=payload.full_name.strip(),
        username=payload.username.strip(),
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        phone=payload.phone,
        dob=payload.dob,
        gender=payload.gender,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/api/login", response_model=UserOut)
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return user

@app.get("/api/trains", response_model=List[TrainOut])
def get_trains(q: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Train)
    if q:
        pattern = f"%{q.lower()}%"
        query = query.filter(
            (func.lower(Train.name).like(pattern))
            | (func.lower(Train.from_station).like(pattern))
            | (func.lower(Train.to_station).like(pattern))
        )
    return query.order_by(Train.id).all()

@app.post("/api/trains", response_model=TrainOut)
def add_train(payload: TrainIn, db: Session = Depends(get_db)):
    train = Train(**payload.dict())
    db.add(train)
    db.commit()
    db.refresh(train)
    return train

@app.get("/api/trains/{train_id}", response_model=TrainOut)
def get_train(train_id: int, db: Session = Depends(get_db)):
    t = db.query(Train).filter(Train.id == train_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Train not found.")
    return t

@app.post("/api/book", response_model=BookingOut)
def create_booking(payload: BookIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    train = db.query(Train).filter(Train.id == payload.train_id).first()
    if not train:
        raise HTTPException(status_code=404, detail="Train not found.")

    total = float(train.price) * int(payload.seats)

    booking = Booking(
        user_id=user.id,
        train_id=train.id,
        passenger_name=payload.passenger_name.strip(),
        seats=int(payload.seats),
        travel_date=train.date,
        travel_time=train.time,
        booked_price=train.price,
        total_price=total,
        payment_id=payload.payment_id,
        notes=payload.notes,
    )

    try:
        db.add(booking)
        db.commit()
        db.refresh(booking)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")

    return booking


@app.get("/api/bookings/{user_id}", response_model=List[BookingOut])
def get_user_bookings(user_id: int, db: Session = Depends(get_db)):
    if not db.query(User).filter(User.id == user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    return db.query(Booking).filter(Booking.user_id == user_id).order_by(Booking.booked_at.desc()).all()
