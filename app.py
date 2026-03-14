import os
import re
from dotenv import load_dotenv

load_dotenv()
import csv
import io
import secrets
import hashlib
import warnings
import socket
from datetime import datetime, timedelta, timezone

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from flask_jwt_extended import JWTManager, create_access_token
from flask_cors import CORS

import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sqlalchemy import text, func

import pymysql
pymysql.install_as_MySQLdb()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def as_naive_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


app = Flask(__name__)
CORS(app)

LOG_FILE = "powerpulse_debug.log"


def dbg(*parts):
    msg = " ".join(str(p) for p in parts)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    print(msg, flush=True)


MYSQL_DB_URL = os.getenv(
    "DB_URL",
    "mysql+pymysql://root:@localhost:3306/powerpulse_backend"
)

DB_FALLBACK_SQLITE = (os.getenv("DB_FALLBACK_SQLITE", "1").strip() == "1")
SQLITE_FALLBACK_URL = os.getenv("SQLITE_FALLBACK_URL", "sqlite:///powerpulse_fallback.db")

app.config["SQLALCHEMY_DATABASE_URI"] = MYSQL_DB_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 280,
    "pool_timeout": 30,
    "connect_args": {
        "connect_timeout": 20,
        "read_timeout": 20,
        "write_timeout": 20,
        "charset": "utf8mb4"
    }
}

app.config["JWT_SECRET_KEY"] = os.getenv(
    "JWT_SECRET_KEY",
    "powerpulse_super_secret_key_2026_secure_12345"
)


def getenv_str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


app.config["MAIL_SERVER"] = getenv_str("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(getenv_str("MAIL_PORT", "587") or "587")
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_USERNAME"] = getenv_str("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = getenv_str("MAIL_PASSWORD")

default_sender_env = getenv_str("MAIL_DEFAULT_SENDER")
if default_sender_env:
    app.config["MAIL_DEFAULT_SENDER"] = default_sender_env
elif app.config["MAIL_USERNAME"]:
    app.config["MAIL_DEFAULT_SENDER"] = f"PowerPulse <{app.config['MAIL_USERNAME']}>"
else:
    app.config["MAIL_DEFAULT_SENDER"] = ""

app.config["MAIL_TIMEOUT"] = 20
socket.setdefaulttimeout(20)

DEV_OTP_FALLBACK = (os.getenv("DEV_OTP_FALLBACK", "1").strip() == "1")

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
mail = Mail(app)
jwt = JWTManager(app)

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

SEQ_LEN = 30

MODEL_PATH = os.getenv("MODEL_PATH", "energy_lstm_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.save")
SARIMA_MODEL_PATH = os.getenv("SARIMA_MODEL_PATH", "energy_sarima_model.pkl")

model = None
scaler = None
sarima_model_fit = None


def load_energy_artifacts():
    global model, scaler

    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model = load_model(MODEL_PATH, compile=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        if scaler is None:
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
            scaler = joblib.load(SCALER_PATH)

        for warn in w:
            msg = str(warn.message)
            if "InconsistentVersionWarning" in msg or "Trying to unpickle estimator" in msg:
                dbg("⚠️ scikit-learn version mismatch for scaler.save")
                dbg("Fix: install same scikit-learn version used during training or recreate scaler.save")
                break


def load_sarima_artifact():
    global sarima_model_fit
    if sarima_model_fit is None:
        if not os.path.exists(SARIMA_MODEL_PATH):
            raise FileNotFoundError(f"SARIMA model file not found: {SARIMA_MODEL_PATH}")
        sarima_model_fit = joblib.load(SARIMA_MODEL_PATH)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    consumer_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False, index=True)
    mandal = db.Column(db.String(120), nullable=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)


class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    org_name = db.Column(db.String(180), nullable=False, index=True)
    board_id = db.Column(db.String(80), nullable=False, index=True)
    email = db.Column(db.String(180), nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)

    __table_args__ = (
        db.UniqueConstraint("org_name", "email", name="uniq_admin_org_email"),
        db.UniqueConstraint("org_name", "board_id", name="uniq_admin_org_boardid"),
    )


class PasswordResetOTP(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(180), nullable=False, index=True)
    org_name = db.Column(db.String(180), nullable=True, index=True)
    otp_hash = db.Column(db.String(255), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    verified_at = db.Column(db.DateTime, nullable=True)
    reset_token_hash = db.Column(db.String(255), nullable=True)
    reset_expires_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow)


class Consumer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    consumer_no = db.Column(db.String(50), unique=True, nullable=False, index=True)
    consumer_name = db.Column(db.String(120), nullable=False)
    mandal = db.Column(db.String(120), nullable=False, index=True)
    sector_type = db.Column(db.String(50), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=utcnow)


class BoardUsageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    consumer_id = db.Column(db.Integer, db.ForeignKey("consumer.id"), nullable=False, index=True)
    consumer = db.relationship("Consumer", backref=db.backref("usage_records", lazy=True))
    reading_month = db.Column(db.Integer, nullable=False, index=True)
    reading_year = db.Column(db.Integer, nullable=False, index=True)
    previous_reading = db.Column(db.Float, nullable=False)
    current_reading = db.Column(db.Float, nullable=False)
    units_used = db.Column(db.Float, nullable=False)
    source_file = db.Column(db.String(255), nullable=True)
    imported_at = db.Column(db.DateTime, default=utcnow)

    __table_args__ = (
        db.UniqueConstraint(
            "consumer_id", "reading_month", "reading_year",
            name="uniq_consumer_month_year_usage"
        ),
    )


class LiveMandalUsage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mandal = db.Column(db.String(120), nullable=False, index=True)
    reading_time = db.Column(db.DateTime, nullable=False, index=True)
    current_month_usage = db.Column(db.Float, nullable=False, default=0.0)
    today_usage = db.Column(db.Float, nullable=False, default=0.0)
    current_load = db.Column(db.Float, nullable=False, default=0.0)
    source = db.Column(db.String(50), nullable=False, default="board_sync")


class SystemSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    one_unit_cost = db.Column(db.Float, nullable=False, default=10.0)
    currency = db.Column(db.String(10), nullable=False, default="INR")
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)


class ActiveRecharge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    consumer_no = db.Column(db.String(50), nullable=False, index=True)
    plan_id = db.Column(db.Integer, nullable=False)
    plan_name = db.Column(db.String(100))
    plan_price = db.Column(db.Float)
    total_units = db.Column(db.Float)
    used_units = db.Column(db.Float, default=0.0)
    remaining_units = db.Column(db.Float)
    validity_days = db.Column(db.Integer)
    status = db.Column(db.String(20), default="ACTIVE")
    recharge_date = db.Column(db.DateTime, default=utcnow)
    expiry_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)


class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    consumer_no = db.Column(db.String(50), nullable=False, index=True)
    alert_type = db.Column(db.String(50))  # LOW_BALANCE, EXHAUSTED
    title = db.Column(db.String(100))
    message = db.Column(db.Text)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=utcnow)


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    consumer_no = db.Column(db.String(50), nullable=False, index=True)
    sender_role = db.Column(db.String(20), nullable=False)  # 'ADMIN' or 'USER'
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=utcnow)


class DailyUsageInput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    consumer_no = db.Column(db.String(50), nullable=False, index=True)
    usage_date = db.Column(db.Date, nullable=False, index=True)
    usage_kwh = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)

    __table_args__ = (
        db.UniqueConstraint("consumer_no", "usage_date", name="uniq_consumer_daily_usage"),
    )


class RechargePlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plan_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    validity_days = db.Column(db.Integer, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    units = db.Column(db.Float, nullable=False)
    rate_per_unit = db.Column(db.Float, nullable=True)  # Calculated or set manually
    tag = db.Column(db.String(50), nullable=True)        # e.g. BEST VALUE, RECOMMENDED
    is_recommended = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    display_order = db.Column(db.Integer, default=0)
    # Keep legacy status column for backward compat
    status = db.Column(db.String(20), default="ACTIVE")
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)


class UsagePredictPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plan_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    validity_days = db.Column(db.Integer, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    units = db.Column(db.Float, nullable=False)
    tag = db.Column(db.String(50), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    display_order = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)


class RechargeOrder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_ref = db.Column(db.String(50), unique=True, nullable=False, index=True)
    consumer_no = db.Column(db.String(50), nullable=False, index=True)
    plan_id = db.Column(db.Integer, nullable=True)
    # Snapshot fields — not affected by later plan edits
    plan_name_snapshot = db.Column(db.String(100))
    units_snapshot = db.Column(db.Float)
    validity_days_snapshot = db.Column(db.Integer)
    base_amount = db.Column(db.Float, nullable=False)
    tax_amount = db.Column(db.Float, default=0.0)
    total_amount = db.Column(db.Float, nullable=False)
    payment_method = db.Column(db.String(20), default="UPI")
    payment_status = db.Column(db.String(20), default="PENDING")  # PENDING/SUCCESS/FAILED/CANCELLED
    transaction_ref = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)


def ok(data=None, status=200):
    payload = {"ok": True}
    if data is not None:
        payload.update(data)
    return jsonify(payload), status


def fail(message, status=400):
    return jsonify({"ok": False, "error": message}), status


def valid_email(email: str) -> bool:
    return bool(email and EMAIL_REGEX.match(email))


def hash_password(pw: str) -> str:
    return bcrypt.generate_password_hash(pw).decode("utf-8")


def check_password(pw: str, pw_hash: str) -> bool:
    return bcrypt.check_password_hash(pw_hash, pw)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def generate_otp() -> str:
    return f"{secrets.randbelow(10**6):06d}"


def email_is_configured() -> bool:
    return bool(
        app.config.get("MAIL_USERNAME")
        and app.config.get("MAIL_PASSWORD")
        and app.config.get("MAIL_DEFAULT_SENDER")
    )


def find_account(role: str, email: str, org_name: str | None):
    if role == "user":
        return User.query.filter_by(email=email).first()
    if role == "admin":
        if not org_name:
            return None
        return Admin.query.filter_by(email=email, org_name=org_name).first()
    return None


def generate_board_id(org_name: str) -> str:
    for _ in range(50):
        part1 = f"{secrets.randbelow(1000):03d}"
        part2 = f"{secrets.randbelow(10000):04d}"
        board_id = f"UB-{part1}-{part2}"
        exists = Admin.query.filter_by(org_name=org_name, board_id=board_id).first()
        if not exists:
            return board_id
    raise RuntimeError("Could not generate unique board ID. Try again.")


def generate_consumer_id() -> str:
    last_user = User.query.filter(User.consumer_id.like("CN%")) \
        .order_by(User.id.desc()) \
        .first()

    if not last_user or not last_user.consumer_id:
        return "CN1001"

    try:
        last_num = int(last_user.consumer_id[2:])
        print(last_num)
    except ValueError:
        return "CN1001"

    if last_num < 1000:
        last_num = 1000

    return f"CN{last_num + 1}"


def parse_int(value, field_name):
    try:
        return int(str(value).strip())
    except Exception:
        raise ValueError(f"Invalid integer for {field_name}")


def parse_float(value, field_name):
    try:
        return float(str(value).strip())
    except Exception:
        raise ValueError(f"Invalid number for {field_name}")


def normalize_text(value: str) -> str:
    return (value or "").strip()


def upsert_consumer(consumer_no: str, consumer_name: str, mandal: str, sector_type: str):
    consumer = Consumer.query.filter_by(consumer_no=consumer_no).first()

    if consumer:
        consumer.consumer_name = consumer_name
        consumer.mandal = mandal
        consumer.sector_type = sector_type
        return consumer

    consumer = Consumer(
        consumer_no=consumer_no,
        consumer_name=consumer_name,
        mandal=mandal,
        sector_type=sector_type
    )
    db.session.add(consumer)
    db.session.flush()
    return consumer


def parse_usage_date(date_str: str):
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except Exception:
        raise ValueError("usageDate must be in YYYY-MM-DD format")


def get_last_30_db_values(consumer_no: str):
    rows = DailyUsageInput.query.filter_by(consumer_no=consumer_no).order_by(
        DailyUsageInput.usage_date.asc()
    ).all()

    if not rows:
        raise ValueError(f"No daily usage records found for {consumer_no}")

    values = [float(r.usage_kwh) for r in rows]

    if len(values) >= SEQ_LEN:
        return values[-SEQ_LEN:], rows[-SEQ_LEN:]

    # pad if fewer than 30 values
    last_value = values[-1]
    padded = values[:] + [last_value] * (SEQ_LEN - len(values))
    return padded, rows


def send_email_otp_sync(to_email: str, otp: str, role: str):
    subject = "Password Reset Verification Code"
    body = (
        f"Your {role.upper()} password reset verification code is: {otp}\n\n"
        f"This code will expire in 10 minutes.\n"
        f"If you didn't request this, ignore this email."
    )

    msg = Message(
        subject=subject,
        recipients=[to_email],
        body=body,
        sender=app.config["MAIL_DEFAULT_SENDER"],
    )
    mail.send(msg)


def send_email_otp(to_email: str, otp: str, role: str):
    result = {"ok": False, "error": None, "fallback": False}

    try:
        dbg("EMAIL DEBUG START")
        dbg("TO =", to_email)
        dbg("ROLE =", role)
        dbg("MAIL_SERVER =", app.config.get("MAIL_SERVER"))
        dbg("MAIL_PORT =", app.config.get("MAIL_PORT"))
        dbg("MAIL_USERNAME =", app.config.get("MAIL_USERNAME"))
        dbg("MAIL_DEFAULT_SENDER =", app.config.get("MAIL_DEFAULT_SENDER"))
        dbg("EMAIL_CONFIGURED =", email_is_configured())

        if not email_is_configured():
            raise ValueError("Email not configured")

        send_email_otp_sync(to_email, otp, role)
        result["ok"] = True
        dbg("EMAIL SENT SUCCESSFULLY")
        dbg("EMAIL DEBUG END")
        return result

    except Exception as e:
        result["ok"] = False
        result["error"] = repr(e)
        dbg("EMAIL SEND FAILED")
        dbg("EMAIL ERROR =", repr(e))
        dbg("EMAIL DEBUG END")

        if DEV_OTP_FALLBACK:
            dbg("EMAIL FAILED, USING DEV FALLBACK")
            dbg("OTP =", otp)
            result["ok"] = True
            result["fallback"] = True
            return result

        return result


def try_db_connection() -> tuple[bool, str]:
    try:
        db.session.execute(text("SELECT 1"))
        db.session.commit()
        return True, "connected"
    except Exception as e:
        db.session.rollback()
        return False, str(e)


def init_db_or_fallback():
    ok_db, msg = try_db_connection()
    if ok_db:
        dbg("✅ DB connected:", app.config["SQLALCHEMY_DATABASE_URI"])
        return

    dbg("❌ DB connection failed!")
    dbg("DB URI:", app.config["SQLALCHEMY_DATABASE_URI"])
    dbg("Error:", msg)

    dbg("✅ Fix steps (XAMPP):")
    dbg("1) Open XAMPP Control Panel")
    dbg("2) Start MySQL")
    dbg("3) Ensure port is 3306")
    dbg("4) Then run: python app.py")

    if DB_FALLBACK_SQLITE:
        dbg("⚠️ Falling back to SQLite (dev mode).")
        app.config["SQLALCHEMY_DATABASE_URI"] = SQLITE_FALLBACK_URL

        try:
            db.session.remove()
        except Exception:
            pass

        try:
            db.engine.dispose()
        except Exception:
            pass

        try:
            db.session.execute(text("SELECT 1"))
            db.session.commit()
            dbg("✅ SQLite fallback connected.")
        except Exception as e:
            db.session.rollback()
            dbg("❌ SQLite fallback also failed:", e)

    # Seed Recharge Plans
    seed_recharge_plans()


def seed_recharge_plans():
    try:
        if RechargePlan.query.first():
            return

        plans = [
            RechargePlan(
                plan_name="PowerPack Pro", description="Perfect for your predicted usage",
                validity_days=30, amount=1500.0, units=200.0, rate_per_unit=7.5,
                tag="BEST VALUE", is_recommended=True, is_active=True, display_order=1, status="ACTIVE"
            ),
            RechargePlan(
                plan_name="Value Pack", description="Great for moderate users",
                validity_days=28, amount=1150.0, units=150.0, rate_per_unit=7.67,
                tag="RECOMMENDED", is_recommended=False, is_active=True, display_order=2, status="ACTIVE"
            ),
            RechargePlan(
                plan_name="Standard", description="The standard home choice",
                validity_days=28, amount=800.0, units=100.0, rate_per_unit=8.0,
                tag=None, is_recommended=False, is_active=True, display_order=3, status="ACTIVE"
            ),
            RechargePlan(
                plan_name="Mini Saver", description="Essential power for short term",
                validity_days=14, amount=450.0, units=50.0, rate_per_unit=9.0,
                tag=None, is_recommended=False, is_active=True, display_order=4, status="ACTIVE"
            ),
            RechargePlan(
                plan_name="Quick Top-up", description="Backup power for small needs",
                validity_days=7, amount=180.0, units=20.0, rate_per_unit=9.0,
                tag=None, is_recommended=False, is_active=True, display_order=5, status="ACTIVE"
            ),
        ]
        db.session.bulk_save_objects(plans)
        db.session.commit()
        dbg("✅ Seeded recharge plans")
    except Exception as e:
        db.session.rollback()
        dbg("❌ Failed to seed plans:", str(e))


@app.get("/")
def home():
    return ok({"message": "PowerPulse backend running"})


@app.get("/health")
def health():
    return ok({"status": "ok"})


@app.get("/db-check")
def db_check():
    try:
        db.session.execute(text("SELECT 1"))
        return ok({"db": "connected", "uri": app.config["SQLALCHEMY_DATABASE_URI"]})
    except Exception as e:
        return fail(f"db error: {str(e)}", 500)


@app.get("/mail-check")
def mail_check():
    dbg("MAIL CHECK ROUTE HIT")
    return ok({
        "mail_configured": email_is_configured(),
        "MAIL_SERVER": app.config.get("MAIL_SERVER"),
        "MAIL_PORT": app.config.get("MAIL_PORT"),
        "MAIL_USERNAME": app.config.get("MAIL_USERNAME"),
        "MAIL_DEFAULT_SENDER": app.config.get("MAIL_DEFAULT_SENDER"),
        "DEV_OTP_FALLBACK": DEV_OTP_FALLBACK
    })


@app.get("/test-mail")
def test_mail():
    try:
        dbg("TEST MAIL ROUTE HIT")
        dbg("MAIL_USERNAME =", app.config.get("MAIL_USERNAME"))
        dbg("MAIL_DEFAULT_SENDER =", app.config.get("MAIL_DEFAULT_SENDER"))
        dbg("EMAIL_CONFIGURED =", email_is_configured())

        if not email_is_configured():
            raise ValueError("Email not configured")

        msg = Message(
            subject="PowerPulse Test Mail",
            recipients=[app.config.get("MAIL_USERNAME")],
            body="This is a PowerPulse test mail.",
            sender=app.config["MAIL_DEFAULT_SENDER"],
        )
        mail.send(msg)

        dbg("TEST MAIL SENT OK")
        return jsonify({"ok": True, "message": "Test mail sent"})
    except Exception as e:
        dbg("TEST MAIL FAILED =", repr(e))
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/debug/user-consumer")
def debug_user_consumer():
    try:
        consumer_no = (request.args.get("consumerNo") or "").strip()
        if not consumer_no:
            return fail("consumerNo required")

        user = User.query.filter_by(consumer_id=consumer_no).first()
        if not user:
            return fail("User not found", 404)

        return ok({
            "userId": user.id,
            "consumerId": user.consumer_id,
            "fullName": user.full_name,
            "email": user.email,
            "mandal": user.mandal,
            "createdAt": user.created_at.isoformat() if user.created_at else None
        })
    except Exception as e:
        return fail(str(e), 500)


@app.get("/debug/daily-usage-table")
def debug_daily_usage_table():
    try:
        db.session.execute(text("SELECT 1 FROM daily_usage_input LIMIT 1"))
        return ok({"message": "daily_usage_input table exists"})
    except Exception as e:
        return fail(str(e), 500)


@app.post("/auth/user/register")
def user_register():
    data = request.get_json(silent=True) or {}
    full_name = (data.get("fullName") or "").strip()
    email = (data.get("email") or "").strip().lower()
    mandal = (data.get("mandal") or "").strip()
    password = data.get("password") or ""
    confirm = data.get("confirmPassword") or ""

    if not full_name:
        return fail("Full name required")
    if not valid_email(email):
        return fail("Invalid email")
    if len(password) < 6:
        return fail("Password must be at least 6 characters")
    if password != confirm:
        return fail("Passwords do not match")
    if User.query.filter_by(email=email).first():
        return fail("User already exists", 409)

    consumer_id = generate_consumer_id()

    u = User(
        consumer_id=consumer_id,
        full_name=full_name,
        email=email,
        mandal=mandal if mandal else None,
        password_hash=hash_password(password)
    )
    db.session.add(u)
    db.session.commit()

    return ok({
        "message": "User registered",
        "consumerId": consumer_id,
        "mandal": u.mandal
    })


@app.post("/auth/user/login")
def user_login():
    try:
        data = request.get_json(silent=True) or {}
        print("LOGIN raw json =", data, flush=True)

        email = str(
            data.get("email")
            or data.get("userEmail")
            or ""
        ).strip().lower()

        password = str(
            data.get("password")
            or data.get("userPassword")
            or ""
        ).strip()

        print("LOGIN email =", email, flush=True)
        print("LOGIN password empty =", password == "", flush=True)

        if not email or not password:
            return jsonify({
                "ok": False,
                "error": "Email and password are required"
            }), 400

        user = User.query.filter(func.lower(User.email) == email).first()
        print("LOGIN user found =", user is not None, flush=True)

        if not user:
            return jsonify({
                "ok": False,
                "error": "User not found"
            }), 404

        if not bcrypt.check_password_hash(user.password_hash, password):
            return jsonify({
                "ok": False,
                "error": "Invalid password"
            }), 401

        access_token = create_access_token(identity=user.email)

        return jsonify({
            "ok": True,
            "message": "Login successful",
            "accessToken": access_token,
            "email": user.email,
            "fullName": user.full_name,
            "consumerNo": user.consumer_id,
            "role": "user"
        }), 200

    except Exception as e:
        print("LOGIN exception =", str(e), flush=True)
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.get("/auth/user-profile")
def get_user_profile():
    try:
        email = (request.args.get("email") or "").strip().lower()

        if not email:
            return fail("email required", 400)

        if not valid_email(email):
            return fail("Invalid email", 400)

        u = User.query.filter_by(email=email).first()

        if not u:
            return fail("User not found", 404)

        return ok({
            "user": {
                "id": u.id,
                "fullName": u.full_name,
                "email": u.email,
                "consumerNo": u.consumer_id,
                "mandal": u.mandal
            }
        })

    except Exception as e:
        return fail(str(e), 500)


@app.post("/auth/user-profile/update")
def update_user_profile():
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()
        full_name = (data.get("fullName") or "").strip()
        mandal = (data.get("mandal") or "").strip()

        if not email:
            return fail("email required", 400)

        u = User.query.filter_by(email=email).first()

        if not u:
            return fail("User not found", 404)

        if full_name:
            u.full_name = full_name
        if mandal:
            u.mandal = mandal

        db.session.commit()
        return ok({"message": "Profile updated successfully"})

    except Exception as e:
        db.session.rollback()
        return fail(str(e), 500)


@app.post("/auth/admin/register")
def admin_register():
    data = request.get_json(silent=True) or {}
    full_name = (data.get("fullName") or "").strip()
    org_name = (data.get("orgName") or "").strip()
    email = (data.get("email") or "").strip().lower()
    board_id = (data.get("boardId") or "").strip()
    password = data.get("password") or ""
    confirm = data.get("confirmPassword") or ""

    if not full_name:
        return fail("Full name required")
    if not org_name:
        return fail("Organization name required")
    if not valid_email(email):
        return fail("Invalid email")
    if len(password) < 6:
        return fail("Password must be at least 6 characters")
    if password != confirm:
        return fail("Passwords do not match")

    if Admin.query.filter_by(org_name=org_name, email=email).first():
        return fail("Admin already exists for this organization and email", 409)

    if not board_id:
        board_id = generate_board_id(org_name)

    if Admin.query.filter_by(org_name=org_name, board_id=board_id).first():
        return fail("Board ID already exists for this organization", 409)

    a = Admin(
        full_name=full_name,
        org_name=org_name,
        board_id=board_id,
        email=email,
        password_hash=hash_password(password),
    )
    db.session.add(a)
    db.session.commit()

    return ok({"message": "Admin created", "boardId": board_id})


@app.post("/auth/admin/login")
def admin_login():
    data = request.get_json(silent=True) or {}
    org_name = (data.get("orgName") or "").strip()
    board_id = (data.get("boardId") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not org_name:
        return fail("Organization name required")
    if not board_id:
        return fail("boardId required")
    if not valid_email(email) or not password:
        return fail("Email and password required")

    a = Admin.query.filter_by(org_name=org_name, board_id=board_id, email=email).first()
    if not a or not check_password(password, a.password_hash):
        return fail("Invalid credentials", 401)

    token = create_access_token(
        identity={"role": "admin", "adminId": a.id, "orgName": a.org_name}
    )
    return jsonify({
        "ok": True,
        "message": "Admin login successful",
        "accessToken": token,
        "email": a.email,
        "fullName": a.full_name,
        "boardId": a.board_id,
        "role": "admin"
    }), 200


@app.get("/auth/admin/profile")
def get_admin_profile():
    try:
        email = (request.args.get("email") or "").strip().lower()
        if not email:
            return fail("email required", 400)
            
        a = Admin.query.filter_by(email=email).first()
        if not a:
            return fail("Admin not found", 404)
            
        return ok({
            "admin": {
                "id": a.id,
                "fullName": a.full_name,
                "email": a.email,
                "boardId": a.board_id,
                "orgName": a.org_name,
                "createdAt": a.created_at.isoformat() if a.created_at else None
            }
        })
    except Exception as e:
        return fail(str(e), 500)


@app.post("/auth/forgot/request")
def forgot_request():
    data = request.get_json(silent=True) or {}
    role = (data.get("role") or "").strip().lower()
    email = (data.get("email") or "").strip().lower()
    org_name = (data.get("orgName") or "").strip() if role == "admin" else None

    dbg("FORGOT REQUEST START")
    dbg("role =", role)
    dbg("email =", email)
    dbg("org_name =", org_name)

    if role not in ("user", "admin"):
        return fail("role must be 'user' or 'admin'")
    if not valid_email(email):
        return fail("Invalid email")
    if role == "admin" and not org_name:
        return fail("orgName required for admin")

    acct = find_account(role, email, org_name)
    dbg("account found =", bool(acct))

    if not acct:
        dbg("No matching account, OTP not created")
        dbg("FORGOT REQUEST END")
        return ok({"message": "If the account exists, an OTP has been sent."})

    otp = generate_otp()
    otp_hash = sha256(otp)

    PasswordResetOTP.query.filter_by(role=role, email=email, org_name=org_name).delete()

    rec = PasswordResetOTP(
        role=role,
        email=email,
        org_name=org_name,
        otp_hash=otp_hash,
        expires_at=utcnow() + timedelta(minutes=10),
    )
    db.session.add(rec)
    db.session.commit()

    dbg("OTP stored in DB")
    dbg("Generated OTP =", otp)

    send_result = send_email_otp(email, otp, role)
    dbg("send_result =", send_result)

    if not send_result.get("ok"):
        dbg("Email failed, deleting OTP row")
        db.session.delete(rec)
        db.session.commit()
        dbg("FORGOT REQUEST END")
        return fail(f"Email send failed: {send_result.get('error')}", 500)

    if send_result.get("fallback"):
        dbg("DEV fallback used")
        dbg("FORGOT REQUEST END")
        return ok({"message": "OTP generated (DEV mode). Check powerpulse_debug.log for OTP."})

    dbg("OTP email sent")
    dbg("FORGOT REQUEST END")
    return ok({"message": "OTP sent to email"})


@app.post("/auth/forgot/verify")
def forgot_verify():
    data = request.get_json(silent=True) or {}
    role = (data.get("role") or "").strip().lower()
    email = (data.get("email") or "").strip().lower()
    org_name = (data.get("orgName") or "").strip() if role == "admin" else None
    otp = (data.get("otp") or "").strip()

    if role not in ("user", "admin"):
        return fail("role must be 'user' or 'admin'")
    if not valid_email(email):
        return fail("Invalid email")
    if role == "admin" and not org_name:
        return fail("orgName required for admin")
    if not otp or len(otp) != 6 or not otp.isdigit():
        return fail("Invalid OTP format")

    rec = PasswordResetOTP.query.filter_by(role=role, email=email, org_name=org_name).first()
    if not rec:
        return fail("OTP not requested or expired", 400)

    now = utcnow()
    exp = as_naive_utc(rec.expires_at)
    if exp is None or now > exp:
        db.session.delete(rec)
        db.session.commit()
        return fail("OTP expired", 400)

    if sha256(otp) != rec.otp_hash:
        return fail("Incorrect OTP", 401)

    reset_token = secrets.token_urlsafe(32)
    rec.verified_at = utcnow()
    rec.reset_token_hash = sha256(reset_token)
    rec.reset_expires_at = utcnow() + timedelta(minutes=15)
    db.session.commit()

    return ok({"message": "OTP verified", "resetToken": reset_token})


@app.post("/auth/forgot/reset")
def forgot_reset():
    data = request.get_json(silent=True) or {}
    role = (data.get("role") or "").strip().lower()
    email = (data.get("email") or "").strip().lower()
    org_name = (data.get("orgName") or "").strip() if role == "admin" else None

    reset_token = (data.get("resetToken") or "").strip()
    new_pw = data.get("newPassword") or ""
    confirm = data.get("confirmPassword") or ""

    if role not in ("user", "admin"):
        return fail("role must be 'user' or 'admin'")
    if not valid_email(email):
        return fail("Invalid email")
    if role == "admin" and not org_name:
        return fail("orgName required for admin")
    if not reset_token:
        return fail("resetToken required")
    if len(new_pw) < 6:
        return fail("Password must be at least 6 characters")
    if new_pw != confirm:
        return fail("Passwords do not match")

    rec = PasswordResetOTP.query.filter_by(role=role, email=email, org_name=org_name).first()
    if not rec or not rec.reset_token_hash:
        return fail("Reset not authorized. Verify OTP first.", 400)

    now = utcnow()
    reset_exp = as_naive_utc(rec.reset_expires_at)
    if reset_exp is None or now > reset_exp:
        db.session.delete(rec)
        db.session.commit()
        return fail("Reset token expired. Request OTP again.", 400)

    if sha256(reset_token) != rec.reset_token_hash:
        return fail("Invalid resetToken", 401)

    acct = find_account(role, email, org_name)
    if not acct:
        return fail("Account not found", 404)

    acct.password_hash = hash_password(new_pw)
    db.session.delete(rec)
    db.session.commit()

    return ok({"message": "Password updated successfully"})


@app.post("/admin/import-board-csv")
def import_board_csv():
    try:
        if "file" not in request.files:
            return fail("CSV file is required as form-data key 'file'")

        file = request.files["file"]
        if not file or not file.filename:
            return fail("No file selected")

        if not file.filename.lower().endswith(".csv"):
            return fail("Only CSV files are supported for now")

        content = file.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))

        required_columns = {
            "consumer_no",
            "consumer_name",
            "mandal",
            "sector_type",
            "reading_month",
            "reading_year",
            "previous_reading",
            "current_reading"
        }

        if not reader.fieldnames:
            return fail("CSV has no header row")

        csv_columns = {c.strip() for c in reader.fieldnames}
        missing = required_columns - csv_columns
        if missing:
            return fail(f"Missing CSV columns: {', '.join(sorted(missing))}")

        imported_count = 0
        updated_count = 0
        row_no = 1

        for row in reader:
            row_no += 1

            consumer_no = normalize_text(row.get("consumer_no"))
            consumer_name = normalize_text(row.get("consumer_name"))
            mandal = normalize_text(row.get("mandal"))
            sector_type = normalize_text(row.get("sector_type"))

            reading_month = parse_int(row.get("reading_month"), "reading_month")
            reading_year = parse_int(row.get("reading_year"), "reading_year")
            previous_reading = parse_float(row.get("previous_reading"), "previous_reading")
            current_reading = parse_float(row.get("current_reading"), "current_reading")

            if not consumer_no:
                raise ValueError(f"Row {row_no}: consumer_no required")
            if not consumer_name:
                raise ValueError(f"Row {row_no}: consumer_name required")
            if not mandal:
                raise ValueError(f"Row {row_no}: mandal required")
            if not sector_type:
                raise ValueError(f"Row {row_no}: sector_type required")
            if reading_month < 1 or reading_month > 12:
                raise ValueError(f"Row {row_no}: reading_month must be between 1 and 12")
            if reading_year < 2000 or reading_year > 2100:
                raise ValueError(f"Row {row_no}: invalid reading_year")
            if current_reading < previous_reading:
                raise ValueError(f"Row {row_no}: current_reading cannot be less than previous_reading")

            units_used = current_reading - previous_reading

            consumer = upsert_consumer(
                consumer_no=consumer_no,
                consumer_name=consumer_name,
                mandal=mandal,
                sector_type=sector_type
            )

            existing = BoardUsageRecord.query.filter_by(
                consumer_id=consumer.id,
                reading_month=reading_month,
                reading_year=reading_year
            ).first()

            if existing:
                existing.previous_reading = previous_reading
                existing.current_reading = current_reading
                existing.units_used = units_used
                existing.source_file = file.filename
                updated_count += 1
            else:
                rec = BoardUsageRecord(
                    consumer_id=consumer.id,
                    reading_month=reading_month,
                    reading_year=reading_year,
                    previous_reading=previous_reading,
                    current_reading=current_reading,
                    units_used=units_used,
                    source_file=file.filename
                )
                db.session.add(rec)
                imported_count += 1

        db.session.commit()

        return ok({
            "message": "Board CSV imported successfully",
            "importedRows": imported_count,
            "updatedRows": updated_count,
            "fileName": file.filename
        })

    except Exception as e:
        db.session.rollback()
        return fail(str(e), 400)


@app.get("/admin/dashboard-summary")
def admin_dashboard_summary():
    try:
        month = request.args.get("month", type=int)
        year = request.args.get("year", type=int)
        mandal = (request.args.get("mandal") or "").strip()

        if not month or not year:
            return fail("month and year are required")

        q = db.session.query(
            func.coalesce(func.sum(BoardUsageRecord.units_used), 0.0).label("total_demand"),
            func.count(func.distinct(BoardUsageRecord.consumer_id)).label("total_consumers"),
            func.coalesce(func.avg(BoardUsageRecord.units_used), 0.0).label("avg_units_per_user")
        ).join(Consumer, Consumer.id == BoardUsageRecord.consumer_id).filter(
            BoardUsageRecord.reading_month == month,
            BoardUsageRecord.reading_year == year
        )

        if mandal:
            q = q.filter(Consumer.mandal == mandal)

        result = q.first()

        total_demand = float(result.total_demand or 0.0)
        total_consumers = int(result.total_consumers or 0)
        avg_units_per_user = float(result.avg_units_per_user or 0.0)

        predicted_demand = round(total_demand * 1.033, 2)

        return ok({
            "month": month,
            "year": year,
            "mandal": mandal if mandal else None,
            "totalDemand": total_demand,
            "totalConsumers": total_consumers,
            "avgUnitsPerUser": avg_units_per_user,
            "predictedDemand": predicted_demand
        })

    except Exception as e:
        return fail(str(e), 500)


@app.get("/admin/monthly-demand-trend")
def monthly_demand_trend():
    try:
        mandal = (request.args.get("mandal") or "").strip()

        q = db.session.query(
            BoardUsageRecord.reading_year,
            BoardUsageRecord.reading_month,
            func.coalesce(func.sum(BoardUsageRecord.units_used), 0.0).label("total_demand")
        ).join(Consumer, Consumer.id == BoardUsageRecord.consumer_id)

        if mandal:
            q = q.filter(Consumer.mandal == mandal)

        rows = q.group_by(
            BoardUsageRecord.reading_year,
            BoardUsageRecord.reading_month
        ).order_by(
            BoardUsageRecord.reading_year.asc(),
            BoardUsageRecord.reading_month.asc()
        ).all()

        trend = []
        for r in rows:
            trend.append({
                "year": int(r.reading_year),
                "month": int(r.reading_month),
                "totalDemand": float(r.total_demand or 0.0)
            })

        return ok({"trend": trend})

    except Exception as e:
        return fail(str(e), 500)


@app.post("/board/live-usage/update")
def board_live_usage_update():
    try:
        data = request.get_json(silent=True) or {}

        mandal = (data.get("mandal") or "").strip()
        current_month_usage = float(data.get("currentMonthUsage") or 0.0)
        today_usage = float(data.get("todayUsage") or 0.0)
        current_load = float(data.get("currentLoad") or 0.0)

        if not mandal:
            return fail("mandal required")

        rec = LiveMandalUsage(
            mandal=mandal,
            reading_time=utcnow(),
            current_month_usage=current_month_usage,
            today_usage=today_usage,
            current_load=current_load,
            source="board_api"
        )
        db.session.add(rec)
        db.session.commit()

        return ok({"message": "Live usage updated"})

    except Exception as e:
        db.session.rollback()
        return fail(str(e), 400)


@app.get("/admin/live-dashboard")
def admin_live_dashboard():
    try:
        mandal = (request.args.get("mandal") or "").strip()
        if not mandal:
            return fail("mandal required")

        rec = LiveMandalUsage.query.filter_by(mandal=mandal).order_by(
            LiveMandalUsage.reading_time.desc()
        ).first()

        if rec:
            return ok({
                "mandal": rec.mandal,
                "totalDemand": rec.current_month_usage,
                "todayUsage": rec.today_usage,
                "currentLoad": rec.current_load,
                "timestamp": rec.reading_time.isoformat(),
                "source": "live_table"
            })

        now = datetime.now()
        month = now.month
        year = now.year

        q = db.session.query(
            func.coalesce(func.sum(BoardUsageRecord.units_used), 0.0).label("total_demand"),
            func.coalesce(func.avg(BoardUsageRecord.units_used), 0.0).label("avg_units")
        ).join(
            Consumer, Consumer.id == BoardUsageRecord.consumer_id
        ).filter(
            Consumer.mandal == mandal,
            BoardUsageRecord.reading_month == month,
            BoardUsageRecord.reading_year == year
        )

        result = q.first()

        total_demand = float(result.total_demand or 0.0)
        avg_units = float(result.avg_units or 0.0)

        today_usage = round(avg_units / 30, 2) if avg_units > 0 else 0.0
        current_load = round(today_usage / 24, 2) if today_usage > 0 else 0.0

        return ok({
            "mandal": mandal,
            "totalDemand": total_demand,
            "todayUsage": today_usage,
            "currentLoad": current_load,
            "timestamp": utcnow().isoformat(),
            "source": "derived_from_board_usage"
        })

    except Exception as e:
        return fail(str(e), 500)


@app.post("/user/daily-usage")
def save_daily_usage():
    try:
        data = request.get_json(silent=True)
        dbg("SAVE DAILY USAGE payload =", data)

        if data is None:
            return fail("Request body must be JSON", 400)

        if not isinstance(data, dict):
            return fail(
                "Request body must be JSON object",
                400
            )

        consumer_no = str(data.get("consumerNo") or "").strip().upper()
        usage_date_str = str(data.get("usageDate") or "").strip()
        usage_kwh = data.get("usageKwh")

        if not consumer_no:
            return fail("consumerNo required", 400)

        if not usage_date_str:
            return fail("usageDate required", 400)

        if usage_kwh is None:
            return fail("usageKwh required", 400)

        usage_date = parse_usage_date(usage_date_str)
        usage_kwh = float(usage_kwh)

        if usage_kwh < 0:
            return fail("usageKwh cannot be negative", 400)

        user = User.query.filter_by(consumer_id=consumer_no).first()
        if not user:
            return fail(f"User not found: {consumer_no}", 404)

        existing = DailyUsageInput.query.filter_by(
            consumer_no=consumer_no,
            usage_date=usage_date
        ).first()

        diff = 0.0
        if existing:
            diff = usage_kwh - float(existing.usage_kwh)
            existing.usage_kwh = usage_kwh
            message = "Daily usage updated"
        else:
            diff = usage_kwh
            row = DailyUsageInput(
                consumer_no=consumer_no,
                usage_date=usage_date,
                usage_kwh=usage_kwh
            )
            db.session.add(row)
            message = "Daily usage stored"

        # --- DEDUCT FROM ACTIVE RECHARGE ---
        active = ActiveRecharge.query.filter_by(consumer_no=consumer_no, status="ACTIVE").first()
        if active:
            # Check validity
            if active.expiry_date >= datetime.now():
                active.used_units += diff
                active.remaining_units -= diff
                
                # Auto-exhaust if remaining <= 0
                if active.remaining_units <= 0:
                    active.remaining_units = 0
                    active.status = "CONSUMED"
                    # Create Alert
                    alert = Alert(
                        consumer_no=consumer_no,
                        alert_type="EXHAUSTED",
                        title="Plan Units Exhausted",
                        message=f"Your {active.plan_name} has been fully used before expiry. Please recharge to continue service."
                    )
                    db.session.add(alert)
                elif active.remaining_units <= (active.total_units * 0.1): # 10% threshold
                    # Check if low balance alert already exists for this recharge cycle
                    low_alert = Alert.query.filter_by(consumer_no=consumer_no, alert_type="LOW_BALANCE").filter(Alert.created_at >= active.recharge_date).first()
                    if not low_alert:
                        alert = Alert(
                            consumer_no=consumer_no,
                            alert_type="LOW_BALANCE",
                            title="Low Balance Warning",
                            message=f"Low balance alert: only {round(active.remaining_units, 2)} kWh remaining in your {active.plan_name}."
                        )
                        db.session.add(alert)
                db.session.add(active)
            else:
                active.status = "EXPIRED"
                db.session.add(active)

        db.session.commit()
        return ok({
            "message": message,
            "consumerNo": consumer_no,
            "fullName": user.full_name,
            "usageDate": usage_date.isoformat(),
            "usageKwh": usage_kwh
        })

    except Exception as e:
        db.session.rollback()
        return fail(str(e), 400)


@app.get("/user/daily-usage-history")
def daily_usage_history():
    try:
        consumer_no = (request.args.get("consumerNo") or "").strip()
        if not consumer_no:
            return fail("consumerNo required")

        rows = DailyUsageInput.query.filter_by(consumer_no=consumer_no).order_by(
            DailyUsageInput.usage_date.asc()
        ).all()

        history = [
            {
                "usageDate": r.usage_date.isoformat(),
                "usageKwh": float(r.usage_kwh)
            }
            for r in rows
        ]

        return ok({
            "consumerNo": consumer_no,
            "count": len(history),
            "history": history
        })

    except Exception as e:
        return fail(str(e), 500)


@app.get("/user/dashboard-summary")
def get_dashboard_summary():
    """Returns dynamic data for the dashboard: this month's total usage,
    today's estimated cost based on usage, and remaining days in the month.
    """
    try:
        consumer_no = (request.args.get("consumerNo") or "").strip()
        if not consumer_no:
            return fail("consumerNo required", 400)
            
        # Fetch unit cost from settings
        settings = SystemSettings.query.first()
        rate_per_unit = settings.one_unit_cost if settings else 10.0

        from datetime import datetime
        import calendar
        
        now = datetime.now()
        current_year = now.year
        current_month = now.month
        
        # 1. This Month Total
        month_prefix = f"{current_year}-{current_month:02d}-"
        month_records = DailyUsageInput.query.filter(
            DailyUsageInput.consumer_no == consumer_no,
            DailyUsageInput.usage_date.like(f"{month_prefix}%")
        ).all()
        
        this_month_total = sum(float(r.usage_kwh) for r in month_records)
        
        # 2. Today's Cost Estimate
        today_str = now.strftime("%Y-%m-%d")
        today_record = DailyUsageInput.query.filter_by(
            consumer_no=consumer_no,
            usage_date=today_str
        ).first()
        
        today_usage = float(today_record.usage_kwh) if today_record else 0.0
        today_cost = today_usage * rate_per_unit
        
        # 3. Remaining Days
        _, days_in_month = calendar.monthrange(current_year, current_month)
        remaining_days = days_in_month - now.day
        
        return ok({
            "consumerNo": consumer_no,
            "thisMonthTotalKwh": round(this_month_total, 2),
            "todayUsageKwh": round(today_usage, 2),
            "todayCost": round(today_cost, 2),
            "remainingDays": remaining_days
        })

    except Exception as e:
        return fail(str(e), 500)


@app.get("/user/latest-daily-usage")
def latest_daily_usage():
    """Returns the most recent daily usage record for the consumer strictly before today.
    Tries to return yesterday's record first; if not found, returns the
    most recently stored record strictly before today.
    """
    try:
        consumer_no = (request.args.get("consumerNo") or "").strip()
        if not consumer_no:
            return fail("consumerNo required", 400)

        from datetime import date, timedelta
        today_str = date.today().isoformat()
        yesterday_str = (date.today() - timedelta(days=1)).isoformat()

        # Get the latest record strictly before today
        latest_row = DailyUsageInput.query.filter(
            DailyUsageInput.consumer_no == consumer_no,
            DailyUsageInput.usage_date < today_str
        ).order_by(DailyUsageInput.usage_date.desc()).first()

        if latest_row:
            is_yesterday = (latest_row.usage_date.isoformat() == yesterday_str)
            return ok({
                "consumerNo": consumer_no,
                "found": True,
                "usageDate": latest_row.usage_date.isoformat(),
                "usageKwh": float(latest_row.usage_kwh),
                "isYesterday": is_yesterday
            })

        return ok({
            "consumerNo": consumer_no,
            "found": False,
            "usageDate": None,
            "usageKwh": None,
            "isYesterday": False
        })

    except Exception as e:
        return fail(str(e), 500)


@app.get("/user/weekly-usage")
def get_weekly_usage():
    """Returns exactly the last 7 days of usage for the consumer."""
    try:
        consumer_no = (request.args.get("consumerNo") or "").strip()
        if not consumer_no:
            return fail("consumerNo required", 400)

        from datetime import date, timedelta
        
        today = date.today()
        # Dates from oldest to today
        dates = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
        
        start_date_str = dates[0].isoformat()
        end_date_str = dates[-1].isoformat()
        
        rows = DailyUsageInput.query.filter(
            DailyUsageInput.consumer_no == consumer_no,
            DailyUsageInput.usage_date >= start_date_str,
            DailyUsageInput.usage_date <= end_date_str
        ).all()
        
        usage_map = {r.usage_date.isoformat(): float(r.usage_kwh) for r in rows}
        
        weekly_usage = []
        for d in dates:
            d_str = d.isoformat()
            day_name = d.strftime("%a")  # e.g., 'Mon', 'Tue'
            weekly_usage.append({
                "date": d_str,
                "day": day_name,
                "usage": usage_map.get(d_str, 0.0)
            })
            
        return ok({
            "consumerNo": consumer_no,
            "weeklyUsage": weekly_usage
        })

    except Exception as e:
        return fail(str(e), 500)


@app.get("/admin/unit-cost")
def get_unit_cost():
    try:
        settings = SystemSettings.query.first()
        if not settings:
            settings = SystemSettings(one_unit_cost=10.0)
            db.session.add(settings)
            db.session.commit()
        
        return ok({
            "oneUnitCost": settings.one_unit_cost,
            "currency": settings.currency
        })
    except Exception as e:
        return fail(str(e), 500)


@app.post("/admin/unit-cost")
def update_unit_cost():
    try:
        data = request.get_json(silent=True) or {}
        new_cost = data.get("oneUnitCost")
        
        if new_cost is None:
            return fail("oneUnitCost required", 400)
            
        settings = SystemSettings.query.first()
        if not settings:
            settings = SystemSettings(one_unit_cost=float(new_cost))
            db.session.add(settings)
        else:
            settings.one_unit_cost = float(new_cost)
            
        db.session.commit()
        return ok({"message": "Unit cost updated", "oneUnitCost": settings.one_unit_cost})
    except Exception as e:
        db.session.rollback()
        return fail(str(e), 500)


@app.get("/user/current-month-daily-trend")
def get_current_month_daily_trend():
    try:
        consumer_no = (request.args.get("consumerNo") or "").strip().upper()
        if not consumer_no:
            return fail("consumerNo required", 400)

        now = datetime.now()
        prefix = f"{now.year}-{now.month:02d}-"
        
        rows = DailyUsageInput.query.filter(
            DailyUsageInput.consumer_no == consumer_no,
            DailyUsageInput.usage_date.like(f"{prefix}%")
        ).order_by(DailyUsageInput.usage_date.asc()).all()
        
        daily_trend = [
            {
                "date": r.usage_date.isoformat(),
                "day": r.usage_date.day,
                "usage": float(r.usage_kwh)
            }
            for r in rows
        ]
        
        return ok({
            "consumerNo": consumer_no,
            "month": now.month,
            "year": now.year,
            "dailyTrend": daily_trend
        })
    except Exception as e:
        return fail(str(e), 500)


@app.get("/user/current-month-weekly-trend")
def get_current_month_weekly_trend():
    try:
        consumer_no = (request.args.get("consumerNo") or "").strip().upper()
        if not consumer_no:
            return fail("consumerNo required", 400)

        now = datetime.now()
        prefix = f"{now.year}-{now.month:02d}-"
        
        rows = DailyUsageInput.query.filter(
            DailyUsageInput.consumer_no == consumer_no,
            DailyUsageInput.usage_date.like(f"{prefix}%")
        ).all()
        
        # Buckets: 1-7, 8-14, 15-21, 22-28, 29-31
        buckets = [0.0] * 5
        for r in rows:
            day = r.usage_date.day
            if 1 <= day <= 7: buckets[0] += float(r.usage_kwh)
            elif 8 <= day <= 14: buckets[1] += float(r.usage_kwh)
            elif 15 <= day <= 21: buckets[2] += float(r.usage_kwh)
            elif 22 <= day <= 28: buckets[3] += float(r.usage_kwh)
            elif 29 <= day <= 31: buckets[4] += float(r.usage_kwh)
            
        weekly_trend = []
        ranges = [(1,7), (8,14), (15,21), (22,28), (29,31)]
        
        for i in range(5):
            if buckets[i] > 0 or i == 0: # Always show week 1 if data exists or at start
                # Only show weeks that have at least one day recorded or it's the current week
                start, end = ranges[i]
                weekly_trend.append({
                    "weekNo": i + 1,
                    "startDay": start,
                    "endDay": end,
                    "usage": round(buckets[i], 2)
                })
        
        return ok({
            "consumerNo": consumer_no,
            "month": now.month,
            "year": now.year,
            "weeklyTrend": weekly_trend
        })
    except Exception as e:
        return fail(str(e), 500)


@app.post("/user/recharge-plan")
def recharge_plan():
    try:
        data = request.get_json(silent=True) or {}
        consumer_no = (data.get("consumerNo") or "").strip().upper()
        plan_id = data.get("planId")
        plan_name = data.get("planName")
        plan_price = data.get("planPrice")
        total_units = data.get("planUnits")
        validity = data.get("validityDays") or 30
        
        if not consumer_no or not plan_id:
            return fail("consumerNo and planId required", 400)

        # Deactivate existing active plans
        ActiveRecharge.query.filter_by(consumer_no=consumer_no, status="ACTIVE").update({"status": "EXPIRED"})
        
        now = datetime.now()
        expiry = now + timedelta(days=validity)
        
        new_recharge = ActiveRecharge(
            consumer_no=consumer_no,
            plan_id=plan_id,
            plan_name=plan_name,
            plan_price=plan_price,
            total_units=total_units,
            remaining_units=total_units,
            validity_days=validity,
            recharge_date=now,
            expiry_date=expiry,
            status="ACTIVE"
        )
        db.session.add(new_recharge)
        db.session.commit()
        
        return ok({"message": "Recharge successful", "expiryDate": expiry.isoformat()})
    except Exception as e:
        db.session.rollback()
        return fail(str(e), 500)


@app.get("/user/active-plan-summary")
def get_active_plan_summary():
    try:
        consumer_no = (request.args.get("consumerNo") or "").strip().upper()
        if not consumer_no:
            return fail("consumerNo required", 400)
            
        active = ActiveRecharge.query.filter_by(consumer_no=consumer_no, status="ACTIVE").first()
        
        if not active:
            return ok({"active": False})

        # Check for expiry
        if active.expiry_date < datetime.now():
            active.status = "EXPIRED"
            db.session.commit()
            return ok({"active": False})

        return ok({
            "active": True,
            "plan": {
                "planName": active.plan_name,
                "totalUnits": active.total_units,
                "usedUnits": active.used_units,
                "remainingUnits": active.remaining_units,
                "rechargeDate": active.recharge_date.date().isoformat(),
                "expiryDate": active.expiry_date.date().isoformat(),
                "status": active.status
            }
        })
    except Exception as e:
        return fail(str(e), 500)


@app.get("/user/alerts")
def get_user_alerts():
    try:
        consumer_no = (request.args.get("consumerNo") or "").strip().upper()
        if not consumer_no:
            return fail("consumerNo required", 400)
            
        alerts = Alert.query.filter_by(consumer_no=consumer_no).order_by(Alert.created_at.desc()).limit(10).all()
        return ok({
            "alerts": [
                {
                    "title": a.title,
                    "message": a.message,
                    "type": a.alert_type,
                    "date": a.created_at.isoformat()
                } for a in alerts
            ]
        })
    except Exception as e:
        return fail(str(e), 500)


def get_user_ai_context(consumer_no):
    """Gathers real-time context for the AI Assistant."""
    try:
        context = ""
        # 1. Active Plan
        active = ActiveRecharge.query.filter_by(consumer_no=consumer_no.upper(), status="ACTIVE").first()
        if active:
            context += f"Active Plan: {active.plan_name}, Remaining: {round(active.remaining_units, 2)} kWh, Expiry: {active.expiry_date.date()}. "
        else:
            context += "No active recharge plan. "

        # 2. Latest Usage
        latest = DailyUsageInput.query.filter_by(consumer_no=consumer_no.upper()).order_by(DailyUsageInput.usage_date.desc()).first()
        if latest:
            context += f"Latest usage recorded on {latest.usage_date}: {latest.usage_kwh} kWh. "

        # 3. Alerts
        alerts = Alert.query.filter_by(consumer_no=consumer_no.upper()).order_by(Alert.created_at.desc()).limit(3).all()
        if alerts:
            alert_msgs = [f"{a.title}: {a.message}" for a in alerts]
            context += "Recent Alerts: " + " | ".join(alert_msgs) + ". "

        return context
    except:
        return ""

@app.post("/api/ai-chat")
def ai_chat():
    try:
        import os
        import requests
        
        dbg("AI Chat request received")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            dbg("AI Error: Key missing")
            return fail("OpenAI API key not configured on server", 500)

        data = request.get_json(silent=True) or {}
        user_msg = data.get("message", "")
        consumer_no = data.get("consumerNo", "").upper()
        history = data.get("history", [])
        
        dbg(f"AI User Message: {user_msg}, Consumer: {consumer_no}")

        if not user_msg:
            return fail("Message is required", 400)

        # 1. Build context
        user_context = get_user_ai_context(consumer_no)
        
        # 2. Build system message
        system_instr = (
            "You are the PowerPulse AI Assistant for a home electricity monitoring and recharge app. "
            "Your goal is to help users with usage, plans, alerts, and troubleshooting. "
            "Be conversational, friendly, and concise. "
            f"\nUSER CONTEXT: {user_context}"
        )

        messages = [{"role": "system", "content": system_instr}]
        
        # 3. Add history
        for h in (history[-6:] if history else []):
            messages.append({"role": h["role"], "content": h["content"]})
            
        # 4. Add current message
        messages.append({"role": "user", "content": user_msg})

        # 5. Call OpenAI
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        dbg(f"Calling OpenAI with model: {model_name}")
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=15)
        dbg(f"OpenAI status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            reply = result['choices'][0]['message']['content']
            return ok({"response": reply.strip()})
        elif response.status_code == 429:
            dbg(f"OpenAI Quota Error: {response.text}")
            return fail("Your OpenAI API quota has been exceeded. Please check your billing details on OpenAI platform.", 429)
        else:
            dbg(f"OpenAI Error: {response.text}")
            return fail("PowerPulse AI is temporarily unavailable. Please try again later.", 502)

    except Exception as e:
        dbg(f"AI Chat Crash: {str(e)}")
        return fail("Sorry, I encountered an internal error. Please try again soon.", 500)


@app.get("/debug/daily-usage-count")
def debug_daily_usage_count():
    try:
        consumer_no = (request.args.get("consumerNo") or "").strip().upper()
        if not consumer_no:
            return fail("consumerNo required", 400)

        rows = DailyUsageInput.query.filter_by(consumer_no=consumer_no).order_by(
            DailyUsageInput.usage_date.asc()
        ).all()

        return ok({
            "consumerNo": consumer_no,
            "count": len(rows),
            "history": [
                {
                    "usageDate": r.usage_date.isoformat(),
                    "usageKwh": float(r.usage_kwh)
                }
                for r in rows
            ]
        })
    except Exception as e:
        return fail(str(e), 500)


def build_model_input_from_values(values):
    arr = np.array(values, dtype=np.float32)

    if arr.shape != (SEQ_LEN,):
        raise ValueError(f"'values' must contain exactly {SEQ_LEN} numbers")

    scaled = scaler.transform(arr.reshape(-1, 1))
    X = scaled.reshape(1, SEQ_LEN, 1)
    return X


@app.post("/predict")
def predict():
    try:
        load_energy_artifacts()
        payload = request.get_json(silent=True) or {}

        values = payload.get("values")
        if not isinstance(values, list):
            return jsonify(ok=False, error="Provide 'values' as a list"), 400

        X = build_model_input_from_values(values)
        pred_scaled = model.predict(X, verbose=0)
        pred_actual = scaler.inverse_transform(pred_scaled)

        return jsonify(
            ok=True,
            prediction={
                "next_day_energy": float(pred_actual[0][0])
            }
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400


@app.post("/predict-next-30")
def predict_next_30():
    try:
        load_energy_artifacts()
        payload = request.get_json(silent=True) or {}

        values = payload.get("values")
        if not isinstance(values, list) or len(values) != 30:
            return jsonify(ok=False, error="Input must contain 30 numbers"), 400

        current_input = scaler.transform(np.array(values).reshape(-1, 1)).flatten()
        future_predictions_scaled = []

        for _ in range(30):
            model_input = current_input.reshape(1, SEQ_LEN, 1)
            pred_scaled = model.predict(model_input, verbose=0)[0][0]
            future_predictions_scaled.append(pred_scaled)
            current_input = np.append(current_input[1:], pred_scaled)

        future_predictions = scaler.inverse_transform(
            np.array(future_predictions_scaled).reshape(-1, 1)
        ).flatten()

        return jsonify(
            ok=True,
            predictions={
                "next_30_days": [float(x) for x in future_predictions]
            }
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400


@app.get("/predict-from-db")
def predict_from_db():
    try:
        load_energy_artifacts()

        consumer_no = (request.args.get("consumerNo") or "").strip()
        if not consumer_no:
            return fail("consumerNo required")

        values, rows = get_last_30_db_values(consumer_no)

        X = build_model_input_from_values(values)
        pred_scaled = model.predict(X, verbose=0)
        pred_actual = scaler.inverse_transform(pred_scaled)

        return ok({
            "consumerNo": consumer_no,
            "prediction": {
                "next_day_energy": float(pred_actual[0][0])
            }
        })

    except Exception as e:
        return fail(str(e), 400)



@app.get("/predict-next-30-from-db")
def predict_next_30_from_db():
    dbg("ROUTE HIT: /predict-next-30-from-db")

    consumer_no = (request.args.get("consumerNo") or "").strip().upper()
    if not consumer_no:
        return fail("consumerNo required", 400)

    try:
        dbg(f"DEBUG: Processing prediction for {consumer_no}")

        # Step 1: Confirm user exists
        user = User.query.filter_by(consumer_id=consumer_no).first()
        if not user:
            return fail(f"User not found for consumerNo={consumer_no}", 404)

        # Step 2: Fetch real user daily usage history from DB
        rows = DailyUsageInput.query.filter_by(consumer_no=consumer_no).order_by(
            DailyUsageInput.usage_date.asc()
        ).all()

        if not rows:
            return fail(
                f"No daily usage records found for {consumer_no}. "
                "Please enter your daily usage data first.",
                422
            )

        user_values = [float(r.usage_kwh) for r in rows]
        dbg(f"DEBUG: {len(user_values)} usage rows found for {consumer_no}")

        # Step 3: Load the SARIMA model (the trained model on disk)
        load_sarima_artifact()

        # Step 4: Get SARIMA 30-day global baseline forecast
        forecast_raw = sarima_model_fit.forecast(steps=30)
        sarima_forecast = [float(x) for x in np.array(forecast_raw).flatten()]
        dbg(f"DEBUG: SARIMA raw forecast (first 5) = {sarima_forecast[:5]}")

        # Step 5: Scale SARIMA output to this user's actual usage level.
        # Different users produce different predictions because their usage means differ.
        sarima_mean = float(np.mean(sarima_forecast)) if sarima_forecast else 1.0
        user_mean = float(np.mean(user_values))

        if sarima_mean > 0 and user_mean > 0:
            scale_factor = user_mean / sarima_mean
        else:
            scale_factor = 1.0

        dbg(f"DEBUG: sarima_mean={sarima_mean:.4f}, user_mean={user_mean:.4f}, scale={scale_factor:.4f}")

        daily_predictions = [round(max(v * scale_factor, 0.0), 2) for v in sarima_forecast]
        dbg(f"DEBUG: Scaled predictions (first 5) = {daily_predictions[:5]}")

    except Exception as e:
        dbg(f"ERROR in predict_next_30_from_db: {str(e)}")
        return fail(f"Prediction failed: {str(e)}", 500)

    # Build response from real predictions only — no demo/static values
    total_kwh = round(sum(daily_predictions), 2)
    avg_daily = round(total_kwh / 30, 2)

    weekly_usage = [
        {
            "week": "Week 1",
            "kwh": round(sum(daily_predictions[0:7]), 2),
            "units": round(sum(daily_predictions[0:7]), 2)
        },
        {
            "week": "Week 2",
            "kwh": round(sum(daily_predictions[7:14]), 2),
            "units": round(sum(daily_predictions[7:14]), 2)
        },
        {
            "week": "Week 3",
            "kwh": round(sum(daily_predictions[14:21]), 2),
            "units": round(sum(daily_predictions[14:21]), 2)
        },
        {
            "week": "Week 4",
            "kwh": round(sum(daily_predictions[21:28]), 2),
            "units": round(sum(daily_predictions[21:28]), 2)
        },
        {
            "week": "Week 5",
            "kwh": round(sum(daily_predictions[28:30]), 2),
            "units": round(sum(daily_predictions[28:30]), 2)
        }
    ]

    graph_data = [
        {
            "day": i + 1,
            "label": f"Day {i + 1}",
            "usage": float(daily_predictions[i]),
            "units": float(daily_predictions[i]),
            "kwh": float(daily_predictions[i])
        }
        for i in range(30)
    ]

    dbg(f"DEBUG: Returning real prediction for {consumer_no}, total={total_kwh} kWh")
    return jsonify({
        "ok": True,
        "predictionSummary": {
            "nextMonthUsageUnits": total_kwh,
            "nextMonthUsageKwh": total_kwh,
            "averageDailyUnits": avg_daily,
            "averageDailyKwh": avg_daily
        },
        "predictions": {
            "next_30_days": [float(x) for x in daily_predictions]
        },
        "weeklyUsage": weekly_usage,
        "graphData": graph_data,
        "message": "Prediction generated successfully"
    }), 200


@app.get("/predict-next-month-cost-from-db")
def predict_next_month_cost_from_db():
    try:
        load_energy_artifacts()
        consumer_no = (request.args.get("consumerNo") or "").strip()
        rate = float(request.args.get("ratePerUnit", 20))

        values, _ = get_last_30_db_values(consumer_no)
        current_input = scaler.transform(np.array(values).reshape(-1, 1)).flatten()
        future_scaled = []
        for _ in range(30):
            ps = model.predict(current_input.reshape(1, 30, 1), verbose=0)[0][0]
            future_scaled.append(ps)
            current_input = np.append(current_input[1:], ps)

        future = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
        total = float(np.sum(future))
        return ok({
            "consumerNo": consumer_no,
            "nextMonthConsumptionKwh": total,
            "estimatedCost": total * rate
        })
    except Exception as e:
        return fail(str(e), 400)


@app.get("/predict-sarima-next-30")
def predict_sarima_next_30():
    try:
        load_sarima_artifact()
        forecast = sarima_model_fit.forecast(steps=30)
        forecast_list = [float(x) for x in np.array(forecast).flatten()]
        total = float(np.sum(forecast_list))
        return ok({
            "predictions": {
                "next_30_days": forecast_list,
                "next_month_consumption": total,
                "estimated_cost_formula_output": (total * 20) / 1000
            }
        })
    except Exception as e:
        return fail(str(e), 500)


# ========== HELPER: Serialize a RechargePlan to dict ==========

def plan_to_dict(p):
    return {
        "id": p.id,
        "plan_name": p.plan_name,
        "description": p.description,
        "validity_days": p.validity_days,
        "amount": p.amount,
        "units": p.units,
        "rate_per_unit": getattr(p, 'rate_per_unit', round(p.amount / p.units, 2) if p.units else 0),
        "tag": p.tag,
        "is_recommended": getattr(p, 'is_recommended', False),
        "is_active": p.is_active,
        "display_order": p.display_order,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }

@app.get("/api/usage-predict-plans")
def get_usage_predict_plans():
    try:
        plans = UsagePredictPlan.query.filter(
            UsagePredictPlan.is_active == True
        ).order_by(UsagePredictPlan.display_order.asc()).all()
        return ok({"plans": [plan_to_dict(p) for p in plans]})
    except Exception as e:
        return fail(str(e), 500)


# ========== ADMIN: Dashboard Overview ==========

@app.get("/admin/dashboard-overview")
def admin_get_dashboard_overview():
    try:
        # 1. Total Consumers (from User table)
        total_consumers = db.session.query(db.func.count(User.id)).scalar() or 0

        # 2. Total Energy Demand (Sum of all mandal usages)
        # First calculate sum of user usage in each mandal, then sum those mandal totals
        now = datetime.now()
        current_year = now.year
        current_month = now.month
        
        mandal_usage_subquery = db.session.query(
            User.mandal,
            db.func.sum(DailyUsageInput.usage_kwh).label('mandal_sum')
        ).join(
            DailyUsageInput, DailyUsageInput.consumer_no == User.consumer_id
        ).filter(
            db.func.extract('year', DailyUsageInput.usage_date) == current_year,
            db.func.extract('month', DailyUsageInput.usage_date) == current_month
        ).group_by(User.mandal).subquery()

        current_demand = db.session.query(db.func.sum(mandal_usage_subquery.c.mandal_sum)).scalar() or 0.0

        # 3. Avg Units Per User
        avg_units = 0.0
        if total_consumers > 0:
            avg_units = current_demand / total_consumers

        # 4. Predicted Demand (Current Monthly Usage * 2)
        predicted_demand = current_demand * 2

        return ok({
            "currentMonthTotalDemand": round(current_demand, 2),
            "totalConsumers": total_consumers,
            "avgUnitsPerUser": round(avg_units, 2),
            "predictedNextMonthDemand": round(predicted_demand, 2),
            "percentageChange": 0.0,
            "currentMonth": now.strftime("%B %Y")
        })

    except Exception as e:
        return fail(str(e), 500)


@app.get("/admin/mandal-analysis")
def admin_mandal_analysis():
    try:
        now = datetime.now()
        current_year = now.year
        current_month = now.month
        
        # Aggregate usage by mandal
        mandal_stats = db.session.query(
            User.mandal,
            db.func.sum(DailyUsageInput.usage_kwh).label('monthly_usage')
        ).join(
            DailyUsageInput, DailyUsageInput.consumer_no == User.consumer_id
        ).filter(
            db.func.extract('year', DailyUsageInput.usage_date) == current_year,
            db.func.extract('month', DailyUsageInput.usage_date) == current_month
        ).group_by(User.mandal).all()

        total_consumption = 0.0
        mandals_data = []
        
        # Threshold for status (example: 1000 kWh per mandal for status calc)
        THRESHOLD = 1000.0

        for row in mandal_stats:
            m_name = row.mandal or "Unknown"
            m_usage = float(row.monthly_usage or 0.0)
            total_consumption += m_usage
            
            # Load calculation
            load_percent = min(100, int((m_usage / THRESHOLD) * 100))
            
            if load_percent < 60:
                status = "STABLE"
            elif load_percent < 90:
                status = "WARNING"
            else:
                status = "OVERLOAD"
                
            mandals_data.append({
                "mandalName": m_name,
                "monthlyUsage": round(m_usage, 2),
                "currentLoadPercent": load_percent,
                "status": status
            })

        return ok({
            "totalMonthlyConsumption": round(total_consumption, 2),
            "mandals": mandals_data
        })

    except Exception as e:
        return fail(str(e), 500)


@app.get("/admin/consumers-management")
def admin_consumers_management():
    try:
        # Load SARIMA once for the request to speed up predictions
        try:
            load_sarima_artifact()
            forecast_raw = sarima_model_fit.forecast(steps=30)
            global_sarima_forecast = [float(x) for x in np.array(forecast_raw).flatten()]
            global_sarima_mean = float(np.mean(global_sarima_forecast)) if global_sarima_forecast else 1.0
        except Exception as e:
            dbg(f"SARIMA load failed in consumers-management: {e}")
            global_sarima_forecast = [15.0] * 30 # fallback
            global_sarima_mean = 15.0

        users = User.query.all()
        consumers_data = []

        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)

        for user in users:
            # 1. Avg Units (30D)
            recent_usage = db.session.query(
                func.sum(DailyUsageInput.usage_kwh)
            ).filter(
                DailyUsageInput.consumer_no == user.consumer_id,
                DailyUsageInput.usage_date >= thirty_days_ago.date()
            ).scalar() or 0.0
            
            avg_30d = round(recent_usage / 30.0, 2)

            # 2. Predicted Next Month (Scaled from SARIMA)
            # Find user mean usage from history to scale global forecast
            all_usage = db.session.query(
                func.avg(DailyUsageInput.usage_kwh)
            ).filter(DailyUsageInput.consumer_no == user.consumer_id).scalar() or 0.0
            
            user_mean = float(all_usage)
            scale_factor = user_mean / global_sarima_mean if global_sarima_mean > 0 else 1.0
            
            predicted_units = round(sum(v * scale_factor for v in global_sarima_forecast), 2)

            # 3. Status Determination
            if predicted_units > 1200 or avg_30d > 40:
                status = "PRIORITY"
            elif avg_30d > 15:
                status = "HIGH USAGE"
            else:
                status = "STABLE"

            consumers_data.append({
                "consumerNo": user.consumer_id,
                "fullName": user.full_name,
                "mandal": user.mandal or "Unknown",
                "avgUnits30D": avg_30d,
                "predictedNextMonth": predicted_units,
                "status": status
            })

        return ok({"consumers": consumers_data})

    except Exception as e:
        dbg(f"Error in admin_consumers_management: {e}")
        return fail(str(e), 500)


@app.get("/admin/consumer-details/<consumer_no>")
def admin_consumer_details(consumer_no):
    try:
        user = User.query.filter_by(consumer_id=consumer_no).first()
        if not user:
            return fail("Consumer not found", 404)

        # 1. Calculate Remaining Days from ActiveRecharge
        active = ActiveRecharge.query.filter_by(consumer_no=consumer_no, status="ACTIVE").first()
        remaining_days = 0
        if active and active.expiry_date:
            delta = active.expiry_date - datetime.now()
            remaining_days = max(0, delta.days)

        # 2. Fetch Daily Usage (Last 14 days or so for the graph)
        usage_records = DailyUsageInput.query.filter_by(consumer_no=consumer_no).order_by(
            DailyUsageInput.usage_date.desc()
        ).limit(14).all()
        
        # Reverse to get chronological order for graph
        usage_records.reverse()
        daily_usage = [{"date": r.usage_date.isoformat(), "usage": float(r.usage_kwh)} for r in usage_records]

        return ok({
            "consumer": {
                "consumerNo": user.consumer_id,
                "fullName": user.full_name,
                "mandal": user.mandal or "Unknown",
                "remainingDays": remaining_days,
                "dailyUsage": daily_usage
            }
        })

    except Exception as e:
        dbg(f"Error in admin_consumer_details: {e}")
        return fail(str(e), 500)


@app.delete("/admin/consumer/<consumer_no>")
def admin_delete_consumer(consumer_no):
    try:
        user = User.query.filter_by(consumer_id=consumer_no).first()
        if not user:
            return fail("Consumer not found", 404)

        # Delete related records
        DailyUsageInput.query.filter_by(consumer_no=consumer_no).delete()
        ActiveRecharge.query.filter_by(consumer_no=consumer_no).delete()
        Alert.query.filter_by(consumer_no=consumer_no).delete()
        RechargeOrder.query.filter_by(consumer_no=consumer_no).delete()
        ChatMessage.query.filter_by(consumer_no=consumer_no).delete()
        
        # Finally delete user
        db.session.delete(user)
        db.session.commit()

        return ok({"message": f"Consumer {consumer_no} deleted successfully"})

    except Exception as e:
        db.session.rollback()
        dbg(f"Error in admin_delete_consumer: {e}")
        return fail(str(e), 500)


@app.post("/admin/send-message")
def admin_send_message():
    try:
        data = request.json
        consumer_no = data.get("consumerNo")
        message_text = data.get("message")

        if not consumer_no or not message_text:
            return fail("Missing consumerNo or message")

        # 1. Save to ChatMessage
        chat_msg = ChatMessage(
            consumer_no=consumer_no,
            sender_role="ADMIN",
            message=message_text
        )
        db.session.add(chat_msg)

        # 2. Create Alert for User
        alert = Alert(
            consumer_no=consumer_no,
            alert_type="NEW_MESSAGE",
            title="New Admin Message",
            message=message_text
        )
        db.session.add(alert)
        
        db.session.commit()
        return ok({"message": "Message sent successfully"})

    except Exception as e:
        dbg(f"Error in admin_send_message: {e}")
        return fail(str(e), 500)


@app.get("/chat/history/<consumer_no>")
def get_chat_history(consumer_no):
    try:
        messages = ChatMessage.query.filter_by(consumer_no=consumer_no).order_by(ChatMessage.created_at.asc()).all()
        history = []
        for m in messages:
            history.append({
                "id": m.id,
                "senderRole": m.sender_role,
                "message": m.message,
                "createdAt": m.created_at.isoformat(),
                "isRead": m.is_read
            })
        return ok({"history": history})

    except Exception as e:
        dbg(f"Error in get_chat_history: {e}")
        return fail(str(e), 500)


# ========== LEGACY ROUTE (keep for backward compat) ==========

@app.get("/api/recharge-plans")
def get_recharge_plans_legacy():
    try:
        plans = RechargePlan.query.filter(
            RechargePlan.is_active == True
        ).order_by(RechargePlan.display_order.asc()).all()
        return ok({"plans": [plan_to_dict(p) for p in plans]})
    except Exception as e:
        return fail(str(e), 500)


# ========== USER: Get active recharge plans ==========

@app.get("/user/recharge-plans")
def user_get_recharge_plans():
    try:
        plans = RechargePlan.query.filter(
            RechargePlan.is_active == True
        ).order_by(RechargePlan.display_order.asc()).all()
        return ok({"plans": [plan_to_dict(p) for p in plans]})
    except Exception as e:
        return fail(str(e), 500)


# ========== ADMIN: Get all recharge plans ==========

@app.get("/admin/recharge-plans")
def admin_get_recharge_plans():
    try:
        plans = RechargePlan.query.order_by(RechargePlan.display_order.asc()).all()
        return ok({"plans": [plan_to_dict(p) for p in plans]})
    except Exception as e:
        return fail(str(e), 500)


# ========== ADMIN: Create recharge plan ==========

@app.post("/admin/recharge-plans")
def admin_create_recharge_plan():
    try:
        data = request.get_json(silent=True) or {}
        plan_name = (data.get("planName") or "").strip()
        description = (data.get("description") or "").strip() or None
        tag = (data.get("tag") or "").strip() or None

        if not plan_name:
            return fail("planName is required")

        try:
            amount = float(data.get("amount", 0))
            units = float(data.get("units", 0))
            validity_days = int(data.get("validityDays", 0))
            display_order = int(data.get("displayOrder", 0))
        except (TypeError, ValueError):
            return fail("amount, units, validityDays must be valid numbers")

        if amount <= 0:
            return fail("amount must be greater than 0")
        if units <= 0:
            return fail("units must be greater than 0")
        if validity_days <= 0:
            return fail("validityDays must be greater than 0")

        is_recommended = bool(data.get("isRecommended", False))
        is_active = bool(data.get("isActive", True))

        rate_per_unit = round(amount / units, 4) if units else None

        # If this plan is recommended, clear other recommended flags
        if is_recommended:
            RechargePlan.query.filter_by(is_recommended=True).update({"is_recommended": False})

        plan = RechargePlan(
            plan_name=plan_name,
            description=description,
            validity_days=validity_days,
            amount=amount,
            units=units,
            rate_per_unit=rate_per_unit,
            tag=tag,
            is_recommended=is_recommended,
            is_active=is_active,
            display_order=display_order,
            status="ACTIVE" if is_active else "INACTIVE",
        )
        db.session.add(plan)
        db.session.commit()
        return ok({"message": "Plan created", "plan": plan_to_dict(plan)}, 201)
    except Exception as e:
        db.session.rollback()
        return fail(str(e), 500)


# ========== ADMIN: Update recharge plan ==========

@app.put("/admin/recharge-plans/<int:plan_id>")
def admin_update_recharge_plan(plan_id):
    try:
        plan = RechargePlan.query.get(plan_id)
        if not plan:
            return fail("Plan not found", 404)

        data = request.get_json(silent=True) or {}

        if "planName" in data:
            v = (data["planName"] or "").strip()
            if not v:
                return fail("planName cannot be empty")
            plan.plan_name = v

        if "description" in data:
            plan.description = (data["description"] or "").strip() or None

        if "tag" in data:
            plan.tag = (data["tag"] or "").strip() or None

        if "amount" in data:
            try:
                v = float(data["amount"])
            except (TypeError, ValueError):
                return fail("amount must be a number")
            if v <= 0:
                return fail("amount must be greater than 0")
            plan.amount = v

        if "units" in data:
            try:
                v = float(data["units"])
            except (TypeError, ValueError):
                return fail("units must be a number")
            if v <= 0:
                return fail("units must be greater than 0")
            plan.units = v

        if "validityDays" in data:
            try:
                v = int(data["validityDays"])
            except (TypeError, ValueError):
                return fail("validityDays must be an integer")
            if v <= 0:
                return fail("validityDays must be greater than 0")
            plan.validity_days = v

        if "displayOrder" in data:
            try:
                plan.display_order = int(data["displayOrder"])
            except (TypeError, ValueError):
                return fail("displayOrder must be an integer")

        if "isRecommended" in data:
            v = bool(data["isRecommended"])
            if v:
                # Only one recommended plan at a time
                RechargePlan.query.filter(
                    RechargePlan.id != plan_id,
                    RechargePlan.is_recommended == True
                ).update({"is_recommended": False})
            plan.is_recommended = v
            if v and not plan.tag:
                plan.tag = "RECOMMENDED"

        if "isActive" in data:
            plan.is_active = bool(data["isActive"])
            plan.status = "ACTIVE" if plan.is_active else "INACTIVE"

        # Recalculate rate per unit
        if plan.units:
            plan.rate_per_unit = round(plan.amount / plan.units, 4)

        db.session.commit()
        return ok({"message": "Plan updated", "plan": plan_to_dict(plan)})
    except Exception as e:
        db.session.rollback()
        return fail(str(e), 500)


# ========== ADMIN: Toggle plan active/inactive ==========

@app.patch("/admin/recharge-plans/<int:plan_id>/status")
def admin_toggle_plan_status(plan_id):
    try:
        plan = RechargePlan.query.get(plan_id)
        if not plan:
            return fail("Plan not found", 404)

        data = request.get_json(silent=True) or {}
        if "isActive" in data:
            plan.is_active = bool(data["isActive"])
        else:
            plan.is_active = not plan.is_active  # Toggle

        plan.status = "ACTIVE" if plan.is_active else "INACTIVE"
        db.session.commit()
        return ok({"message": "Status updated", "isActive": plan.is_active, "plan": plan_to_dict(plan)})
    except Exception as e:
        db.session.rollback()
        return fail(str(e), 500)


# ========== USER: Create recharge order ==========

@app.post("/user/recharge-order/create")
def create_recharge_order():
    try:
        data = request.get_json(silent=True) or {}
        consumer_no = (data.get("consumerNo") or "").strip()
        plan_id = data.get("planId")
        payment_method = (data.get("paymentMethod") or "UPI").strip().upper()

        if not consumer_no:
            return fail("consumerNo is required")

        try:
            base_amount = float(data.get("baseAmount", 0))
            tax_amount = float(data.get("taxAmount", 0))
            total_amount = float(data.get("totalAmount", 0))
            units_snapshot = float(data.get("unitsSnapshot", 0))
            validity_days_snapshot = int(data.get("validityDaysSnapshot", 0))
        except (TypeError, ValueError):
            return fail("Invalid numeric values in request")

        if total_amount <= 0:
            return fail("totalAmount must be greater than 0")

        plan_name_snapshot = (data.get("planNameSnapshot") or "").strip()

        # Generate unique order reference
        import uuid
        order_ref = "ORD-" + str(uuid.uuid4()).upper()[:16]

        order = RechargeOrder(
            order_ref=order_ref,
            consumer_no=consumer_no,
            plan_id=plan_id,
            plan_name_snapshot=plan_name_snapshot,
            units_snapshot=units_snapshot,
            validity_days_snapshot=validity_days_snapshot,
            base_amount=base_amount,
            tax_amount=tax_amount,
            total_amount=total_amount,
            payment_method=payment_method,
            payment_status="PENDING",
        )
        db.session.add(order)
        db.session.commit()

        return ok({
            "message": "Order created",
            "orderId": order.id,
            "orderRef": order.order_ref,
        }, 201)
    except Exception as e:
        db.session.rollback()
        return fail(str(e), 500)


# ========== USER: Update recharge order status ==========

@app.post("/user/recharge-order/update-status")
def update_recharge_order_status():
    try:
        data = request.get_json(silent=True) or {}
        order_id = data.get("orderId")
        order_ref = (data.get("orderRef") or "").strip()
        status = (data.get("paymentStatus") or "").strip().upper()
        transaction_ref = (data.get("transactionRef") or "").strip() or None

        if status not in ("PENDING", "SUCCESS", "FAILED", "CANCELLED"):
            return fail("paymentStatus must be PENDING, SUCCESS, FAILED, or CANCELLED")

        order = None
        if order_id:
            order = RechargeOrder.query.get(order_id)
        elif order_ref:
            order = RechargeOrder.query.filter_by(order_ref=order_ref).first()

        if not order:
            return fail("Order not found", 404)

        order.payment_status = status
        if transaction_ref:
            order.transaction_ref = transaction_ref

        # If payment success, activate the recharge plan for the consumer
        if status == "SUCCESS" and order.plan_id:
            try:
                plan = RechargePlan.query.get(order.plan_id)
                if plan:
                    # Expire any existing active recharge
                    existing = ActiveRecharge.query.filter_by(
                        consumer_no=order.consumer_no, status="ACTIVE"
                    ).first()
                    if existing:
                        existing.status = "EXPIRED"

                    expiry = utcnow() + timedelta(days=order.validity_days_snapshot or plan.validity_days)
                    active = ActiveRecharge(
                        consumer_no=order.consumer_no,
                        plan_id=order.plan_id,
                        plan_name=order.plan_name_snapshot or plan.plan_name,
                        plan_price=order.total_amount,
                        total_units=order.units_snapshot or plan.units,
                        used_units=0.0,
                        remaining_units=order.units_snapshot or plan.units,
                        validity_days=order.validity_days_snapshot or plan.validity_days,
                        status="ACTIVE",
                        expiry_date=expiry,
                    )
                    db.session.add(active)
            except Exception as activate_err:
                dbg(f"WARNING: Could not activate plan: {activate_err}")

        db.session.commit()
        return ok({"message": "Order status updated", "orderId": order.id, "status": order.payment_status})
    except Exception as e:
        db.session.rollback()
        return fail(str(e), 500)


# ========== USER: Get recharge order by ID ==========

@app.get("/user/recharge-order/<int:order_id>")
def get_recharge_order(order_id):
    try:
        order = RechargeOrder.query.get(order_id)
        if not order:
            return fail("Order not found", 404)
        return ok({
            "order": {
                "id": order.id,
                "orderRef": order.order_ref,
                "consumerNo": order.consumer_no,
                "planId": order.plan_id,
                "planNameSnapshot": order.plan_name_snapshot,
                "unitsSnapshot": order.units_snapshot,
                "validityDaysSnapshot": order.validity_days_snapshot,
                "baseAmount": order.base_amount,
                "taxAmount": order.tax_amount,
                "totalAmount": order.total_amount,
                "paymentMethod": order.payment_method,
                "paymentStatus": order.payment_status,
                "transactionRef": order.transaction_ref,
                "createdAt": order.created_at.isoformat() if order.created_at else None,
                "updatedAt": order.updated_at.isoformat() if order.updated_at else None,
            }
        })
    except Exception as e:
        return fail(str(e), 500)


if __name__ == "__main__":
    with app.app_context():
        init_db_or_fallback()
        db.create_all()

        try:
            load_energy_artifacts()
            dbg("LSTM model + scaler loaded successfully")
        except Exception as e:
            dbg(f"WARNING: LSTM artifacts not loaded: {str(e)}")

        try:
            load_sarima_artifact()
            dbg("SARIMA model loaded successfully")
        except Exception as e:
            dbg(f"WARNING: SARIMA artifact not loaded: {str(e)}")

    app.run(host="0.0.0.0", port=5000, debug=False)

