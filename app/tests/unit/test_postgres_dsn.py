from app.infra.db import build_postgres_dsn


def test_build_postgres_dsn_prefers_explicit_dsn() -> None:
    dsn = build_postgres_dsn(
        db_dsn="postgresql://u:p@h:5432/db",
        user="ignored",
        password="ignored",
        host="ignored",
        port=5432,
        db_name="ignored",
        ssl_mode=None,
    )

    assert dsn == "postgresql://u:p@h:5432/db"


def test_build_postgres_dsn_from_parts() -> None:
    dsn = build_postgres_dsn(
        db_dsn=None,
        user="user",
        password="pass",
        host="localhost",
        port=5432,
        db_name="mydb",
        ssl_mode="require",
    )

    assert dsn.startswith("postgresql://user:pass@localhost:5432/mydb")
    assert "sslmode=require" in dsn
