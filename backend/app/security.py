"""服务端请求鉴权与内存限流。生产环境应替换为网关或 Redis 限流。"""

from __future__ import annotations

import hmac
import base64
import hashlib
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock

import jwt


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    roles: tuple[str, ...] = ("user",)


class AuthenticationError(ValueError):
    """访问令牌缺失、失效或不满足服务端策略。"""


def auth_required() -> bool:
    return os.getenv("AUTH_REQUIRED", "false").strip().lower() in {"1", "true", "yes"}


def is_production_environment() -> bool:
    return os.getenv("ENVIRONMENT", "development").strip().lower() in {"production", "prod"}


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "")
    if len(secret) < 32:
        raise AuthenticationError("JWT_SECRET 必须设置为至少 32 字符的随机密钥。")
    return secret


def password_hash(password: str, salt: bytes | None = None, iterations: int = 600_000) -> str:
    """生成 PBKDF2-SHA256 密码哈希，仅用于配置初始本地账号。"""
    if len(password) < 12:
        raise ValueError("密码至少需要 12 个字符。")
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        scheme, iterations, salt, expected = encoded_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        actual = password_hash(password, base64.urlsafe_b64decode(salt), int(iterations))
        return hmac.compare_digest(actual, encoded_hash)
    except (ValueError, TypeError):
        return False


def issue_access_token(user: AuthenticatedUser) -> str:
    now = datetime.now(timezone.utc)
    expires_minutes = max(5, min(int(os.getenv("JWT_EXPIRES_MINUTES", "30")), 480))
    payload = {
        "sub": user.user_id,
        "roles": list(user.roles),
        "iss": os.getenv("JWT_ISSUER", "zhixing-agent"),
        "aud": os.getenv("JWT_AUDIENCE", "zhixing-web"),
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def authenticate_bearer(authorization: str | None) -> AuthenticatedUser:
    if not auth_required():
        return AuthenticatedUser("local-development", ("developer",))
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthenticationError("缺少 Bearer 访问令牌。")
    try:
        claims = jwt.decode(
            authorization.removeprefix("Bearer "),
            _jwt_secret(),
            algorithms=["HS256"],
            issuer=os.getenv("JWT_ISSUER", "zhixing-agent"),
            audience=os.getenv("JWT_AUDIENCE", "zhixing-web"),
            options={"require": ["sub", "exp", "iat", "iss", "aud"]},
        )
        user_id = claims["sub"]
        roles = claims.get("roles", ["user"])
        if not isinstance(user_id, str) or not user_id or not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
            raise AuthenticationError("令牌主体格式无效。")
        return AuthenticatedUser(user_id, tuple(roles))
    except (jwt.PyJWTError, KeyError, AuthenticationError) as error:
        raise AuthenticationError("访问令牌无效或已过期。") from error


def verify_bootstrap_credentials(username: str, password: str) -> AuthenticatedUser | None:
    """校验环境变量中的单个初始账号；生产可替换为企业 IdP/OIDC。"""
    configured_user = os.getenv("AUTH_BOOTSTRAP_USER", "")
    configured_hash = os.getenv("AUTH_BOOTSTRAP_PASSWORD_HASH", "")
    if not configured_user or not configured_hash:
        return None
    if not hmac.compare_digest(username, configured_user) or not verify_password(password, configured_hash):
        return None
    return AuthenticatedUser(configured_user)


class SlidingWindowRateLimiter:
    """按客户端标识限制固定时间窗口内的请求数。"""

    def __init__(self, limit: int = 20, window_seconds: int = 60):
        if limit < 1 or window_seconds < 1:
            raise ValueError("限流参数必须为正数。")
        self.limit = limit
        self.window_seconds = window_seconds
        self._records: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, identity: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        with self._lock:
            records = self._records[identity]
            threshold = current - self.window_seconds
            while records and records[0] <= threshold:
                records.popleft()
            if len(records) >= self.limit:
                return False
            records.append(current)
            return True


class RedisRateLimiter:
    """使用 Redis INCR/EXPIRE 的跨实例固定窗口限流。"""
    def __init__(self, url: str, limit: int, window_seconds: int):
        import redis
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.limit, self.window_seconds = limit, window_seconds

    def allow(self, identity: str, now: float | None = None) -> bool:
        bucket = int(time.time() // self.window_seconds)
        key = f"agent:rate:{bucket}:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        count = self.client.incr(key)
        if count == 1:
            self.client.expire(key, self.window_seconds + 1)
        return count <= self.limit


def create_rate_limiter(limit: int, window_seconds: int):
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        return RedisRateLimiter(redis_url, limit, window_seconds)
    return SlidingWindowRateLimiter(limit, window_seconds)


def login_rate_identity(client_host: str, username: str) -> str:
    """不在 Redis 键或日志中暴露登录名；同一用户名的大小写归一化。"""
    digest = hashlib.sha256(username.strip().casefold().encode("utf-8")).hexdigest()[:24]
    return f"login:{client_host}:user:{digest}"


def validate_multi_instance_configuration() -> None:
    """生产多实例模式不允许降级到本地安全状态。"""
    enabled = os.getenv("MULTI_INSTANCE_MODE", "false").lower() in {"1", "true", "yes"}
    if not enabled:
        return
    missing = [name for name in ("REDIS_URL", "CHECKPOINT_DATABASE_URL", "MEMORY_DATABASE_URL") if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"多实例模式缺少必需配置：{', '.join(missing)}")


def validate_production_configuration() -> None:
    """生产环境 fail-closed：禁止静默回退到开发认证和本地状态。"""
    if not is_production_environment():
        return
    if not auth_required():
        raise RuntimeError("生产环境必须设置 AUTH_REQUIRED=true。")
    required = (
        "JWT_SECRET",
        "AUTH_BOOTSTRAP_USER",
        "AUTH_BOOTSTRAP_PASSWORD_HASH",
        "AGENT_CORS_ORIGINS",
        "REDIS_URL",
        "MEMORY_DATABASE_URL",
        "CHECKPOINT_DATABASE_URL",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"生产环境缺少必需配置：{', '.join(missing)}")
    _jwt_secret()
    origins = [origin.strip() for origin in os.environ["AGENT_CORS_ORIGINS"].split(",")]
    if not origins or any(not origin.startswith("https://") or origin == "*" or "localhost" in origin or "127.0.0.1" in origin for origin in origins):
        raise RuntimeError("生产环境 AGENT_CORS_ORIGINS 必须为明确的非本地 HTTPS 前端域名。")
    if os.getenv("MULTI_INSTANCE_MODE", "false").lower() not in {"1", "true", "yes"}:
        raise RuntimeError("生产环境必须设置 MULTI_INSTANCE_MODE=true，禁止本地状态回退。")
    if os.getenv("KNOWLEDGE_EMBEDDING_PROVIDER", "").strip().lower() == "openai":
        if not os.getenv("KNOWLEDGE_EMBEDDING_MODEL_REVISION"):
            raise RuntimeError("生产环境启用 Embedding 时必须显式设置 KNOWLEDGE_EMBEDDING_MODEL_REVISION。")
        if os.getenv("KNOWLEDGE_EMBEDDING_SERVICE_URL") and not os.getenv("EMBEDDING_SERVICE_API_KEY"):
            raise RuntimeError("使用独立 Embedding 服务时必须设置 EMBEDDING_SERVICE_API_KEY。")
        if os.getenv("KNOWLEDGE_EMBEDDING_SERVICE_URL") and len(os.getenv("EMBEDDING_SERVICE_API_KEY", "")) < 32:
            raise RuntimeError("生产环境 EMBEDDING_SERVICE_API_KEY 至少需要 32 个字符。")
    validate_multi_instance_configuration()


def valid_api_key(provided_key: str | None, expected_key: str | None) -> bool:
    """未配置服务密钥时仅允许本地开发；配置后使用恒定时间比较。"""
    if not expected_key:
        return True
    return bool(provided_key) and hmac.compare_digest(
        provided_key.encode("utf-8"), expected_key.encode("utf-8")
    )
