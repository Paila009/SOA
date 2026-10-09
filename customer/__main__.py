import os
import uvicorn
from .app import create_app
from .config import load_settings

if __name__ == "__main__":
    settings = load_settings()
    host = os.getenv("CUSTOMER_HOST", "127.0.0.1")
    if settings.preview and host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("Preview must bind to localhost. Configure Firebase before exposing this server.")
    uvicorn.run(create_app(settings), host=host, port=int(os.getenv("CUSTOMER_PORT", "8770")),
                access_log=False, proxy_headers=settings.environment == "production", workers=1)
