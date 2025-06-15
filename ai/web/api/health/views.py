from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status

"""
# deployment.yaml
spec:
  containers:
    - name: app
      livenessProbe:
        httpGet:
          path: /health/liveness
          port: 5000
        initialDelaySeconds: 5
        periodSeconds: 10
        
      readinessProbe:
        httpGet:
          path: /health/readiness
          port: 5000
        initialDelaySeconds: 3
        periodSeconds: 5
        failureThreshold: 3
        
      startupProbe:
        httpGet:
          path: /health/startup
          port: 5000
        failureThreshold: 30
        periodSeconds: 5
"""

router = APIRouter()


@router.get("/liveness")
async def liveness():
    return {"status": "UP", "timestamp": datetime.now(timezone.utc)}


@router.get("/readiness")
async def readiness(request: Request):
    redis = request.app.state.redis_pool

    if not redis.can_get_connection():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "DOWN",
                "component": "redis",
                "error": getattr(redis, "last_error", "Not registered"),
            },
        )

    return {"status": "UP", "components": {"redis": "UP"}}


@router.get("/startup")
async def startup(request: Request):
    redis = request.app.state.redis_pool

    if not redis.can_get_connection():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "STARTING", "connfig_loaded": False},
        )

    return {"status": "UP"}
