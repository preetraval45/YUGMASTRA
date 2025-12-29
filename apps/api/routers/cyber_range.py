"""Cyber Range API Router"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
async def get_cyber_range_status():
    """Get cyber range status"""
    return {
        "status": "running",
        "hosts": 5,
        "compromised": 2,
        "uptime": "2h 34m"
    }


@router.post("/reset")
async def reset_cyber_range():
    """Reset cyber range to initial state"""
    return {"status": "reset_initiated"}
