"""Analytics API Router"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/dashboard")
async def get_dashboard_data():
    """Get dashboard analytics"""
    return {
        "overview": {
            "total_episodes": 523,
            "red_wins": 271,
            "blue_wins": 252,
            "current_phase": "exploration"
        },
        "recent_activity": []
    }


@router.get("/trends")
async def get_trends(timeframe: str = "24h"):
    """Get trend analysis"""
    return {"trends": []}
