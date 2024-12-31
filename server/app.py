from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from seymour import run_trading_strategy

# Define the input schema
class StrategyParams(BaseModel):
    asset: str
    start_date: str
    params: dict = {}

# Initialize FastAPI app
app = FastAPI()

@app.post("/api/backtest")
def backtest(params: StrategyParams):
    try:
        results = run_trading_strategy(
            asset=params.asset,
            start_date=params.start_date,
            params=params.params
        )
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# Add a simple health check
@app.get("/")
def root():
    return {"message": "API is running!"}
