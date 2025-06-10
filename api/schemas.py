from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    income_am: float
    profit_last_am: float
    profit_am: float
    damage_am: float
    damage_inc: float
    crd_lim_rec: float
    credit_use_ic: float
    lactose_ic: float
    insurance_ic: float
    spa_ic: float
    empl_ic: float
    cab_requests: float
    married_cd: bool
    bar_no: float
    sport_ic: float
    neighbor_income: float
    age: float
    marketing_permit: float
    urban_ic: float
    dining_ic: float
    presidential: float
    client_segment: float
    sect_empl: float
    prev_stay: float
    prev_all_in_stay: float
    fam_adult_size: float
    children_no: float
    tenure_mts: float
    company_ic: float
    claims_no: float
    claims_am: float
    nights_booked: float
    gender: int
    shop_am: float
    shop_use: float
    retired: float
    gold_status: float


class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="The predicted class (0 or 1).")
    probability: float = Field(
        ...,
        description="The probability of the prediction, ranging from 0.0 to 1.0.",
    )
