from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

class Transaction(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    customer_id: UUID
    account_id: UUID
    merchant_id: UUID
    amount: float
    timestamp: datetime
    category: str
    description: Optional[str] = None
    location: Optional[str] = None
    is_anomaly: bool = False
    anomaly_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class Customer(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    email: str
    phone: Optional[str] = None
    risk_score: float = 0.0
    behavioral_cluster: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    accounts: List[UUID] = []

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class Account(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    customer_id: UUID
    account_type: str
    balance: float
    currency: str = "INR"
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class Merchant(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    category: str
    location: Optional[str] = None
    risk_score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        } 