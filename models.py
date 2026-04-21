from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    deficiency = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.now)