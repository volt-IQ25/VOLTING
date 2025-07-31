from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import uuid
import qrcode
import os
import logging
import tempfile
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
from io import BytesIO
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Battery Passport API",
    description="API for generating comprehensive battery passports",
    version="2.1"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required directories
REQUIRED_DIRS = ["reports", "qr_codes", "temp_charts"]
for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)

# Data file path
DATA_FILE = "battery_passport.csv"

# Initialize or load data
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
else:
    df = pd.DataFrame(columns=[
        'vehicle_id', 'timestamp', 'cycle_number', 'SoH', 'SoC',
        'voltage_avg', 'current_avg', 'temp_avg', 'fast_charge_pct',
        'stress_score', 'suitable', 'Predicted_Health_Score', 'Second_Life_Suitable'
    ])
    df.to_csv(DATA_FILE, index=False)

# Data models
class BatteryRecord(BaseModel):
    vehicle_id: str
    timestamp: datetime
    cycle_number: int
    SoH: float
    SoC: float
    voltage_avg: float
    current_avg: float
    temp_avg: float
    fast_charge_pct: float
    stress_score: float
    suitable: bool
    Predicted_Health_Score: float
    Second_Life_Suitable: bool

class StatsResponse(BaseModel):
    vehicle_id: str
    avg_soh: float
    min_soh: float
    max_soh: float
    total_cycles: int
    second_life_suitable_pct: float
    avg_temp: float
    avg_voltage: float

class ReportInfo(BaseModel):
    report_id: str
    vehicle_id: str
    created_at: datetime
    pdf_path: str

class UploadResponse(BaseModel):
    message: str
    records_added: int
    invalid_records: int

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived metrics"""
    df['stress_score'] = (100 - df['SoH']) * 0.3 + df['fast_charge_pct'] * 0.2 + abs(df['temp_avg'] - 25) * 0.1 + (df['current_avg'] / 10 * 0.4)
    df['Predicted_Health_Score'] = (100 - (df['stress_score'] * 0.8) - (df['cycle_number'] * 0.002)).clip(0, 100)
    df['suitable'] = (df['SoH'] >= 70) & (df['Predicted_Health_Score'] >= 65) & (df['temp_avg'].between(15, 35))
    df['Second_Life_Suitable'] = (df['SoH'] >= 60) & (df['Predicted_Health_Score'] >= 50) & (df['temp_avg'].between(10, 40))
    return df

def generate_health_chart(vehicle_df, report_id):
    """Generate health trend chart image"""
    plt.figure(figsize=(8, 4))
    vehicle_df['date'] = pd.to_datetime(vehicle_df['timestamp']).dt.date
    trend = vehicle_df.groupby('date')['SoH'].mean().reset_index()
    
    plt.plot(trend['date'], trend['SoH'], marker='o', color='#1f77b4')
    plt.title('State of Health Trend', pad=20)
    plt.xlabel('Date')
    plt.ylabel('SoH (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    chart_path = f"temp_charts/{report_id}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    return chart_path

@app.post("/upload_data/", response_model=UploadResponse)
async def upload_data(file: UploadFile = File(...)):
    global df
    temp_dir = tempfile.mkdtemp()
    try:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        upload_df = pd.read_csv(
            file_path,
            parse_dates=['timestamp'],
            dtype={
                'vehicle_id': str,
                'cycle_number': 'Int64',
                'SoH': float,
                'SoC': float,
                'voltage_avg': float,
                'current_avg': float,
                'temp_avg': float,
                'fast_charge_pct': float
            }
        )
        
        required_columns = {'vehicle_id', 'timestamp', 'cycle_number', 'SoH', 'SoC', 
                          'voltage_avg', 'current_avg', 'temp_avg', 'fast_charge_pct'}
        if not required_columns.issubset(upload_df.columns):
            raise ValueError("Missing required columns")
        
        upload_df = calculate_metrics(upload_df)
        df = pd.concat([df, upload_df]).drop_duplicates().sort_values('timestamp')
        df.to_csv(DATA_FILE, index=False)
        
        return {
            "message": "Data uploaded successfully",
            "records_added": len(upload_df),
            "invalid_records": 0
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(400, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/vehicles", response_model=List[str])
def list_vehicles():
    if not df.empty:
        return df['vehicle_id'].unique().tolist()
    return []

@app.get("/stats/{vehicle_id}", response_model=StatsResponse)
def get_vehicle_stats(vehicle_id: str):
    vehicle_df = df[df['vehicle_id'] == vehicle_id]
    if vehicle_df.empty:
        raise HTTPException(404, detail="Vehicle not found")
    
    return {
        "vehicle_id": vehicle_id,
        "avg_soh": vehicle_df['SoH'].mean(),
        "min_soh": vehicle_df['SoH'].min(),
        "max_soh": vehicle_df['SoH'].max(),
        "total_cycles": int(vehicle_df['cycle_number'].max()),
        "second_life_suitable_pct": vehicle_df['Second_Life_Suitable'].mean() * 100,
        "avg_temp": vehicle_df['temp_avg'].mean(),
        "avg_voltage": vehicle_df['voltage_avg'].mean()
    }

@app.get("/records/{vehicle_id}", response_model=List[BatteryRecord])
def get_vehicle_records(vehicle_id: str):
    vehicle_df = df[df['vehicle_id'] == vehicle_id]
    if vehicle_df.empty:
        raise HTTPException(404, detail="Vehicle not found")
    return vehicle_df.to_dict('records')

@app.post("/generate_passport/{vehicle_id}", response_model=ReportInfo)
def generate_passport(vehicle_id: str):
    """Generate battery passport PDF with enhanced design"""
    try:
        vehicle_df = df[df['vehicle_id'] == vehicle_id]
        if vehicle_df.empty:
            raise HTTPException(404, detail="Vehicle not found")
        
        report_id = str(uuid.uuid4())
        filename = f"reports/{report_id}.pdf"
        chart_path = generate_health_chart(vehicle_df, report_id)
        
        # Create PDF with enhanced design
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        styles = getSampleStyleSheet()
        
        # Add custom styles with unique names to avoid conflicts
        if 'BatteryTitle' not in styles:
            styles.add(ParagraphStyle(
                name='BatteryTitle',
                fontSize=18,
                leading=22,
                alignment=1,
                spaceAfter=20,
                fontName='Helvetica-Bold',
                textColor=colors.white
            ))
        
        if 'BatteryHeader' not in styles:
            styles.add(ParagraphStyle(
                name='BatteryHeader',
                fontSize=14,
                leading=18,
                spaceAfter=12,
                fontName='Helvetica-Bold',
                textColor=colors.darkblue
            ))
        
        if 'BatteryNormalBold' not in styles:
            styles.add(ParagraphStyle(
                name='BatteryNormalBold',
                fontSize=12,
                leading=14,
                fontName='Helvetica-Bold'
            ))
        
        # Page 1: Cover Page
        c.setFillColor(colors.HexColor('#1a5276'))
        c.rect(0, height-120, width, 120, fill=True, stroke=False)
        c.setFillColor(colors.white)
        
        # Use our custom style
        title_style = styles['BatteryTitle'] if 'BatteryTitle' in styles else styles['Title']
        p = Paragraph("BATTERY PASSPORT", title_style)
        p.wrapOn(c, width, 50)
        p.drawOn(c, 0, height-90)
        
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 12)
        c.drawCentredString(width/2, height-140, "Comprehensive Battery Lifecycle Document")
        
        # Battery Info Section
        latest = vehicle_df.iloc[-1]
        stats = vehicle_df.agg({
            'SoH': ['mean', 'min', 'max'],
            'cycle_number': 'max',
            'temp_avg': 'mean',
            'voltage_avg': 'mean',
            'current_avg': 'mean',
            'fast_charge_pct': 'mean'
        })
        
        y_position = height-180
        info_sections = [
            ("Battery Identification", [
                f"<b>Vehicle ID:</b> {vehicle_id}",
                f"<b>Passport ID:</b> {report_id}",
                f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"<b>First Record:</b> {vehicle_df['timestamp'].min().strftime('%Y-%m-%d')}",
                f"<b>Last Record:</b> {latest['timestamp'].strftime('%Y-%m-%d %H:%M')}",
                f"<b>Total Records:</b> {len(vehicle_df)}"
            ]),
            ("Current Status", [
                f"<b>State of Health:</b> {latest['SoH']:.1f}%",
                f"<b>State of Charge:</b> {latest['SoC']:.1f}%",
                f"<b>Temperature:</b> {latest['temp_avg']:.1f}°C",
                f"<b>Voltage:</b> {latest['voltage_avg']:.2f}V",
                f"<b>Current:</b> {latest['current_avg']:.2f}A",
                f"<b>Cycle Number:</b> {latest['cycle_number']}"
            ]),
            ("Lifetime Statistics", [
                f"<b>Total Cycles:</b> {int(stats['cycle_number']['max'])}",
                f"<b>Average SoH:</b> {stats['SoH']['mean']:.1f}%",
                f"<b>Minimum SoH:</b> {stats['SoH']['min']:.1f}%",
                f"<b>Maximum SoH:</b> {stats['SoH']['max']:.1f}%",
                f"<b>Avg Temperature:</b> {stats['temp_avg']['mean']:.1f}°C",
                f"<b>Fast Charge Usage:</b> {stats['fast_charge_pct']['mean']:.1f}%"
            ])
        ]
        
        header_style = styles['BatteryHeader'] if 'BatteryHeader' in styles else styles['Heading2']
        normal_style = styles['Normal']
        
        for section_title, items in info_sections:
            p = Paragraph(f"<font color='#1a5276'><b>{section_title}</b></font>", header_style)
            p.wrapOn(c, width-100, 50)
            p.drawOn(c, 72, y_position)
            y_position -= 24
            
            for item in items:
                p = Paragraph(item, normal_style)
                p.wrapOn(c, width-100, 50)
                p.drawOn(c, 72, y_position)
                y_position -= 18
            y_position -= 12
        
        # QR Code
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(report_id)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="#1a5276", back_color="white")
        qr_img.save(f"qr_codes/{report_id}.png")
        c.drawImage(f"qr_codes/{report_id}.png", width-120, height-500, width=80, height=80)
        
        # Page 2: Detailed Statistics
        c.showPage()
        
        # Health Trend Chart
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(colors.HexColor('#1a5276'))
        c.drawString(72, height-72, "Battery Health Trend")
        c.drawImage(chart_path, 72, height-320, width=450, height=200)
        
        # Technical Specifications
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height-350, "Technical Specifications")
        
        tech_specs = [
            ("Stress Score", f"{latest['stress_score']:.2f}"),
            ("Predicted Health Score", f"{latest['Predicted_Health_Score']:.1f}"),
            ("Second Life Suitability", "Suitable" if latest['Second_Life_Suitable'] else "Not Suitable"),
            ("Battery Status", "Optimal" if latest['suitable'] else "Requires Attention"),
            ("Average Voltage", f"{stats['voltage_avg']['mean']:.2f}V"),
            ("Average Current", f"{stats['current_avg']['mean']:.2f}A")
        ]
        
        y_position = height-380
        for spec, value in tech_specs:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, y_position, spec + ":")
            c.setFont("Helvetica", 12)
            c.drawString(200, y_position, value)
            y_position -= 24
        
        # Maintenance Recommendations
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, y_position-30, "Maintenance Recommendations")
        c.setFont("Helvetica", 12)
        
        recommendations = []
        if latest['SoH'] < 70:
            recommendations.append("Consider battery health assessment")
        if latest['temp_avg'] > 35 or latest['temp_avg'] < 15:
            recommendations.append("Monitor operating temperature range")
        if latest['fast_charge_pct'] > 50:
            recommendations.append("Reduce fast charging frequency")
        if not recommendations:
            recommendations.append("No immediate maintenance required")
        
        y_position -= 60
        for rec in recommendations:
            c.drawString(72, y_position, "• " + rec)
            y_position -= 20
        
        c.save()
        
        # Clean up chart file
        os.remove(chart_path)
        
        return {
            "report_id": report_id,
            "vehicle_id": vehicle_id,
            "created_at": datetime.now(),
            "pdf_path": filename
        }
        
    except Exception as e:
        logger.error(f"Passport generation failed: {str(e)}\n{traceback.format_exc()}")
        if 'chart_path' in locals() and os.path.exists(chart_path):
            os.remove(chart_path)
        if 'filename' in locals() and os.path.exists(filename):
            os.remove(filename)
        raise HTTPException(500, detail=str(e))

@app.get("/download_passport/{report_id}")
def download_passport(report_id: str):
    filename = f"reports/{report_id}.pdf"
    if not os.path.exists(filename):
        raise HTTPException(404, detail="Passport not found")
    return FileResponse(
        filename,
        media_type="application/pdf",
        filename=f"battery_passport_{report_id}.pdf"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")