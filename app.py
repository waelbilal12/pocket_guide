from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
from typing import Dict, Any
from datetime import datetime

app = FastAPI(
    title="Whisper Arabic STT API",
    description="API لتحويل الصوت إلى نص مع دعم اللهجات العربية (خاصة الشامية)",
    version="1.0.0"
)

# تمكين CORS للسماح باتصالات من تطبيق Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النموذج عند بدء التشغيل
MODEL_SIZE = "tiny"  # يمكن تغييرها إلى large-v3 إذا كانت الموارد كافية
model = whisper.load_model(MODEL_SIZE)

# قاموس تحويل الكلمات العامية إلى الفصحى (مع الاحتفاظ بالأصل)
DIALECT_MAPPING = {
    # يمكن إضافة المزيد من الكلمات
}

def post_process_arabic_text(text: str) -> str:
    """معالجة النص لتحسين اللهجات العربية"""
    processed_text = text
    for dialect, fusha in DIALECT_MAPPING.items():
        if dialect in processed_text:
            processed_text = processed_text.replace(dialect, f"{dialect} ({fusha})")
    return processed_text

@app.post("/transcribe", response_model=Dict[str, Any])
async def transcribe_audio(
    file: UploadFile = File(..., description="ملف صوتي لتحويله إلى نص (MP3, WAV, etc.)"),
    language: str = "ar",
    include_dialect_info: bool = True,
    temperature: float = 0.2,
    beam_size: int = 5
):
    """
    تحويل الصوت إلى نص مع دعم خاص للهجات العربية
    
    Parameters:
    - file: ملف الصوت المراد تحويله
    - language: لغة الصوت (افتراضي: العربية)
    - include_dialect_info: تضمين تفسير الكلمات العامية
    - temperature: درجة الإبداع (0-1، كلما قل الرقم زادت الدقة)
    - beam_size: حجم الحزمة للبحث (يزيد الدقة لكن يبطئ المعالجة)
    """
    try:
        start_time = datetime.now()
        
        # التحقق من صيغة الملف
        valid_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"صيغة الملف غير مدعومة. الصيغ المدعومة: {', '.join(valid_extensions)}"
            )

        # حفظ الملف مؤقتاً
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # خيارات النسخ
        options = {
            "language": language,
            "task": "transcribe",
            "temperature": temperature,
            "beam_size": beam_size,
            "initial_prompt": "نص عربي يحتوي على لهجات شامية مع بعض الكلمات العامية."
        }

        # تنفيذ التحويل
        result = model.transcribe(temp_file_path, **options)
        
        # معالجة النص إذا كانت اللغة عربية
        if language == "ar" and include_dialect_info:
            result["text"] = post_process_arabic_text(result["text"])
            for segment in result["segments"]:
                segment["text"] = post_process_arabic_text(segment["text"])

        # حساب وقت المعالجة
        processing_time = (datetime.now() - start_time).total_seconds()

        # تنظيف الملف المؤقت
        os.unlink(temp_file_path)

        return {
            "status": "success",
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": language,
            "processing_time_seconds": processing_time,
            "model_used": MODEL_SIZE
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"حدث خطأ أثناء معالجة الملف: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """الحصول على معلومات عن النموذج المستخدم"""
    return {
        "model_size": MODEL_SIZE,
        "supported_languages": "متعددة (الافتراضي: العربية)",
        "arabic_dialects_support": True,
        "status": "جاهز"
    }