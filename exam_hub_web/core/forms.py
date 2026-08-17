from django import forms

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".webm"}
MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB — tune to your infra

# Maps the "human" choices shown in the UI to the codes Whisper actually expects.
LANGUAGE_CHOICES = [
    ("", "Auto-Detect Language"),
    ("si", "Sinhala"),
    ("en", "English"),
]

TASK_CHOICES = [
    ("transcribe", "Transcribe (Keep Original Language)"),
    ("translate", "Translate to English"),
]


class AudioUploadForm(forms.Form):
    audio_file = forms.FileField(required=True)
    language = forms.ChoiceField(choices=LANGUAGE_CHOICES, required=False)
    task = forms.ChoiceField(choices=TASK_CHOICES, required=True)

    def clean_audio_file(self):
        f = self.cleaned_data["audio_file"]

        if f.size == 0:
            raise forms.ValidationError("The uploaded file is empty.")
        if f.size > MAX_UPLOAD_SIZE_BYTES:
            mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            raise forms.ValidationError(f"File is too large. Maximum size is {mb} MB.")

        ext = "." + f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
            raise forms.ValidationError(f"Unsupported file type '{ext}'. Allowed: {allowed}")

        return f

    def clean_language(self):
        # Empty string means "auto-detect" — normalize to None for the processor.
        return self.cleaned_data["language"] or None