ECHO "START FEATURE EXTRACTION BAT FILE"

"C:\Anaconda51_py36x64\python.exe" feature_extraction.py

IF %ERRORLEVEL% NEQ 0 (
"C:\Anaconda51_py36x64\python.exe" feature_extraction.py
)

ECHO "FEATURE EXTRACTION FINISH, LAUNCH MODEL TRAINING"

"C:\Anaconda51_py36x64\python.exe" train_classifiers.py

ECHO "FINISH FEATURE EXTRACTION BATCH FILE"