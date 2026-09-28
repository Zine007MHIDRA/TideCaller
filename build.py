"""Build dist/Tidecaller.exe (one file, windowed).

    python build.py
"""
import PyInstaller.__main__

PyInstaller.__main__.run([
    "tidecaller/__main__.py",
    "--name=Tidecaller",
    "--onefile",
    "--windowed",
    "--noconfirm",
    "--clean",
    "--paths=.",
    "--add-data=tidecaller/config/rods.json;tidecaller/config",
    "--collect-submodules=tidecaller",
    # heavy optional deps the runtime never imports
    "--exclude-module=torch",
    "--exclude-module=ultralytics",
    "--exclude-module=matplotlib",
    "--exclude-module=pytest",
])
