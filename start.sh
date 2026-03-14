#!/bin/bash
# Startup script: patch app/config.py BEFORE launching Python
# This fixes the DaytonaSettings.daytona_api_key required field issue

set -e

CONFIG_PY="/app/OpenManus/app/config.py"

if [ -f "$CONFIG_PY" ]; then
    echo "[start.sh] Patching $CONFIG_PY ..."
    
    # Fix 1: make daytona_api_key optional (str -> Optional[str] with default None)
    sed -i 's/    daytona_api_key: str$/    daytona_api_key: Optional[str] = Field(None, description="Daytona API key (optional)")/' "$CONFIG_PY"
    
    # Fix 2: replace DaytonaSettings() with None when no config provided
    sed -i 's/            daytona_settings = DaytonaSettings()$/            daytona_settings = None/' "$CONFIG_PY"
    
    echo "[start.sh] Patch applied. Verifying..."
    grep -n "daytona_api_key" "$CONFIG_PY" | head -3
    grep -n "daytona_settings = " "$CONFIG_PY" | head -5
fi

echo "[start.sh] Starting OpenManus server..."
exec python server.py
