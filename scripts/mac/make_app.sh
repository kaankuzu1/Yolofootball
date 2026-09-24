#!/bin/bash
# Put "1v1 Analiz" in ~/Applications, so it opens from Launchpad, Spotlight
# and the Dock like any other app, with its own entry under
# System Settings > Privacy & Security > Camera.
#
#   bash scripts/mac/make_app.sh
#
# Run "scripts/mac/1v1 Analiz.command" once first: the app uses the Python
# environment that sets up in .venv. The bundle is a small launcher that
# points at this checkout, so moving the checkout means running this again.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
APP="${APP_DIR:-$HOME/Applications}/1v1 Analiz.app"

if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "Önce \"scripts/mac/1v1 Analiz.command\" dosyasını bir kez çalıştırın (.venv kurulur)."
  exit 1
fi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/MacOS/1v1Analiz" <<LAUNCH
#!/bin/bash
cd "$REPO"
exec "$REPO/.venv/bin/python" -m football_analysis.app "\$@"
LAUNCH
chmod +x "$APP/Contents/MacOS/1v1Analiz"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>1v1 Analiz</string>
  <key>CFBundleDisplayName</key><string>1v1 Analiz</string>
  <key>CFBundleIdentifier</key><string>com.yolofootball.analiz</string>
  <key>CFBundleExecutable</key><string>1v1Analiz</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSCameraUsageDescription</key>
  <string>1v1 maçını kaydetmek ve analiz etmek için kamerayı kullanır.</string>
</dict>
</plist>
PLIST

ICON="$REPO/football_analysis/app/icon.png"
if [ -f "$ICON" ] && command -v iconutil >/dev/null; then
  SET="$(mktemp -d)/AppIcon.iconset"
  mkdir -p "$SET"
  for size in 16 32 128 256 512; do
    sips -z $size $size "$ICON" --out "$SET/icon_${size}x${size}.png" >/dev/null
    sips -z $((size * 2)) $((size * 2)) "$ICON" --out "$SET/icon_${size}x${size}@2x.png" >/dev/null
  done
  iconutil -c icns "$SET" -o "$APP/Contents/Resources/AppIcon.icns"
fi

echo "Hazır: $APP"
echo "İlk açılışta kamera izni sorulur. Sorulmazsa: Sistem Ayarları > Gizlilik ve Güvenlik > Kamera."
