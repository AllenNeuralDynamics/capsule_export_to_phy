version=$(grep -rho '"version": *"0\.1[0-9][0-9]\.[0-9]\+"' /root/capsule/data --include="*.json" | head -n 1 | sed -E 's/.*"version": *"([^"]+)".*/\1/')

echo Installing SpikeInterface version extracted from sorted asset: $version

pip install --no-cache-dir --ignore-installed spikeinterface[full]=="$version" -q