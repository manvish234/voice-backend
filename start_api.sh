#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

DOMAIN="voicehelp.myclassboard.com"
FRONTEND_DIR="$(pwd)/frontend"

# --- Step 1: Unpack backend if needed ---
python unpack_backend.py

# --- Step 2: Write nginx config ---
NGINX_CONF="/etc/nginx/sites-available/$DOMAIN"
if [ ! -f "$NGINX_CONF" ]; then
  echo "📝 Creating nginx config for $DOMAIN ..."
  sudo tee "$NGINX_CONF" > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;

    # Serve frontend
    root $FRONTEND_DIR;
    index index.html;

    location / {
        try_files \$uri /index.html;
    }

    # Proxy API calls
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

  # Enable site
  sudo ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/
fi

echo "🔄 Reloading nginx..."
sudo nginx -t && sudo systemctl reload nginx

# --- Step 3: Start Backend API ---
echo "🚀 Starting API server at http://localhost:8000 ..."
if command -v uvicorn >/dev/null 2>&1; then
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
else
  python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
fi
