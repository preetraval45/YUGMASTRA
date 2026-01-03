#!/bin/bash

# Generate Self-Signed SSL Certificate for Local Development
# For production, use Let's Encrypt instead

echo "Generating self-signed SSL certificate for local development..."

# Create ssl directory if it doesn't exist
mkdir -p nginx/ssl

# Generate private key and certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=YUGMASTRA/OU=Development/CN=localhost"

echo "✅ SSL certificate generated successfully!"
echo "   Certificate: nginx/ssl/cert.pem"
echo "   Private Key: nginx/ssl/key.pem"
echo ""
echo "⚠️  This is a SELF-SIGNED certificate for DEVELOPMENT only!"
echo "   For production, use Let's Encrypt: https://letsencrypt.org/"
echo ""
echo "To enable HTTPS:"
echo "  1. Update docker-compose.yml to use nginx-https.conf"
echo "  2. Expose port 443"
echo "  3. Restart nginx container"
