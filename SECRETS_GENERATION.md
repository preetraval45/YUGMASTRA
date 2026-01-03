# Production Secrets Generation Guide

**IMPORTANT**: Never commit actual secrets to git! This file shows you HOW to generate them.

## Quick Setup

### 1. Generate All Secrets at Once

Run these commands and save the output securely:

```bash
# Generate JWT secret (64 characters hex)
echo "JWT_SECRET=$(openssl rand -hex 32)"

# Generate NextAuth secret (base64)
echo "NEXTAUTH_SECRET=$(openssl rand -base64 32)"

# Generate session secret (64 characters hex)
echo "SESSION_SECRET=$(openssl rand -hex 32)"

# Generate database password (32 characters base64)
echo "POSTGRES_PASSWORD=$(openssl rand -base64 24)"

# Generate Redis password (optional)
echo "REDIS_PASSWORD=$(openssl rand -base64 24)"
```

### 2. Example Output (DO NOT USE THESE - GENERATE YOUR OWN!)

```env
JWT_SECRET=147553315a520cf50d810148611eeb64903558c026af80fa75d29bea61e76640
NEXTAUTH_SECRET=0O6rGYp/+tUsRxFubMusKyEgMTj5pEt9/8GnzIQtcJQ=
SESSION_SECRET=265aa707bfbec562c103bd2a1a719d84e4ae76046b396238ed56242a01a73906
POSTGRES_PASSWORD=lKrI9fGrmXRghXRUjvuyTxSK0suu1ZwJ
```

### 3. Update Your .env File

**For Development**:
```bash
# Copy example file
cp .env.example .env

# Edit .env with your editor
nano .env  # or vim, code, etc.

# Replace the example secrets with generated ones
```

**For Production**:
```bash
# NEVER use .env file in production
# Use environment variables or secrets management service

# Example for Docker:
docker-compose up -d -e JWT_SECRET=your-secret-here

# Example for Kubernetes:
kubectl create secret generic yugmastra-secrets \
  --from-literal=jwt-secret=your-secret-here \
  --from-literal=nextauth-secret=your-secret-here

# Example for AWS (use AWS Secrets Manager)
# Example for Azure (use Azure Key Vault)
# Example for GCP (use Secret Manager)
```

## Secret Requirements

### JWT_SECRET
- **Format**: 64 character hexadecimal string
- **Used For**: Signing JWT tokens for authentication
- **Generation**: `openssl rand -hex 32`
- **Strength**: 256 bits of entropy
- **Rotation**: Every 90 days recommended

### NEXTAUTH_SECRET
- **Format**: Base64 encoded string (44 characters)
- **Used For**: NextAuth.js session encryption
- **Generation**: `openssl rand -base64 32`
- **Strength**: 256 bits of entropy
- **Rotation**: Every 90 days recommended

### SESSION_SECRET
- **Format**: 64 character hexadecimal string
- **Used For**: Session cookie signing
- **Generation**: `openssl rand -hex 32`
- **Strength**: 256 bits of entropy
- **Rotation**: Every 90 days recommended

### POSTGRES_PASSWORD
- **Format**: Base64 encoded string (32 characters)
- **Used For**: PostgreSQL database authentication
- **Generation**: `openssl rand -base64 24`
- **Strength**: 192 bits of entropy
- **Rotation**: Every 180 days recommended
- **Additional**: Enable PostgreSQL SSL/TLS in production

### REDIS_PASSWORD (Optional)
- **Format**: Base64 encoded string
- **Used For**: Redis authentication
- **Generation**: `openssl rand -base64 24`
- **Strength**: 192 bits of entropy
- **Note**: Redis runs in isolated Docker network, password optional

## Security Best Practices

### 1. Never Commit Secrets to Git
```bash
# Ensure .env is in .gitignore
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore
echo ".env.production" >> .gitignore

# Check for accidentally committed secrets
git log -S "your-secret-here" --all
```

### 2. Use Environment-Specific Secrets
```
Development: Different secrets (can be simpler)
Staging: Different secrets (production-strength)
Production: Unique, strong secrets (managed securely)
```

### 3. Secret Rotation Schedule
```
Critical Secrets (JWT, Auth):  Every 90 days
Database Passwords:            Every 180 days
API Keys:                      Every 90 days
SSL Certificates:              Before expiration (auto-renew)
```

### 4. Secret Storage Options

**Development (Local)**:
- `.env` file (gitignored)
- Local password manager

**Production**:
- ✅ AWS Secrets Manager
- ✅ Azure Key Vault
- ✅ Google Secret Manager
- ✅ HashiCorp Vault
- ✅ Docker Secrets
- ✅ Kubernetes Secrets
- ❌ Environment variables in docker-compose.yml
- ❌ Hardcoded in application code

## Environment Variable Setup

### Docker Compose (Development)
```yaml
# docker-compose.yml
services:
  web:
    environment:
      JWT_SECRET: ${JWT_SECRET}
      NEXTAUTH_SECRET: ${NEXTAUTH_SECRET}
      SESSION_SECRET: ${SESSION_SECRET}
```

### Docker Swarm (Production)
```bash
# Create secrets
echo "your-jwt-secret" | docker secret create jwt_secret -
echo "your-nextauth-secret" | docker secret create nextauth_secret -

# Use in docker-compose.yml
secrets:
  - jwt_secret
  - nextauth_secret
```

### Kubernetes (Production)
```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: yugmastra-secrets
type: Opaque
data:
  jwt-secret: <base64-encoded-secret>
  nextauth-secret: <base64-encoded-secret>
```

## Verification

### Test Your Secrets Are Loaded
```bash
# Start containers
docker-compose up -d

# Check environment variables (be careful not to expose)
docker exec yugmastra-web printenv | grep -E "JWT_SECRET|NEXTAUTH"
# Should show "JWT_SECRET=***" (redacted by Docker)

# Test authentication
curl -X POST http://localhost:200/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"Test123!@#Test"}'
```

### Check Secret Strength
```bash
# JWT_SECRET should be 64 chars
echo -n "your-jwt-secret" | wc -c
# Should output: 64

# NEXTAUTH_SECRET should be ~44 chars (base64)
echo -n "your-nextauth-secret" | wc -c
# Should output: 43-44
```

## Troubleshooting

### "Invalid JWT Token" Error
- **Cause**: JWT_SECRET changed but old tokens still in use
- **Fix**: Clear browser cookies and restart application

### "Session Invalid" Error
- **Cause**: SESSION_SECRET changed
- **Fix**: Users need to re-login

### "Database Authentication Failed"
- **Cause**: POSTGRES_PASSWORD doesn't match database
- **Fix**:
  ```bash
  # Reset database password
  docker exec yugmastra-postgres psql -U postgres -c \
    "ALTER USER yugmastra PASSWORD 'new-password';"
  ```

### "OAuth Configuration Error"
- **Cause**: NEXTAUTH_SECRET not set
- **Fix**: Generate and set NEXTAUTH_SECRET

## Emergency Secret Rotation

If secrets are compromised:

### Immediate Actions (< 1 hour)
1. Generate new secrets immediately
2. Update environment variables
3. Restart all services
4. Invalidate all existing sessions
5. Force all users to re-login

### Follow-up Actions (< 24 hours)
1. Review access logs for suspicious activity
2. Notify users of security incident (if required)
3. Update monitoring alerts
4. Document incident
5. Review and improve secret management

### Commands
```bash
# 1. Generate new secrets
NEW_JWT=$(openssl rand -hex 32)
NEW_NEXTAUTH=$(openssl rand -base64 32)
NEW_SESSION=$(openssl rand -hex 32)

# 2. Update .env file
sed -i "s/JWT_SECRET=.*/JWT_SECRET=$NEW_JWT/" .env
sed -i "s/NEXTAUTH_SECRET=.*/NEXTAUTH_SECRET=$NEW_NEXTAUTH/" .env
sed -i "s/SESSION_SECRET=.*/SESSION_SECRET=$NEW_SESSION/" .env

# 3. Restart services
docker-compose restart

# 4. Clear sessions (PostgreSQL)
docker exec yugmastra-postgres psql -U yugmastra -d yugmastra -c \
  "DELETE FROM \"Session\";"

# 5. Clear Redis cache
docker exec yugmastra-redis redis-cli FLUSHALL
```

## Checklist

Before deploying to production:

- [ ] Generated unique JWT_SECRET (64 chars hex)
- [ ] Generated unique NEXTAUTH_SECRET (base64)
- [ ] Generated unique SESSION_SECRET (64 chars hex)
- [ ] Generated unique POSTGRES_PASSWORD (base64)
- [ ] Stored secrets in secure secret manager (not .env)
- [ ] Verified .env is in .gitignore
- [ ] Tested authentication with new secrets
- [ ] Documented secret rotation schedule
- [ ] Set up monitoring alerts for failed auth
- [ ] Configured automatic secret rotation (if available)
- [ ] Reviewed access controls on secret storage
- [ ] Backed up secrets in secure location
- [ ] Tested secret recovery procedure

## Additional Resources

- [OWASP Secret Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [NextAuth.js Security Best Practices](https://next-auth.js.org/configuration/options#secret)
- [Docker Secrets Documentation](https://docs.docker.com/engine/swarm/secrets/)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)

---

**Last Updated**: 2026-01-02
**Security Level**: Production-Ready
**Rotation Schedule**: 90 days (auth secrets), 180 days (database)
