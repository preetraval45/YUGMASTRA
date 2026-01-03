# Security Guidelines

## Environment Variables

### Production Deployment

**CRITICAL**: Never deploy to production with default secrets!

Before deploying to production, you MUST change the following environment variables:

1. **JWT_SECRET** - Used for authentication tokens
   ```bash
   # Generate a secure random secret (minimum 32 characters)
   openssl rand -hex 32
   ```

2. **POSTGRES_PASSWORD** - Database password
   ```bash
   # Generate a strong password
   openssl rand -base64 24
   ```

3. **SESSION_SECRET** - Session encryption key
   ```bash
   openssl rand -hex 32
   ```

### Setting Environment Variables

1. Create a `.env` file in the project root (NOT committed to git):
   ```bash
   cp .env.example .env
   ```

2. Update all secrets in `.env`:
   ```env
   JWT_SECRET=your-secure-random-secret-here
   POSTGRES_PASSWORD=your-secure-db-password-here
   SESSION_SECRET=your-secure-session-secret-here
   ```

3. Docker Compose will automatically load these variables

### Security Checklist

- [ ] All secrets changed from defaults
- [ ] `.env` file added to `.gitignore`
- [ ] Database uses strong password
- [ ] JWT secret is random and unique
- [ ] HTTPS enabled in production
- [ ] CORS origins configured for production domains only
- [ ] Rate limiting enabled
- [ ] Error messages don't leak sensitive information

## API Security

### Rate Limiting

The following endpoints have rate limiting enabled:

- **Login**: 5 attempts per minute per IP
- **Signup**: 3 attempts per minute per IP
- **Password Reset**: Default rate limit applies

### Authentication

- JWT tokens expire after 7 days
- Passwords are hashed using bcrypt with 12 rounds
- HttpOnly cookies prevent XSS attacks
- Secure cookies in production (HTTPS only)

### Database Security

- SQL injection protection via Prisma ORM
- Parameterized queries only
- Connection pooling with max connections limit
- Database credentials stored in environment variables

## Reporting Security Issues

If you discover a security vulnerability, please email:
- security@yugmastra.com (if available)
- Or create a private security advisory on GitHub

**Do NOT create public issues for security vulnerabilities.**

## Security Headers

The application implements the following security headers:

- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security` (in production)

## CORS Policy

CORS is configured to only allow requests from:
- Production domain
- Development localhost (only in dev mode)

## Regular Security Updates

- Keep all dependencies up to date
- Run `npm audit` regularly
- Monitor security advisories
- Apply patches promptly
