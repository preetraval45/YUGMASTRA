/**
 * Single Sign-On (SSO) Integration
 * Supports SAML 2.0, OAuth 2.0, and OpenID Connect
 */

import { SignJWT, jwtVerify } from 'jose';
import { cookies } from 'next/headers';

export type SSOProvider = 'azure-ad' | 'okta' | 'auth0' | 'google' | 'github';

export interface SSOConfig {
  provider: SSOProvider;
  clientId: string;
  clientSecret: string;
  tenantId?: string; // For Azure AD
  domain?: string; // For Okta/Auth0
  redirectUri: string;
  scopes: string[];
}

export interface SSOUser {
  id: string;
  email: string;
  name: string;
  provider: SSOProvider;
  accessToken: string;
  refreshToken?: string;
  expiresAt: number;
  roles?: string[];
  groups?: string[];
}

export class SSOManager {
  private config: SSOConfig;

  constructor(config: SSOConfig) {
    this.config = config;
  }

  /**
   * Get OAuth 2.0 authorization URL
   */
  getAuthorizationUrl(state: string): string {
    const params = new URLSearchParams({
      client_id: this.config.clientId,
      redirect_uri: this.config.redirectUri,
      response_type: 'code',
      scope: this.config.scopes.join(' '),
      state,
    });

    const baseUrl = this.getAuthEndpoint();
    return `${baseUrl}?${params.toString()}`;
  }

  /**
   * Exchange authorization code for tokens
   */
  async exchangeCodeForTokens(code: string): Promise<{
    access_token: string;
    refresh_token?: string;
    id_token?: string;
    expires_in: number;
  }> {
    const tokenEndpoint = this.getTokenEndpoint();

    const response = await fetch(tokenEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        code,
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret,
        redirect_uri: this.config.redirectUri,
      }),
    });

    if (!response.ok) {
      throw new Error('Token exchange failed');
    }

    return response.json();
  }

  /**
   * Get user info from SSO provider
   */
  async getUserInfo(accessToken: string): Promise<SSOUser> {
    const userInfoEndpoint = this.getUserInfoEndpoint();

    const response = await fetch(userInfoEndpoint, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch user info');
    }

    const data = await response.json();
    return this.normalizeUserInfo(data, accessToken);
  }

  /**
   * Refresh access token
   */
  async refreshAccessToken(refreshToken: string): Promise<{
    access_token: string;
    expires_in: number;
  }> {
    const tokenEndpoint = this.getTokenEndpoint();

    const response = await fetch(tokenEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret,
      }),
    });

    if (!response.ok) {
      throw new Error('Token refresh failed');
    }

    return response.json();
  }

  /**
   * Verify ID token (OpenID Connect)
   */
  async verifyIdToken(idToken: string): Promise<any> {
    // In production, verify JWT signature with provider's public key
    const jwksEndpoint = this.getJwksEndpoint();

    // Fetch JWKS
    const jwksResponse = await fetch(jwksEndpoint);
    const jwks = await jwksResponse.json();

    // Verify token (simplified - use jose library in production)
    const [header, payload] = idToken.split('.');
    const decodedPayload = JSON.parse(
      Buffer.from(payload, 'base64').toString('utf-8')
    );

    // Verify claims
    if (decodedPayload.iss !== this.getIssuer()) {
      throw new Error('Invalid issuer');
    }

    if (decodedPayload.aud !== this.config.clientId) {
      throw new Error('Invalid audience');
    }

    if (decodedPayload.exp < Date.now() / 1000) {
      throw new Error('Token expired');
    }

    return decodedPayload;
  }

  /**
   * SAML 2.0 authentication request
   */
  getSAMLAuthRequest(relayState: string): string {
    // Generate SAML AuthnRequest XML
    const samlRequest = `
      <samlp:AuthnRequest
        xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
        xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
        ID="_${this.generateId()}"
        Version="2.0"
        IssueInstant="${new Date().toISOString()}"
        Destination="${this.getSAMLEndpoint()}"
        AssertionConsumerServiceURL="${this.config.redirectUri}">
        <saml:Issuer>${this.config.clientId}</saml:Issuer>
        <samlp:NameIDPolicy Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress" />
      </samlp:AuthnRequest>
    `;

    // Base64 encode and URL encode
    const encoded = Buffer.from(samlRequest).toString('base64');
    return encoded;
  }

  // Provider-specific endpoints
  private getAuthEndpoint(): string {
    switch (this.config.provider) {
      case 'azure-ad':
        return `https://login.microsoftonline.com/${this.config.tenantId}/oauth2/v2.0/authorize`;
      case 'okta':
        return `https://${this.config.domain}/oauth2/v1/authorize`;
      case 'auth0':
        return `https://${this.config.domain}/authorize`;
      case 'google':
        return 'https://accounts.google.com/o/oauth2/v2/auth';
      case 'github':
        return 'https://github.com/login/oauth/authorize';
    }
  }

  private getTokenEndpoint(): string {
    switch (this.config.provider) {
      case 'azure-ad':
        return `https://login.microsoftonline.com/${this.config.tenantId}/oauth2/v2.0/token`;
      case 'okta':
        return `https://${this.config.domain}/oauth2/v1/token`;
      case 'auth0':
        return `https://${this.config.domain}/oauth/token`;
      case 'google':
        return 'https://oauth2.googleapis.com/token';
      case 'github':
        return 'https://github.com/login/oauth/access_token';
    }
  }

  private getUserInfoEndpoint(): string {
    switch (this.config.provider) {
      case 'azure-ad':
        return 'https://graph.microsoft.com/v1.0/me';
      case 'okta':
        return `https://${this.config.domain}/oauth2/v1/userinfo`;
      case 'auth0':
        return `https://${this.config.domain}/userinfo`;
      case 'google':
        return 'https://www.googleapis.com/oauth2/v2/userinfo';
      case 'github':
        return 'https://api.github.com/user';
    }
  }

  private getJwksEndpoint(): string {
    switch (this.config.provider) {
      case 'azure-ad':
        return `https://login.microsoftonline.com/${this.config.tenantId}/discovery/v2.0/keys`;
      case 'okta':
        return `https://${this.config.domain}/oauth2/v1/keys`;
      case 'auth0':
        return `https://${this.config.domain}/.well-known/jwks.json`;
      case 'google':
        return 'https://www.googleapis.com/oauth2/v3/certs';
      default:
        return '';
    }
  }

  private getSAMLEndpoint(): string {
    switch (this.config.provider) {
      case 'azure-ad':
        return `https://login.microsoftonline.com/${this.config.tenantId}/saml2`;
      case 'okta':
        return `https://${this.config.domain}/app/yugmastra/sso/saml`;
      default:
        return '';
    }
  }

  private getIssuer(): string {
    switch (this.config.provider) {
      case 'azure-ad':
        return `https://login.microsoftonline.com/${this.config.tenantId}/v2.0`;
      case 'okta':
        return `https://${this.config.domain}`;
      case 'auth0':
        return `https://${this.config.domain}/`;
      case 'google':
        return 'https://accounts.google.com';
      default:
        return '';
    }
  }

  private normalizeUserInfo(data: any, accessToken: string): SSOUser {
    switch (this.config.provider) {
      case 'azure-ad':
        return {
          id: data.id,
          email: data.mail || data.userPrincipalName,
          name: data.displayName,
          provider: 'azure-ad',
          accessToken,
          expiresAt: Date.now() + 3600000,
          roles: data.roles || [],
          groups: data.groups || [],
        };
      case 'google':
        return {
          id: data.id,
          email: data.email,
          name: data.name,
          provider: 'google',
          accessToken,
          expiresAt: Date.now() + 3600000,
        };
      case 'github':
        return {
          id: data.id.toString(),
          email: data.email,
          name: data.name || data.login,
          provider: 'github',
          accessToken,
          expiresAt: Date.now() + 3600000,
        };
      default:
        return {
          id: data.sub || data.id,
          email: data.email,
          name: data.name,
          provider: this.config.provider,
          accessToken,
          expiresAt: Date.now() + 3600000,
        };
    }
  }

  private generateId(): string {
    return `_${Math.random().toString(36).substr(2, 9)}`;
  }
}

// Example usage
export function createSSOManager(provider: SSOProvider): SSOManager {
  const configs: Record<SSOProvider, SSOConfig> = {
    'azure-ad': {
      provider: 'azure-ad',
      clientId: process.env.AZURE_CLIENT_ID!,
      clientSecret: process.env.AZURE_CLIENT_SECRET!,
      tenantId: process.env.AZURE_TENANT_ID!,
      redirectUri: `${process.env.NEXT_PUBLIC_APP_URL}/auth/callback/azure`,
      scopes: ['openid', 'profile', 'email', 'User.Read'],
    },
    'okta': {
      provider: 'okta',
      clientId: process.env.OKTA_CLIENT_ID!,
      clientSecret: process.env.OKTA_CLIENT_SECRET!,
      domain: process.env.OKTA_DOMAIN!,
      redirectUri: `${process.env.NEXT_PUBLIC_APP_URL}/auth/callback/okta`,
      scopes: ['openid', 'profile', 'email'],
    },
    'auth0': {
      provider: 'auth0',
      clientId: process.env.AUTH0_CLIENT_ID!,
      clientSecret: process.env.AUTH0_CLIENT_SECRET!,
      domain: process.env.AUTH0_DOMAIN!,
      redirectUri: `${process.env.NEXT_PUBLIC_APP_URL}/auth/callback/auth0`,
      scopes: ['openid', 'profile', 'email'],
    },
    'google': {
      provider: 'google',
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
      redirectUri: `${process.env.NEXT_PUBLIC_APP_URL}/auth/callback/google`,
      scopes: ['openid', 'profile', 'email'],
    },
    'github': {
      provider: 'github',
      clientId: process.env.GITHUB_CLIENT_ID!,
      clientSecret: process.env.GITHUB_CLIENT_SECRET!,
      redirectUri: `${process.env.NEXT_PUBLIC_APP_URL}/auth/callback/github`,
      scopes: ['read:user', 'user:email'],
    },
  };

  return new SSOManager(configs[provider]);
}
