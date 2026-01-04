/**
 * Role-Based Access Control (RBAC) with Attribute-Based Extensions
 * Manages permissions, roles, and access control policies
 */

export enum Role {
  ADMIN = 'admin',
  SECURITY_ANALYST = 'security_analyst',
  RED_TEAM = 'red_team',
  BLUE_TEAM = 'blue_team',
  SOC_MANAGER = 'soc_manager',
  THREAT_HUNTER = 'threat_hunter',
  INCIDENT_RESPONDER = 'incident_responder',
  VIEWER = 'viewer',
  GUEST = 'guest',
}

export enum Permission {
  // System permissions
  SYSTEM_ADMIN = 'system:admin',
  SYSTEM_SETTINGS = 'system:settings',

  // AI model permissions
  AI_TRAIN = 'ai:train',
  AI_DEPLOY = 'ai:deploy',
  AI_INFERENCE = 'ai:inference',
  AI_VIEW = 'ai:view',

  // Attack simulation permissions
  ATTACK_CREATE = 'attack:create',
  ATTACK_RUN = 'attack:run',
  ATTACK_VIEW = 'attack:view',
  ATTACK_DELETE = 'attack:delete',

  // Defense permissions
  DEFENSE_CREATE = 'defense:create',
  DEFENSE_DEPLOY = 'defense:deploy',
  DEFENSE_VIEW = 'defense:view',

  // Incident permissions
  INCIDENT_CREATE = 'incident:create',
  INCIDENT_MANAGE = 'incident:manage',
  INCIDENT_VIEW = 'incident:view',
  INCIDENT_CLOSE = 'incident:close',

  // Threat intelligence permissions
  THREAT_INTEL_READ = 'threat_intel:read',
  THREAT_INTEL_WRITE = 'threat_intel:write',
  THREAT_INTEL_SHARE = 'threat_intel:share',

  // SIEM permissions
  SIEM_QUERY = 'siem:query',
  SIEM_RULES_CREATE = 'siem:rules:create',
  SIEM_RULES_EDIT = 'siem:rules:edit',

  // Zero-day permissions
  ZERO_DAY_DISCOVER = 'zero_day:discover',
  ZERO_DAY_EXPLOIT = 'zero_day:exploit',
  ZERO_DAY_REPORT = 'zero_day:report',

  // User management
  USERS_VIEW = 'users:view',
  USERS_MANAGE = 'users:manage',
  USERS_DELETE = 'users:delete',
}

// Role to permissions mapping
const rolePermissions: Record<Role, Permission[]> = {
  [Role.ADMIN]: Object.values(Permission), // All permissions

  [Role.SECURITY_ANALYST]: [
    Permission.AI_VIEW,
    Permission.AI_INFERENCE,
    Permission.ATTACK_VIEW,
    Permission.DEFENSE_VIEW,
    Permission.INCIDENT_VIEW,
    Permission.INCIDENT_CREATE,
    Permission.THREAT_INTEL_READ,
    Permission.THREAT_INTEL_WRITE,
    Permission.SIEM_QUERY,
    Permission.SIEM_RULES_CREATE,
    Permission.ZERO_DAY_DISCOVER,
    Permission.ZERO_DAY_REPORT,
  ],

  [Role.RED_TEAM]: [
    Permission.AI_VIEW,
    Permission.AI_TRAIN,
    Permission.ATTACK_CREATE,
    Permission.ATTACK_RUN,
    Permission.ATTACK_VIEW,
    Permission.DEFENSE_VIEW,
    Permission.THREAT_INTEL_READ,
    Permission.ZERO_DAY_DISCOVER,
    Permission.ZERO_DAY_EXPLOIT,
  ],

  [Role.BLUE_TEAM]: [
    Permission.AI_VIEW,
    Permission.AI_TRAIN,
    Permission.DEFENSE_CREATE,
    Permission.DEFENSE_DEPLOY,
    Permission.DEFENSE_VIEW,
    Permission.ATTACK_VIEW,
    Permission.INCIDENT_CREATE,
    Permission.INCIDENT_MANAGE,
    Permission.THREAT_INTEL_READ,
    Permission.SIEM_QUERY,
    Permission.SIEM_RULES_CREATE,
    Permission.SIEM_RULES_EDIT,
  ],

  [Role.SOC_MANAGER]: [
    Permission.AI_VIEW,
    Permission.AI_DEPLOY,
    Permission.ATTACK_VIEW,
    Permission.DEFENSE_VIEW,
    Permission.DEFENSE_DEPLOY,
    Permission.INCIDENT_VIEW,
    Permission.INCIDENT_MANAGE,
    Permission.INCIDENT_CLOSE,
    Permission.THREAT_INTEL_READ,
    Permission.THREAT_INTEL_WRITE,
    Permission.THREAT_INTEL_SHARE,
    Permission.SIEM_QUERY,
    Permission.SIEM_RULES_CREATE,
    Permission.SIEM_RULES_EDIT,
    Permission.USERS_VIEW,
  ],

  [Role.THREAT_HUNTER]: [
    Permission.AI_INFERENCE,
    Permission.ATTACK_VIEW,
    Permission.DEFENSE_VIEW,
    Permission.INCIDENT_VIEW,
    Permission.INCIDENT_CREATE,
    Permission.THREAT_INTEL_READ,
    Permission.THREAT_INTEL_WRITE,
    Permission.SIEM_QUERY,
    Permission.ZERO_DAY_DISCOVER,
  ],

  [Role.INCIDENT_RESPONDER]: [
    Permission.AI_INFERENCE,
    Permission.ATTACK_VIEW,
    Permission.DEFENSE_VIEW,
    Permission.DEFENSE_CREATE,
    Permission.INCIDENT_CREATE,
    Permission.INCIDENT_MANAGE,
    Permission.INCIDENT_CLOSE,
    Permission.THREAT_INTEL_READ,
    Permission.SIEM_QUERY,
  ],

  [Role.VIEWER]: [
    Permission.AI_VIEW,
    Permission.ATTACK_VIEW,
    Permission.DEFENSE_VIEW,
    Permission.INCIDENT_VIEW,
    Permission.THREAT_INTEL_READ,
  ],

  [Role.GUEST]: [
    Permission.AI_VIEW,
    Permission.ATTACK_VIEW,
  ],
};

export interface User {
  id: string;
  email: string;
  roles: Role[];
  attributes?: Record<string, any>; // For ABAC
}

export interface AccessContext {
  user: User;
  resource?: string;
  action?: string;
  environment?: Record<string, any>; // Time, IP, location, etc.
}

export class RBACManager {
  /**
   * Check if user has specific permission
   */
  static hasPermission(user: User, permission: Permission): boolean {
    // Admin has all permissions
    if (user.roles.includes(Role.ADMIN)) {
      return true;
    }

    // Check each role's permissions
    for (const role of user.roles) {
      const permissions = rolePermissions[role];
      if (permissions.includes(permission)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Check if user has any of the specified permissions
   */
  static hasAnyPermission(user: User, permissions: Permission[]): boolean {
    return permissions.some((perm) => this.hasPermission(user, perm));
  }

  /**
   * Check if user has all of the specified permissions
   */
  static hasAllPermissions(user: User, permissions: Permission[]): boolean {
    return permissions.every((perm) => this.hasPermission(user, perm));
  }

  /**
   * Get all permissions for user
   */
  static getUserPermissions(user: User): Permission[] {
    const permissions = new Set<Permission>();

    for (const role of user.roles) {
      const rolePerms = rolePermissions[role] || [];
      rolePerms.forEach((perm) => permissions.add(perm));
    }

    return Array.from(permissions);
  }

  /**
   * Check if user can access resource
   * Combines RBAC with ABAC (Attribute-Based Access Control)
   */
  static canAccess(context: AccessContext): boolean {
    const { user, resource, action, environment } = context;

    // 1. Check RBAC permissions
    if (!action) return false;

    const permission = `${resource}:${action}` as Permission;
    if (!this.hasPermission(user, permission)) {
      return false;
    }

    // 2. Apply ABAC rules (attribute-based constraints)
    if (environment) {
      // Time-based access control
      if (environment.time) {
        const hour = new Date(environment.time).getHours();
        // Example: Red Team can only run attacks during business hours
        if (
          user.roles.includes(Role.RED_TEAM) &&
          action === 'run' &&
          (hour < 9 || hour > 17)
        ) {
          return false;
        }
      }

      // IP-based access control
      if (environment.ip && user.attributes?.allowedIPs) {
        if (!user.attributes.allowedIPs.includes(environment.ip)) {
          return false;
        }
      }

      // Classification-based access (data sensitivity)
      if (environment.classification) {
        const userClearance = user.attributes?.clearanceLevel || 0;
        const requiredClearance = environment.classification;
        if (userClearance < requiredClearance) {
          return false;
        }
      }
    }

    return true;
  }

  /**
   * Enforce permission check (throws error if denied)
   */
  static enforce(user: User, permission: Permission): void {
    if (!this.hasPermission(user, permission)) {
      throw new Error(
        `Access denied: User ${user.email} does not have permission ${permission}`
      );
    }
  }

  /**
   * Check if user has role
   */
  static hasRole(user: User, role: Role): boolean {
    return user.roles.includes(role);
  }

  /**
   * Check if user has any of the specified roles
   */
  static hasAnyRole(user: User, roles: Role[]): boolean {
    return roles.some((role) => this.hasRole(user, role));
  }

  /**
   * Add role to user (returns new user object)
   */
  static addRole(user: User, role: Role): User {
    if (user.roles.includes(role)) {
      return user;
    }

    return {
      ...user,
      roles: [...user.roles, role],
    };
  }

  /**
   * Remove role from user (returns new user object)
   */
  static removeRole(user: User, role: Role): User {
    return {
      ...user,
      roles: user.roles.filter((r) => r !== role),
    };
  }
}

// React Hook for permission checking
export function usePermissions(user: User) {
  return {
    hasPermission: (permission: Permission) =>
      RBACManager.hasPermission(user, permission),
    hasAnyPermission: (permissions: Permission[]) =>
      RBACManager.hasAnyPermission(user, permissions),
    hasAllPermissions: (permissions: Permission[]) =>
      RBACManager.hasAllPermissions(user, permissions),
    hasRole: (role: Role) => RBACManager.hasRole(user, role),
    hasAnyRole: (roles: Role[]) => RBACManager.hasAnyRole(user, roles),
    permissions: RBACManager.getUserPermissions(user),
    canAccess: (context: Omit<AccessContext, 'user'>) =>
      RBACManager.canAccess({ ...context, user }),
  };
}

// Middleware helper for API routes
export function requirePermission(permission: Permission) {
  return (user: User) => {
    RBACManager.enforce(user, permission);
  };
}

// Example usage
export const exampleUser: User = {
  id: '1',
  email: 'analyst@yugmastra.com',
  roles: [Role.SECURITY_ANALYST, Role.THREAT_HUNTER],
  attributes: {
    clearanceLevel: 3,
    allowedIPs: ['10.0.0.0/8', '192.168.1.0/24'],
    department: 'Security Operations',
  },
};
