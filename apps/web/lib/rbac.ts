import { JWTPayload } from './auth';

export enum Role {
  ADMIN = 'admin',
  ANALYST = 'analyst',
  USER = 'user',
}

export enum Permission {
  // Battle permissions
  CREATE_BATTLE = 'battle:create',
  VIEW_BATTLE = 'battle:view',
  UPDATE_BATTLE = 'battle:update',
  DELETE_BATTLE = 'battle:delete',

  // Attack/Defense permissions
  VIEW_ATTACKS = 'attacks:view',
  CREATE_ATTACK = 'attacks:create',
  VIEW_DEFENSES = 'defenses:view',
  CREATE_DEFENSE = 'defenses:create',

  // User management
  MANAGE_USERS = 'users:manage',
  VIEW_USERS = 'users:view',

  // Settings
  MANAGE_SETTINGS = 'settings:manage',
  VIEW_SETTINGS = 'settings:view',

  // System
  VIEW_SYSTEM_STATS = 'system:stats',
  MANAGE_SYSTEM = 'system:manage',

  // Export
  EXPORT_DATA = 'data:export',
}

const rolePermissions: Record<Role, Permission[]> = {
  [Role.ADMIN]: [
    // Admins have all permissions
    ...Object.values(Permission),
  ],
  [Role.ANALYST]: [
    // Analysts can view and create battles/attacks/defenses
    Permission.CREATE_BATTLE,
    Permission.VIEW_BATTLE,
    Permission.UPDATE_BATTLE,
    Permission.VIEW_ATTACKS,
    Permission.CREATE_ATTACK,
    Permission.VIEW_DEFENSES,
    Permission.CREATE_DEFENSE,
    Permission.VIEW_SETTINGS,
    Permission.VIEW_SYSTEM_STATS,
    Permission.EXPORT_DATA,
  ],
  [Role.USER]: [
    // Regular users have limited permissions
    Permission.VIEW_BATTLE,
    Permission.VIEW_ATTACKS,
    Permission.VIEW_DEFENSES,
    Permission.VIEW_SETTINGS,
  ],
};

export function hasPermission(user: JWTPayload | null, permission: Permission): boolean {
  if (!user) return false;

  const userRole = user.role as Role;
  const permissions = rolePermissions[userRole] || [];

  return permissions.includes(permission);
}

export function hasAnyPermission(user: JWTPayload | null, permissions: Permission[]): boolean {
  return permissions.some((permission) => hasPermission(user, permission));
}

export function hasAllPermissions(user: JWTPayload | null, permissions: Permission[]): boolean {
  return permissions.every((permission) => hasPermission(user, permission));
}

export function requirePermission(user: JWTPayload | null, permission: Permission): void {
  if (!hasPermission(user, permission)) {
    throw new Error(`Insufficient permissions. Required: ${permission}`);
  }
}

export function requireRole(user: JWTPayload | null, role: Role): void {
  if (!user || user.role !== role) {
    throw new Error(`Insufficient role. Required: ${role}`);
  }
}
