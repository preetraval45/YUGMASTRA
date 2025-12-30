import { z } from 'zod';

export const userLoginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
});

export const userSignupSchema = z.object({
  name: z.string().min(2, 'Name must be at least 2 characters'),
  email: z.string().email('Invalid email address'),
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
    .regex(/[0-9]/, 'Password must contain at least one number')
    .regex(/[^A-Za-z0-9]/, 'Password must contain at least one special character'),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ['confirmPassword'],
});

export const updateProfileSchema = z.object({
  name: z.string().min(2).optional(),
  email: z.string().email().optional(),
  bio: z.string().max(500).optional(),
  organization: z.string().max(100).optional(),
});

export const aiChatRequestSchema = z.object({
  message: z.string().min(1, 'Message cannot be empty').max(5000, 'Message too long'),
  mode: z.enum(['red-team', 'blue-team', 'evolution'], {
    errorMap: () => ({ message: 'Invalid AI mode' }),
  }),
  history: z.array(z.object({
    role: z.enum(['user', 'assistant', 'system']),
    content: z.string(),
  })).optional(),
  context: z.record(z.any()).optional(),
  stream: z.boolean().optional(),
});

export const aiTrainingRequestSchema = z.object({
  dataset_path: z.string().min(1, 'Dataset path required'),
  model_type: z.string().min(1, 'Model type required'),
  epochs: z.number().int().min(1).max(100).default(3),
  batch_size: z.number().int().min(1).max(128).default(8),
});

export const attackPatternSchema = z.object({
  name: z.string().min(1, 'Attack name required'),
  description: z.string().optional(),
  technique: z.string().min(1, 'Technique required'),
  tactics: z.array(z.string()).min(1, 'At least one tactic required'),
  severity: z.enum(['low', 'medium', 'high', 'critical']),
  indicators: z.array(z.string()).optional(),
  mitreId: z.string().optional(),
});

export const defenseStrategySchema = z.object({
  name: z.string().min(1, 'Defense name required'),
  description: z.string().optional(),
  type: z.enum(['preventive', 'detective', 'responsive']),
  effectiveness: z.number().min(0).max(1).optional(),
  cost: z.number().min(0).optional(),
  requirements: z.array(z.string()).optional(),
});

export const graphNodeSchema = z.object({
  id: z.string().min(1, 'Node ID required'),
  type: z.enum(['attack', 'defense', 'vulnerability', 'asset', 'threat', 'mitigation']),
  properties: z.record(z.any()),
});

export const graphEdgeSchema = z.object({
  source: z.string().min(1, 'Source node required'),
  target: z.string().min(1, 'Target node required'),
  relationship: z.string().min(1, 'Relationship type required'),
  weight: z.number().min(0).max(1).optional(),
  properties: z.record(z.any()).optional(),
});

export const vulnerabilitySchema = z.object({
  cveId: z.string().regex(/^CVE-\d{4}-\d{4,7}$/, 'Invalid CVE ID format').optional(),
  title: z.string().min(1, 'Title required'),
  description: z.string().min(1, 'Description required'),
  severity: z.enum(['low', 'medium', 'high', 'critical']),
  cvssScore: z.number().min(0).max(10).optional(),
  affectedSystems: z.array(z.string()).optional(),
  exploitAvailable: z.boolean().optional(),
  patchAvailable: z.boolean().optional(),
});

export const siemRuleSchema = z.object({
  title: z.string().min(1, 'Title required'),
  description: z.string().min(1, 'Description required'),
  severity: z.enum(['low', 'medium', 'high', 'critical']),
  format: z.enum(['sigma', 'splunk', 'elastic', 'suricata', 'snort', 'yara']),
  tags: z.array(z.string()).optional(),
  mitreAttack: z.array(z.string()).optional(),
  falsePositives: z.array(z.string()).optional(),
});

export const battleConfigSchema = z.object({
  duration: z.number().int().min(60).max(3600, 'Duration must be between 60s and 1 hour'),
  attackFrequency: z.number().min(0.1).max(10, 'Invalid attack frequency'),
  defenseStrategy: z.enum(['reactive', 'proactive', 'hybrid']),
  difficulty: z.enum(['easy', 'medium', 'hard', 'expert']),
  enableLearning: z.boolean().optional(),
});

export const notificationSettingsSchema = z.object({
  emailNotifications: z.boolean(),
  pushNotifications: z.boolean(),
  slackIntegration: z.boolean().optional(),
  webhookUrl: z.string().url('Invalid webhook URL').optional(),
});

export const securitySettingsSchema = z.object({
  twoFactorAuth: z.boolean(),
  sessionTimeout: z.number().int().min(5).max(1440), // 5 minutes to 24 hours
  ipWhitelist: z.array(z.string().ip()).optional(),
  apiKeyRotationDays: z.number().int().min(1).max(365).optional(),
});

export const exportRequestSchema = z.object({
  format: z.enum(['json', 'csv', 'pdf', 'excel']),
  data: z.enum(['attacks', 'defenses', 'battles', 'vulnerabilities', 'graph']),
  dateRange: z.object({
    from: z.string().datetime().optional(),
    to: z.string().datetime().optional(),
  }).optional(),
  filters: z.record(z.any()).optional(),
});

export type UserLogin = z.infer<typeof userLoginSchema>;
export type UserSignup = z.infer<typeof userSignupSchema>;
export type UpdateProfile = z.infer<typeof updateProfileSchema>;
export type AIChatRequest = z.infer<typeof aiChatRequestSchema>;
export type AITrainingRequest = z.infer<typeof aiTrainingRequestSchema>;
export type AttackPattern = z.infer<typeof attackPatternSchema>;
export type DefenseStrategy = z.infer<typeof defenseStrategySchema>;
export type GraphNode = z.infer<typeof graphNodeSchema>;
export type GraphEdge = z.infer<typeof graphEdgeSchema>;
export type Vulnerability = z.infer<typeof vulnerabilitySchema>;
export type SIEMRule = z.infer<typeof siemRuleSchema>;
export type BattleConfig = z.infer<typeof battleConfigSchema>;
export type NotificationSettings = z.infer<typeof notificationSettingsSchema>;
export type SecuritySettings = z.infer<typeof securitySettingsSchema>;
export type ExportRequest = z.infer<typeof exportRequestSchema>;

export function validateData<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; errors: z.ZodError } {
  try {
    const validData = schema.parse(data);
    return { success: true, data: validData };
  } catch (error) {
    if (error instanceof z.ZodError) {
      return { success: false, errors: error };
    }
    throw error;
  }
}

export function formatValidationErrors(error: z.ZodError): string[] {
  return error.errors.map((err) => {
    const path = err.path.join('.');
    return `${path ? path + ': ' : ''}${err.message}`;
  });
}
