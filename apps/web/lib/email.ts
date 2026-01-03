import nodemailer from 'nodemailer';

const transporter = nodemailer.createTransport({
  host: process.env.SMTP_HOST || 'smtp.gmail.com',
  port: parseInt(process.env.SMTP_PORT || '587'),
  secure: false,
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS,
  },
});

interface EmailOptions {
  to: string;
  subject: string;
  html: string;
  text?: string;
}

export async function sendEmail(options: EmailOptions): Promise<void> {
  if (!process.env.SMTP_USER || !process.env.SMTP_PASS) {
    console.warn('[Email] SMTP credentials not configured, skipping email');
    return;
  }

  try {
    await transporter.sendMail({
      from: `"YUGMĀSTRA" <${process.env.SMTP_USER}>`,
      to: options.to,
      subject: options.subject,
      html: options.html,
      text: options.text || options.html.replace(/<[^>]*>/g, ''),
    });
    console.log(`[Email] Sent to ${options.to}: ${options.subject}`);
  } catch (error) {
    console.error('[Email] Failed to send:', error);
    throw error;
  }
}

// Email templates
export const emailTemplates = {
  battleComplete: (battleId: string, redScore: number, blueScore: number) => ({
    subject: `Battle ${battleId} Complete`,
    html: `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h1 style="color: #3b82f6;">Battle Complete</h1>
        <p>Your co-evolution training battle has completed.</p>
        <div style="background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0;">
          <h2>Results:</h2>
          <p><strong>Battle ID:</strong> ${battleId}</p>
          <p><strong>Red Team Score:</strong> ${redScore}</p>
          <p><strong>Blue Team Score:</strong> ${blueScore}</p>
          <p><strong>Winner:</strong> ${redScore > blueScore ? 'Red Team' : 'Blue Team'}</p>
        </div>
        <p>View full details in your <a href="${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/dashboard">dashboard</a>.</p>
      </div>
    `,
  }),

  criticalAttack: (attackType: string, target: string, severity: string) => ({
    subject: `🚨 Critical Attack Detected: ${attackType}`,
    html: `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h1 style="color: #ef4444;">Critical Attack Detected</h1>
        <div style="background: #fee2e2; border-left: 4px solid #dc2626; padding: 20px; margin: 20px 0;">
          <h2 style="color: #dc2626;">Attack Details:</h2>
          <p><strong>Type:</strong> ${attackType}</p>
          <p><strong>Target:</strong> ${target}</p>
          <p><strong>Severity:</strong> ${severity.toUpperCase()}</p>
          <p><strong>Time:</strong> ${new Date().toLocaleString()}</p>
        </div>
        <p>Immediate action may be required. Check your <a href="${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/attacks">attacks dashboard</a>.</p>
      </div>
    `,
  }),

  weeklyReport: (stats: any) => ({
    subject: 'Your Weekly YUGMĀSTRA Report',
    html: `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h1 style="color: #3b82f6;">Weekly Activity Report</h1>
        <p>Here's your summary for the past week:</p>
        <div style="background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0;">
          <h2>Statistics:</h2>
          <p><strong>Total Battles:</strong> ${stats.battles}</p>
          <p><strong>Total Attacks:</strong> ${stats.attacks}</p>
          <p><strong>Detection Rate:</strong> ${stats.detectionRate}%</p>
          <p><strong>Avg Nash Equilibrium:</strong> ${stats.avgNash}</p>
        </div>
        <p>Keep up the great work! View more details in your <a href="${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/dashboard">dashboard</a>.</p>
      </div>
    `,
  }),

  welcomeEmail: (userName: string) => ({
    subject: 'Welcome to YUGMĀSTRA',
    html: `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h1 style="color: #3b82f6;">Welcome to YUGMĀSTRA!</h1>
        <p>Hi ${userName},</p>
        <p>Thank you for joining YUGMĀSTRA, the autonomous cyber defense platform powered by co-evolving AI agents.</p>
        <div style="background: #eff6ff; padding: 20px; border-radius: 8px; margin: 20px 0;">
          <h2>Get Started:</h2>
          <ul style="line-height: 1.8;">
            <li>Launch your first <a href="${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/live-battle">Live Battle</a></li>
            <li>Explore the <a href="${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/knowledge-graph">Knowledge Graph</a></li>
            <li>Customize your <a href="${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/settings">Settings</a></li>
          </ul>
        </div>
        <p>If you have any questions, feel free to reach out to our support team.</p>
        <p>Happy defending!</p>
      </div>
    `,
  }),

  passwordReset: (resetUrl: string, userName?: string) => ({
    subject: 'Password Reset Request - YUGMĀSTRA',
    html: `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0;">
          <h1>🛡️ YUGMĀSTRA</h1>
          <p style="margin: 0; font-size: 18px;">Password Reset Request</p>
        </div>
        <div style="background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px;">
          <p>Hello${userName ? ` ${userName}` : ''},</p>
          <p>We received a request to reset your password. Click the button below to create a new password:</p>
          <div style="text-align: center; margin: 30px 0;">
            <a href="${resetUrl}" style="display: inline-block; background: #667eea; color: white; padding: 15px 40px; text-decoration: none; border-radius: 5px; font-weight: bold;">Reset Password</a>
          </div>
          <p>Or copy and paste this link into your browser:</p>
          <p style="word-break: break-all; color: #667eea; background: white; padding: 10px; border-radius: 4px;">${resetUrl}</p>
          <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0;">
            <strong>⚠️ Security Notice:</strong>
            <ul style="margin: 10px 0; padding-left: 20px;">
              <li>This link will expire in 1 hour</li>
              <li>If you didn't request this reset, please ignore this email</li>
              <li>Never share this link with anyone</li>
            </ul>
          </div>
          <p style="color: #666; font-size: 14px; margin-top: 30px;">If you have any questions or concerns, please contact our support team.</p>
        </div>
        <div style="text-align: center; padding: 20px; color: #999; font-size: 12px;">
          <p>This is an automated message from YUGMĀSTRA. Please do not reply to this email.</p>
          <p>&copy; ${new Date().getFullYear()} YUGMĀSTRA. All rights reserved.</p>
        </div>
      </div>
    `,
  }),
};
