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
};
