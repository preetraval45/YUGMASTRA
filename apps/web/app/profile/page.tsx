'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import {
  User,
  Mail,
  Phone,
  MapPin,
  Briefcase,
  Calendar,
  Shield,
  Key,
  Bell,
  Lock,
  Eye,
  Edit2,
  Save,
  X,
  Camera,
  Award,
  Target,
  TrendingUp,
  Activity
} from 'lucide-react';

export default function ProfilePage() {
  const [isEditing, setIsEditing] = useState(false);
  const [profile, setProfile] = useState({
    name: 'Preet Raval',
    role: 'System Owner',
    email: 'preet@yugmastra.ai',
    phone: '+1 (555) 123-4567',
    location: 'San Francisco, CA',
    company: 'YUGMĀSTRA',
    joined: 'January 2024',
    bio: 'Cybersecurity expert specializing in AI-powered threat detection and simulation. Passionate about building next-generation security platforms.',
  });

  const stats = [
    { label: 'Simulations Run', value: '1,247', icon: Activity, color: 'text-blue-500' },
    { label: 'Threats Detected', value: '3,891', icon: Shield, color: 'text-red-500' },
    { label: 'Success Rate', value: '94.2%', icon: Target, color: 'text-green-500' },
    { label: 'Total Score', value: '48,392', icon: TrendingUp, color: 'text-purple-500' },
  ];

  const achievements = [
    { name: 'First Simulation', date: 'Jan 15, 2024', icon: '🚀' },
    { name: 'APT Hunter', date: 'Feb 3, 2024', icon: '🎯' },
    { name: 'Defense Master', date: 'Mar 12, 2024', icon: '🛡️' },
    { name: 'Threat Analyst', date: 'Apr 8, 2024', icon: '🔍' },
    { name: '1000 Simulations', date: 'May 20, 2024', icon: '🏆' },
    { name: 'Elite Operator', date: 'Jun 15, 2024', icon: '⭐' },
  ];

  const recentActivity = [
    { action: 'Completed APT29 simulation', time: '2 hours ago', type: 'success' },
    { action: 'Detected ransomware attack', time: '5 hours ago', type: 'warning' },
    { action: 'Updated defense rules', time: '1 day ago', type: 'info' },
    { action: 'Analyzed threat intelligence', time: '2 days ago', type: 'info' },
    { action: 'Blocked 12 attack attempts', time: '3 days ago', type: 'success' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5 p-6 pt-32">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              Profile Settings
            </h1>
            <p className="text-muted-foreground mt-2">
              Manage your account settings and preferences
            </p>
          </div>
          <Button
            onClick={() => setIsEditing(!isEditing)}
            className="gap-2"
            variant={isEditing ? "outline" : "default"}
          >
            {isEditing ? (
              <>
                <X className="h-4 w-4" />
                Cancel
              </>
            ) : (
              <>
                <Edit2 className="h-4 w-4" />
                Edit Profile
              </>
            )}
          </Button>
        </div>

        <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 flex items-start gap-3">
          <User className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong className="text-foreground">What this page does:</strong> Manage your YUGMASTRA user profile and account information. Update personal details (name, email, phone, location), view performance statistics (simulations run, threats detected, success rate), track achievements and badges, review recent activity, and configure security settings (2FA, password, active sessions). Monitor your journey as a cyber defense operator.
            </p>
          </div>
        </div>

        {/* Profile Card */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-xl">
          <CardContent className="p-8">
            <div className="flex flex-col md:flex-row gap-8">
              {/* Avatar */}
              <div className="flex flex-col items-center gap-4">
                <div className="relative group">
                  <div className="h-32 w-32 rounded-full bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center ring-4 ring-background shadow-2xl">
                    <User className="h-16 w-16 text-white" />
                  </div>
                  {isEditing && (
                    <button className="absolute bottom-0 right-0 h-10 w-10 rounded-full bg-primary text-primary-foreground flex items-center justify-center shadow-lg hover:scale-110 transition-transform">
                      <Camera className="h-5 w-5" />
                    </button>
                  )}
                </div>
                <Badge className="bg-primary/10 text-primary border-primary/20">
                  System Owner
                </Badge>
              </div>

              {/* Profile Info */}
              <div className="flex-1 space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <label className="text-sm font-medium text-muted-foreground flex items-center gap-2 mb-2">
                      <User className="h-4 w-4" />
                      Full Name
                    </label>
                    {isEditing ? (
                      <Input
                        value={profile.name}
                        onChange={(e) => setProfile({ ...profile, name: e.target.value })}
                        className="bg-background/50"
                      />
                    ) : (
                      <p className="text-lg font-semibold text-foreground">{profile.name}</p>
                    )}
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground flex items-center gap-2 mb-2">
                      <Mail className="h-4 w-4" />
                      Email Address
                    </label>
                    {isEditing ? (
                      <Input
                        type="email"
                        value={profile.email}
                        onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                        className="bg-background/50"
                      />
                    ) : (
                      <p className="text-lg font-semibold text-foreground">{profile.email}</p>
                    )}
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground flex items-center gap-2 mb-2">
                      <Phone className="h-4 w-4" />
                      Phone Number
                    </label>
                    {isEditing ? (
                      <Input
                        type="tel"
                        value={profile.phone}
                        onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
                        className="bg-background/50"
                      />
                    ) : (
                      <p className="text-lg font-semibold text-foreground">{profile.phone}</p>
                    )}
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground flex items-center gap-2 mb-2">
                      <MapPin className="h-4 w-4" />
                      Location
                    </label>
                    {isEditing ? (
                      <Input
                        value={profile.location}
                        onChange={(e) => setProfile({ ...profile, location: e.target.value })}
                        className="bg-background/50"
                      />
                    ) : (
                      <p className="text-lg font-semibold text-foreground">{profile.location}</p>
                    )}
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground flex items-center gap-2 mb-2">
                      <Briefcase className="h-4 w-4" />
                      Company
                    </label>
                    <p className="text-lg font-semibold text-foreground">{profile.company}</p>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-muted-foreground flex items-center gap-2 mb-2">
                      <Calendar className="h-4 w-4" />
                      Joined
                    </label>
                    <p className="text-lg font-semibold text-foreground">{profile.joined}</p>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium text-muted-foreground mb-2 block">
                    Bio
                  </label>
                  {isEditing ? (
                    <textarea
                      value={profile.bio}
                      onChange={(e) => setProfile({ ...profile, bio: e.target.value })}
                      className="w-full min-h-[100px] px-4 py-3 rounded-xl border border-border bg-background/50 text-foreground resize-none focus:ring-2 focus:ring-primary focus:border-transparent"
                    />
                  ) : (
                    <p className="text-foreground leading-relaxed">{profile.bio}</p>
                  )}
                </div>

                {isEditing && (
                  <div className="flex gap-3">
                    <Button className="gap-2">
                      <Save className="h-4 w-4" />
                      Save Changes
                    </Button>
                    <Button variant="outline" onClick={() => setIsEditing(false)}>
                      Cancel
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <Card key={index} className="border-border/50 bg-card/50 backdrop-blur-xl shadow-lg hover:shadow-xl transition-all hover:-translate-y-1">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className={`h-12 w-12 rounded-xl bg-gradient-to-br from-${stat.color}/20 to-${stat.color}/5 flex items-center justify-center`}>
                    <stat.icon className={`h-6 w-6 ${stat.color}`} />
                  </div>
                </div>
                <div className="text-3xl font-bold text-foreground mb-1">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Achievements */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-foreground">
                <Award className="h-5 w-5 text-yellow-500" />
                Achievements
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {achievements.map((achievement, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-4 p-4 rounded-xl bg-accent/30 hover:bg-accent/50 transition-colors border border-border/30"
                  >
                    <div className="text-3xl">{achievement.icon}</div>
                    <div className="flex-1">
                      <p className="font-semibold text-foreground">{achievement.name}</p>
                      <p className="text-sm text-muted-foreground">{achievement.date}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recent Activity */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-foreground">
                <Activity className="h-5 w-5 text-blue-500" />
                Recent Activity
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentActivity.map((activity, index) => (
                  <div
                    key={index}
                    className="flex items-start gap-4 p-4 rounded-xl bg-accent/30 hover:bg-accent/50 transition-colors border border-border/30"
                  >
                    <div className={`h-2 w-2 rounded-full mt-2 ${
                      activity.type === 'success' ? 'bg-green-500' :
                      activity.type === 'warning' ? 'bg-yellow-500' :
                      'bg-blue-500'
                    }`} />
                    <div className="flex-1">
                      <p className="font-medium text-foreground">{activity.action}</p>
                      <p className="text-sm text-muted-foreground">{activity.time}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Security Settings */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-xl">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-foreground">
              <Shield className="h-5 w-5 text-green-500" />
              Security & Privacy
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="flex items-center justify-between p-4 rounded-xl bg-accent/30 border border-border/30">
                <div className="flex items-center gap-3">
                  <Key className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium text-foreground">Two-Factor Authentication</p>
                    <p className="text-sm text-muted-foreground">Extra security for your account</p>
                  </div>
                </div>
                <Badge className="bg-green-500/10 text-green-500 border-green-500/20">Enabled</Badge>
              </div>

              <div className="flex items-center justify-between p-4 rounded-xl bg-accent/30 border border-border/30">
                <div className="flex items-center gap-3">
                  <Bell className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium text-foreground">Email Notifications</p>
                    <p className="text-sm text-muted-foreground">Receive security alerts</p>
                  </div>
                </div>
                <Badge className="bg-blue-500/10 text-blue-500 border-blue-500/20">Active</Badge>
              </div>

              <div className="flex items-center justify-between p-4 rounded-xl bg-accent/30 border border-border/30">
                <div className="flex items-center gap-3">
                  <Lock className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium text-foreground">Password</p>
                    <p className="text-sm text-muted-foreground">Last changed 30 days ago</p>
                  </div>
                </div>
                <Button variant="outline" size="sm">Change</Button>
              </div>

              <div className="flex items-center justify-between p-4 rounded-xl bg-accent/30 border border-border/30">
                <div className="flex items-center gap-3">
                  <Eye className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium text-foreground">Active Sessions</p>
                    <p className="text-sm text-muted-foreground">3 devices logged in</p>
                  </div>
                </div>
                <Button variant="outline" size="sm">Manage</Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
