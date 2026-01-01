'use client';

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Eye, EyeOff, Mail, Lock, User, ArrowRight, Shield, CheckCircle2 } from 'lucide-react';

export default function AuthPage() {
  const [showPassword, setShowPassword] = useState(false);
  const [activeTab, setActiveTab] = useState('login');

  return (
    <div className="min-h-screen flex">
      {/* Left Side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-12 flex-col justify-between relative overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0" style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          }}/>
        </div>

        {/* Logo */}
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-8">
            <div className="h-14 w-14 rounded-2xl bg-white/20 backdrop-blur-xl flex items-center justify-center">
              <Shield className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white leading-[1.2]">YUGMĀSTRA</h1>
              <p className="text-white/80 text-xs font-medium tracking-wider uppercase">AI Cyber Warfare</p>
            </div>
          </div>
        </div>

        {/* Features */}
        <div className="relative z-10 space-y-6">
          <h2 className="text-4xl font-bold text-white leading-[1.2] mb-8">
            Next-Gen Cyber Defense Platform
          </h2>

          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="mt-1 h-6 w-6 rounded-full bg-white/20 backdrop-blur-xl flex items-center justify-center flex-shrink-0">
                <CheckCircle2 className="h-4 w-4 text-white" />
              </div>
              <div>
                <h3 className="text-white font-semibold text-lg">AI-Powered Simulations</h3>
                <p className="text-white/70 text-sm">Real-time adversary-defender co-evolution with advanced ML models</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="mt-1 h-6 w-6 rounded-full bg-white/20 backdrop-blur-xl flex items-center justify-center flex-shrink-0">
                <CheckCircle2 className="h-4 w-4 text-white" />
              </div>
              <div>
                <h3 className="text-white font-semibold text-lg">Autonomous Defense</h3>
                <p className="text-white/70 text-sm">Self-learning systems that adapt to emerging threats</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="mt-1 h-6 w-6 rounded-full bg-white/20 backdrop-blur-xl flex items-center justify-center flex-shrink-0">
                <CheckCircle2 className="h-4 w-4 text-white" />
              </div>
              <div>
                <h3 className="text-white font-semibold text-lg">Enterprise Ready</h3>
                <p className="text-white/70 text-sm">Scalable infrastructure for organizations of any size</p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="relative z-10 text-white/60 text-sm">
          © 2025 YUGMĀSTRA. All rights reserved.
        </div>
      </div>

      {/* Right Side - Auth Forms */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8 bg-background">
        <div className="w-full max-w-md">
          {/* Mobile Logo */}
          <div className="lg:hidden flex items-center gap-2.5 mb-8">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center">
              <Shield className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">
                YUGMĀSTRA
              </h1>
              <p className="text-[8px] text-muted-foreground tracking-wider uppercase">AI Cyber Warfare</p>
            </div>
          </div>

          <Tabs defaultValue="login" className="w-full" onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3 bg-muted/50 p-1 rounded-xl">
              <TabsTrigger value="login" className="rounded-lg data-[state=active]:bg-background data-[state=active]:shadow-sm">
                Login
              </TabsTrigger>
              <TabsTrigger value="signup" className="rounded-lg data-[state=active]:bg-background data-[state=active]:shadow-sm">
                Sign Up
              </TabsTrigger>
              <TabsTrigger value="reset" className="rounded-lg data-[state=active]:bg-background data-[state=active]:shadow-sm">
                Reset
              </TabsTrigger>
            </TabsList>

            {/* Login Tab */}
            <TabsContent value="login" className="mt-6 space-y-4">
              <div>
                <h2 className="text-2xl font-bold text-foreground">Welcome Back</h2>
                <p className="text-muted-foreground text-sm mt-1">Enter your credentials to access your account</p>
              </div>

              <form className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Email</label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="email"
                      placeholder="name@example.com"
                      className="pl-10 h-11 bg-background border-border/60"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Password</label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type={showPassword ? 'text' : 'password'}
                      placeholder="••••••••"
                      className="pl-10 pr-10 h-11 bg-background border-border/60"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <input type="checkbox" className="rounded border-border/60" />
                    <span className="text-muted-foreground">Remember me</span>
                  </label>
                  <button
                    type="button"
                    onClick={() => setActiveTab('reset')}
                    className="text-sm text-primary hover:text-primary/80 font-medium"
                  >
                    Forgot password?
                  </button>
                </div>

                <button
                  type="submit"
                  className="w-full h-11 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 hover:from-blue-600 hover:via-purple-600 hover:to-pink-600 text-white font-semibold rounded-lg flex items-center justify-center gap-2 transition-all shadow-lg hover:shadow-xl"
                >
                  Sign In
                  <ArrowRight className="h-4 w-4" />
                </button>
              </form>

              <div className="text-center text-sm text-muted-foreground">
                Don't have an account?{' '}
                <button
                  onClick={() => setActiveTab('signup')}
                  className="text-primary hover:text-primary/80 font-medium"
                >
                  Sign up
                </button>
              </div>
            </TabsContent>

            {/* Sign Up Tab */}
            <TabsContent value="signup" className="mt-6 space-y-4">
              <div>
                <h2 className="text-2xl font-bold text-foreground">Create Account</h2>
                <p className="text-muted-foreground text-sm mt-1">Get started with YUGMĀSTRA today</p>
              </div>

              <form className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Full Name</label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="text"
                      placeholder="John Doe"
                      className="pl-10 h-11 bg-background border-border/60"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Email</label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="email"
                      placeholder="name@example.com"
                      className="pl-10 h-11 bg-background border-border/60"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Password</label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type={showPassword ? 'text' : 'password'}
                      placeholder="••••••••"
                      className="pl-10 pr-10 h-11 bg-background border-border/60"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                  <p className="text-xs text-muted-foreground">Must be at least 8 characters</p>
                </div>

                <label className="flex items-start gap-2 text-sm cursor-pointer">
                  <input type="checkbox" className="mt-0.5 rounded border-border/60" required />
                  <span className="text-muted-foreground">
                    I agree to the{' '}
                    <Link href="/terms" className="text-primary hover:text-primary/80">Terms of Service</Link>
                    {' '}and{' '}
                    <Link href="/privacy" className="text-primary hover:text-primary/80">Privacy Policy</Link>
                  </span>
                </label>

                <button
                  type="submit"
                  className="w-full h-11 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 hover:from-blue-600 hover:via-purple-600 hover:to-pink-600 text-white font-semibold rounded-lg flex items-center justify-center gap-2 transition-all shadow-lg hover:shadow-xl"
                >
                  Create Account
                  <ArrowRight className="h-4 w-4" />
                </button>
              </form>

              <div className="text-center text-sm text-muted-foreground">
                Already have an account?{' '}
                <button
                  onClick={() => setActiveTab('login')}
                  className="text-primary hover:text-primary/80 font-medium"
                >
                  Sign in
                </button>
              </div>
            </TabsContent>

            {/* Reset Password Tab */}
            <TabsContent value="reset" className="mt-6 space-y-4">
              <div>
                <h2 className="text-2xl font-bold text-foreground">Reset Password</h2>
                <p className="text-muted-foreground text-sm mt-1">Enter your email to receive a reset link</p>
              </div>

              <form className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Email Address</label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="email"
                      placeholder="name@example.com"
                      className="pl-10 h-11 bg-background border-border/60"
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    We'll send you an email with instructions to reset your password
                  </p>
                </div>

                <button
                  type="submit"
                  className="w-full h-11 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 hover:from-blue-600 hover:via-purple-600 hover:to-pink-600 text-white font-semibold rounded-lg flex items-center justify-center gap-2 transition-all shadow-lg hover:shadow-xl"
                >
                  Send Reset Link
                  <ArrowRight className="h-4 w-4" />
                </button>
              </form>

              <div className="text-center text-sm text-muted-foreground">
                Remember your password?{' '}
                <button
                  onClick={() => setActiveTab('login')}
                  className="text-primary hover:text-primary/80 font-medium"
                >
                  Sign in
                </button>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
