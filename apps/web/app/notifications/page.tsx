'use client';

import { useState } from 'react';
import { Bell, CheckCheck, Trash2, Filter, Search, Shield, Swords, Brain, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';

interface Notification {
  id: string;
  type: 'attack' | 'defense' | 'training' | 'system' | 'alert' | 'success';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

const mockNotifications: Notification[] = [
  {
    id: '1',
    type: 'attack',
    title: 'SQL Injection Detected',
    message: 'Blocked SQL injection attempt on /api/users endpoint from IP 192.168.1.45',
    timestamp: '2 minutes ago',
    read: false,
    priority: 'critical'
  },
  {
    id: '2',
    type: 'training',
    title: 'Model Training Complete',
    message: 'Red Team AI v2.3 training completed successfully. Accuracy improved by 12%.',
    timestamp: '1 hour ago',
    read: false,
    priority: 'medium'
  },
  {
    id: '3',
    type: 'defense',
    title: 'Firewall Rules Updated',
    message: 'New firewall rules applied to block suspicious traffic patterns.',
    timestamp: '3 hours ago',
    read: true,
    priority: 'low'
  },
  {
    id: '4',
    type: 'attack',
    title: 'DDoS Attack Mitigated',
    message: 'Successfully mitigated DDoS attack targeting main gateway. Peak traffic: 50K req/s.',
    timestamp: '5 hours ago',
    read: true,
    priority: 'high'
  },
  {
    id: '5',
    type: 'system',
    title: 'System Maintenance Scheduled',
    message: 'Scheduled maintenance on Jan 5, 2025 at 2:00 AM UTC. Expected downtime: 30 minutes.',
    timestamp: '1 day ago',
    read: false,
    priority: 'medium'
  },
  {
    id: '6',
    type: 'alert',
    title: 'Unusual Login Activity',
    message: 'Multiple failed login attempts detected from new location: Tokyo, Japan.',
    timestamp: '1 day ago',
    read: true,
    priority: 'high'
  },
  {
    id: '7',
    type: 'success',
    title: 'Vulnerability Patched',
    message: 'CVE-2024-1234 successfully patched across all systems.',
    timestamp: '2 days ago',
    read: true,
    priority: 'medium'
  },
  {
    id: '8',
    type: 'training',
    title: 'Blue Team AI Updated',
    message: 'Defense model updated with new attack patterns. Performance metrics improved.',
    timestamp: '2 days ago',
    read: true,
    priority: 'low'
  },
  {
    id: '9',
    type: 'attack',
    title: 'XSS Attempt Blocked',
    message: 'Cross-site scripting attack blocked on user input form.',
    timestamp: '3 days ago',
    read: true,
    priority: 'medium'
  },
  {
    id: '10',
    type: 'system',
    title: 'Backup Completed',
    message: 'Automated system backup completed successfully. Size: 2.4 GB.',
    timestamp: '3 days ago',
    read: true,
    priority: 'low'
  }
];

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState<Notification[]>(mockNotifications);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState('all');

  const getNotificationIcon = (type: Notification['type']) => {
    switch (type) {
      case 'attack':
        return <Swords className="h-5 w-5 text-red-500" />;
      case 'defense':
        return <Shield className="h-5 w-5 text-blue-500" />;
      case 'training':
        return <Brain className="h-5 w-5 text-purple-500" />;
      case 'alert':
        return <AlertTriangle className="h-5 w-5 text-orange-500" />;
      case 'success':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      default:
        return <Info className="h-5 w-5 text-gray-500" />;
    }
  };

  const getPriorityColor = (priority: Notification['priority']) => {
    switch (priority) {
      case 'critical':
        return 'bg-red-500/10 text-red-500 border-red-500/20';
      case 'high':
        return 'bg-orange-500/10 text-orange-500 border-orange-500/20';
      case 'medium':
        return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
      default:
        return 'bg-gray-500/10 text-gray-500 border-gray-500/20';
    }
  };

  const markAsRead = (id: string) => {
    setNotifications(notifications.map(n => n.id === id ? { ...n, read: true } : n));
  };

  const markAllAsRead = () => {
    setNotifications(notifications.map(n => ({ ...n, read: true })));
  };

  const deleteNotification = (id: string) => {
    setNotifications(notifications.filter(n => n.id !== id));
  };

  const filteredNotifications = notifications.filter(n => {
    const matchesSearch = n.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         n.message.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesTab = activeTab === 'all' ||
                      (activeTab === 'unread' && !n.read) ||
                      (activeTab === 'read' && n.read);
    return matchesSearch && matchesTab;
  });

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <div className="min-h-screen bg-background p-8 pt-32">
      <div className="max-w-7xl mx-auto">
        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 flex items-start gap-3 mb-6">
          <Bell className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong className="text-foreground">What this page does:</strong> View real-time security alerts and system notifications from YUGMASTRA's AI agents. Receive critical attack alerts, defense updates, training completion notices, system maintenance schedules, and suspicious activity warnings. Filter by read/unread status, search notifications, and manage alert preferences for email and push notifications.
            </p>
          </div>
        </div>
      </div>

      {/* Header */}
      <div className="border-b bg-card/50 backdrop-blur-xl sticky top-16 z-40">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg">
                <Bell className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 bg-clip-text text-transparent leading-[1.3]">
                  Notifications
                </h1>
                <p className="text-sm text-muted-foreground leading-[1.4]">
                  {unreadCount} unread notification{unreadCount !== 1 ? 's' : ''}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={markAllAsRead}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-primary/10 hover:bg-primary/20 text-primary font-semibold text-sm transition-all"
              >
                <CheckCheck className="h-4 w-4" />
                Mark all read
              </button>
            </div>
          </div>

          {/* Search and Filter */}
          <div className="flex items-center gap-3">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Search notifications..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-background/60 border-border/60 rounded-xl h-11"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <Tabs defaultValue="all" className="w-full" onValueChange={setActiveTab}>
          <TabsList className="bg-muted/40 border border-border/40 rounded-xl p-1.5 mb-6">
            <TabsTrigger value="all" className="rounded-lg data-[state=active]:bg-background data-[state=active]:shadow-md">
              All ({notifications.length})
            </TabsTrigger>
            <TabsTrigger value="unread" className="rounded-lg data-[state=active]:bg-background data-[state=active]:shadow-md">
              Unread ({unreadCount})
            </TabsTrigger>
            <TabsTrigger value="read" className="rounded-lg data-[state=active]:bg-background data-[state=active]:shadow-md">
              Read ({notifications.length - unreadCount})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="all" className="space-y-3">
            {filteredNotifications.length === 0 ? (
              <div className="text-center py-16">
                <Bell className="h-16 w-16 text-muted-foreground/30 mx-auto mb-4" />
                <p className="text-muted-foreground">No notifications found</p>
              </div>
            ) : (
              <ScrollArea className="h-[calc(100vh-320px)]">
                <div className="space-y-3 pr-4">
                  {filteredNotifications.map((notification) => (
                    <div
                      key={notification.id}
                      className={`group relative rounded-2xl border transition-all ${
                        notification.read
                          ? 'bg-card/30 border-border/30 hover:border-border/50'
                          : 'bg-card/80 border-border/60 hover:border-primary/30 shadow-sm'
                      }`}
                    >
                      {!notification.read && (
                        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-12 bg-gradient-to-b from-blue-500 via-purple-500 to-pink-500 rounded-r-full" />
                      )}

                      <div className="flex items-start gap-4 p-5 pl-6">
                        <div className="flex-shrink-0 mt-1">
                          {getNotificationIcon(notification.type)}
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex items-start justify-between gap-4 mb-2">
                            <div className="flex items-center gap-3">
                              <h3 className={`font-semibold leading-[1.4] ${notification.read ? 'text-foreground/70' : 'text-foreground'}`}>
                                {notification.title}
                              </h3>
                              <Badge className={`text-[10px] px-2 py-0.5 ${getPriorityColor(notification.priority)}`}>
                                {notification.priority.toUpperCase()}
                              </Badge>
                            </div>

                            <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                              {!notification.read && (
                                <button
                                  onClick={() => markAsRead(notification.id)}
                                  className="p-2 rounded-lg hover:bg-accent/50 text-muted-foreground hover:text-foreground transition-all"
                                  title="Mark as read"
                                >
                                  <CheckCheck className="h-4 w-4" />
                                </button>
                              )}
                              <button
                                onClick={() => deleteNotification(notification.id)}
                                className="p-2 rounded-lg hover:bg-red-500/10 text-muted-foreground hover:text-red-500 transition-all"
                                title="Delete"
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>
                            </div>
                          </div>

                          <p className={`text-sm leading-[1.5] mb-2 ${notification.read ? 'text-muted-foreground/60' : 'text-muted-foreground'}`}>
                            {notification.message}
                          </p>

                          <p className="text-xs text-muted-foreground/50 leading-[1.4]">
                            {notification.timestamp}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </TabsContent>

          <TabsContent value="unread" className="space-y-3">
            {filteredNotifications.length === 0 ? (
              <div className="text-center py-16">
                <CheckCircle className="h-16 w-16 text-green-500/30 mx-auto mb-4" />
                <p className="text-muted-foreground">No unread notifications</p>
              </div>
            ) : (
              <ScrollArea className="h-[calc(100vh-320px)]">
                <div className="space-y-3 pr-4">
                  {filteredNotifications.map((notification) => (
                    <div
                      key={notification.id}
                      className="group relative rounded-2xl border bg-card/80 border-border/60 hover:border-primary/30 shadow-sm transition-all"
                    >
                      <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-12 bg-gradient-to-b from-blue-500 via-purple-500 to-pink-500 rounded-r-full" />

                      <div className="flex items-start gap-4 p-5 pl-6">
                        <div className="flex-shrink-0 mt-1">
                          {getNotificationIcon(notification.type)}
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex items-start justify-between gap-4 mb-2">
                            <div className="flex items-center gap-3">
                              <h3 className="font-semibold text-foreground leading-[1.4]">
                                {notification.title}
                              </h3>
                              <Badge className={`text-[10px] px-2 py-0.5 ${getPriorityColor(notification.priority)}`}>
                                {notification.priority.toUpperCase()}
                              </Badge>
                            </div>

                            <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                              <button
                                onClick={() => markAsRead(notification.id)}
                                className="p-2 rounded-lg hover:bg-accent/50 text-muted-foreground hover:text-foreground transition-all"
                                title="Mark as read"
                              >
                                <CheckCheck className="h-4 w-4" />
                              </button>
                              <button
                                onClick={() => deleteNotification(notification.id)}
                                className="p-2 rounded-lg hover:bg-red-500/10 text-muted-foreground hover:text-red-500 transition-all"
                                title="Delete"
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>
                            </div>
                          </div>

                          <p className="text-sm text-muted-foreground leading-[1.5] mb-2">
                            {notification.message}
                          </p>

                          <p className="text-xs text-muted-foreground/50 leading-[1.4]">
                            {notification.timestamp}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </TabsContent>

          <TabsContent value="read" className="space-y-3">
            {filteredNotifications.length === 0 ? (
              <div className="text-center py-16">
                <Bell className="h-16 w-16 text-muted-foreground/30 mx-auto mb-4" />
                <p className="text-muted-foreground">No read notifications</p>
              </div>
            ) : (
              <ScrollArea className="h-[calc(100vh-320px)]">
                <div className="space-y-3 pr-4">
                  {filteredNotifications.map((notification) => (
                    <div
                      key={notification.id}
                      className="group relative rounded-2xl border bg-card/30 border-border/30 hover:border-border/50 transition-all"
                    >
                      <div className="flex items-start gap-4 p-5 pl-6">
                        <div className="flex-shrink-0 mt-1">
                          {getNotificationIcon(notification.type)}
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex items-start justify-between gap-4 mb-2">
                            <div className="flex items-center gap-3">
                              <h3 className="font-semibold text-foreground/70 leading-[1.4]">
                                {notification.title}
                              </h3>
                              <Badge className={`text-[10px] px-2 py-0.5 ${getPriorityColor(notification.priority)}`}>
                                {notification.priority.toUpperCase()}
                              </Badge>
                            </div>

                            <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                              <button
                                onClick={() => deleteNotification(notification.id)}
                                className="p-2 rounded-lg hover:bg-red-500/10 text-muted-foreground hover:text-red-500 transition-all"
                                title="Delete"
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>
                            </div>
                          </div>

                          <p className="text-sm text-muted-foreground/60 leading-[1.5] mb-2">
                            {notification.message}
                          </p>

                          <p className="text-xs text-muted-foreground/50 leading-[1.4]">
                            {notification.timestamp}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
