'use client';

import { useEffect, useState } from 'react';
import type { Notification } from '@/components/notifications/notification-panel';

let notificationId = 0;

export function sendNotification(
  type: Notification['type'],
  title: string,
  message: string,
  severity?: Notification['severity']
) {
  const notification: Notification = {
    id: `notification-${++notificationId}-${Date.now()}`,
    type,
    title,
    message,
    timestamp: new Date(),
    read: false,
    severity
  };

  // Dispatch custom event
  const event = new CustomEvent('yugmastra:notification', { detail: notification });
  window.dispatchEvent(event);

  // Also show browser notification if permission granted and settings allow it
  const settings = localStorage.getItem('yugmastra_settings');
  if (settings) {
    try {
      const parsed = JSON.parse(settings);
      if (parsed.pushNotifications && 'Notification' in window) {
        if (Notification.permission === 'granted') {
          new Notification(`YUGMĀSTRA: ${title}`, {
            body: message,
            icon: '/favicon.ico',
            badge: '/favicon.ico',
            tag: notification.id,
          });
        } else if (Notification.permission !== 'denied') {
          Notification.requestPermission().then(permission => {
            if (permission === 'granted') {
              new Notification(`YUGMĀSTRA: ${title}`, {
                body: message,
                icon: '/favicon.ico',
                badge: '/favicon.ico',
                tag: notification.id,
              });
            }
          });
        }
      }
    } catch (e) {
      console.error('Error sending browser notification:', e);
    }
  }

  return notification;
}

export function useNotifications() {
  const [unreadCount, setUnreadCount] = useState(0);

  useEffect(() => {
    // Count unread notifications on mount
    const updateUnreadCount = () => {
      const savedNotifications = localStorage.getItem('yugmastra_notifications');
      if (savedNotifications) {
        try {
          const notifications = JSON.parse(savedNotifications);
          const count = notifications.filter((n: Notification) => !n.read).length;
          setUnreadCount(count);
        } catch (e) {
          console.error('Error parsing notifications:', e);
        }
      }
    };

    updateUnreadCount();

    // Listen for new notifications
    const handleNewNotification = () => {
      updateUnreadCount();
    };

    window.addEventListener('yugmastra:notification', handleNewNotification);
    // Also listen for storage changes (from other tabs or notification panel)
    window.addEventListener('storage', updateUnreadCount);

    // Poll for updates (in case localStorage is updated directly)
    const interval = setInterval(updateUnreadCount, 2000);

    return () => {
      window.removeEventListener('yugmastra:notification', handleNewNotification);
      window.removeEventListener('storage', updateUnreadCount);
      clearInterval(interval);
    };
  }, []);

  return {
    unreadCount,
    sendNotification
  };
}
