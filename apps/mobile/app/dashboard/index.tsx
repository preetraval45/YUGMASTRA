import { View, Text, ScrollView, StyleSheet } from 'react-native';
import { Card } from 'react-native-paper';
import { Activity, Shield, Swords, TrendingUp } from 'lucide-react-native';

export default function MobileDashboard() {
  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Dashboard</Text>
        <Text style={styles.subtitle}>Real-time co-evolution metrics</Text>
      </View>

      <View style={styles.metricsGrid}>
        <Card style={styles.metricCard}>
          <Card.Content>
            <View style={styles.metricHeader}>
              <Activity color="#3b82f6" size={24} />
              <Text style={styles.metricChange}>+12</Text>
            </View>
            <Text style={styles.metricValue}>523</Text>
            <Text style={styles.metricLabel}>Total Episodes</Text>
          </Card.Content>
        </Card>

        <Card style={styles.metricCard}>
          <Card.Content>
            <View style={styles.metricHeader}>
              <Swords color="#ef4444" size={24} />
              <Text style={styles.metricChangeRed}>52%</Text>
            </View>
            <Text style={styles.metricValue}>271</Text>
            <Text style={styles.metricLabel}>Red Wins</Text>
          </Card.Content>
        </Card>

        <Card style={styles.metricCard}>
          <Card.Content>
            <View style={styles.metricHeader}>
              <Shield color="#3b82f6" size={24} />
              <Text style={styles.metricChangeBlue}>48%</Text>
            </View>
            <Text style={styles.metricValue}>252</Text>
            <Text style={styles.metricLabel}>Blue Wins</Text>
          </Card.Content>
        </Card>

        <Card style={styles.metricCard}>
          <Card.Content>
            <View style={styles.metricHeader}>
              <TrendingUp color="#8b5cf6" size={24} />
              <Text style={styles.metricStatus}>ACTIVE</Text>
            </View>
            <Text style={styles.metricValue}>Training</Text>
            <Text style={styles.metricLabel}>Evolution Phase</Text>
          </Card.Content>
        </Card>
      </View>

      <Card style={styles.activityCard}>
        <Card.Title title="Recent Activity" />
        <Card.Content>
          {[
            'Red agent discovered new attack path',
            'Blue agent updated detection rule',
            'Nash equilibrium distance decreased',
          ].map((activity, index) => (
            <View key={index} style={styles.activityItem}>
              <Activity size={16} color="#3b82f6" />
              <Text style={styles.activityText}>{activity}</Text>
            </View>
          ))}
        </Card.Content>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  header: {
    padding: 24,
    paddingTop: 48,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#93c5fd',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 16,
    gap: 16,
  },
  metricCard: {
    width: '47%',
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  metricHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  metricChange: {
    fontSize: 12,
    color: '#22c55e',
    fontWeight: '600',
  },
  metricChangeRed: {
    fontSize: 12,
    color: '#ef4444',
    fontWeight: '600',
  },
  metricChangeBlue: {
    fontSize: 12,
    color: '#3b82f6',
    fontWeight: '600',
  },
  metricStatus: {
    fontSize: 12,
    color: '#8b5cf6',
    fontWeight: '600',
  },
  metricValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  metricLabel: {
    fontSize: 12,
    color: '#cbd5e1',
  },
  activityCard: {
    margin: 16,
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  activityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 8,
  },
  activityText: {
    flex: 1,
    fontSize: 14,
    color: '#fff',
  },
});
