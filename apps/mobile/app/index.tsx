import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { Link } from 'expo-router';
import { Button, Card } from 'react-native-paper';

export default function Home() {
  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>YUGMĀSTRA</Text>
        <Text style={styles.subtitle}>
          Autonomous Adversary-Defender Co-Evolution Platform
        </Text>
      </View>

      <View style={styles.content}>
        <Link href="/dashboard" asChild>
          <Button
            mode="contained"
            style={styles.primaryButton}
            labelStyle={styles.buttonLabel}
          >
            Launch Dashboard
          </Button>
        </Link>

        <Card style={styles.card}>
          <Card.Title title="Red Team AI" titleStyle={styles.cardTitle} />
          <Card.Content>
            <Text style={styles.cardText}>
              Autonomous attack agent discovering novel exploitation strategies
              through reinforcement learning.
            </Text>
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Title title="Blue Team AI" titleStyle={styles.cardTitle} />
          <Card.Content>
            <Text style={styles.cardText}>
              Adaptive defense system learning detection patterns and generating
              countermeasures automatically.
            </Text>
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Title title="Co-Evolution" titleStyle={styles.cardTitle} />
          <Card.Content>
            <Text style={styles.cardText}>
              Multi-agent system where strategies emerge through adversarial
              competition.
            </Text>
          </Card.Content>
        </Card>
      </View>
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
    alignItems: 'center',
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#93c5fd',
    textAlign: 'center',
    paddingHorizontal: 16,
  },
  content: {
    padding: 16,
  },
  primaryButton: {
    marginBottom: 24,
    backgroundColor: '#2563eb',
  },
  buttonLabel: {
    fontSize: 16,
    paddingVertical: 8,
  },
  card: {
    marginBottom: 16,
    backgroundColor: '#1e293b',
  },
  cardTitle: {
    color: '#fff',
  },
  cardText: {
    color: '#cbd5e1',
    fontSize: 14,
  },
});
